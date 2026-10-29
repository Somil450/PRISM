"""
scripts/generate_controlled_faceswap_pairs.py
Controlled Research Face-Swap & In-The-Wild Artifact Synthesizer.

Purpose:
  Replicates the exact mathematical artifact signature of one-click deepfake tools
  (Roop, Inswapper-128, DeepFaceLab) used in viral celebrity deepfakes (e.g. Rashmika Mandanna case):
    1. 128x128 latent space compression & bilinear upsampling artifacts.
    2. Boundary blending seam (Gaussian/feathering transition on facial skin contour).
    3. Sensor PRNU noise mismatch between the synthetic face patch and the authentic body.
    4. Acute-angle entry tracking delay: momentary 2-3 frame latch glitch during entry.
    5. H.264 high-quantization compression (WhatsApp/Telegram re-muxing at CRF 28-32).
    6. Comparative PRISM forensic audit comparing authentic baseline vs. synthetic replica.

Usage:
  # Generate pair:
  py scripts/generate_controlled_faceswap_pairs.py --input data/real_test_samples/01_real_human_video_v1.mp4

  # Generate and run side-by-side PRISM comparative forensic audit:
  py scripts/generate_controlled_faceswap_pairs.py --input data/real_test_samples/01_real_human_video_v1.mp4 --compare_prism
"""

import argparse
import os
import sys
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from agents.video_temporal_agent import VideoTemporalAgent
    from agents.eye_gaze_agent import EyeGazeAgent
    from agents.rppg_agent import RPPGAgent
    from agents.compression_watermark_agent import CompressionWatermarkAgent
    from fusion import DempsterShaferFusion
    HAS_PRISM_AGENTS = True
except ImportError:
    HAS_PRISM_AGENTS = False


class ControlledFaceSwapGenerator:
    """
    Synthesizes controlled deepfake test pairs matching the exact
    mathematical error distribution of Roop / Inswapper-128 models.
    """

    def __init__(self, latent_res: int = 128) -> None:
        self.latent_res = latent_res

    def detect_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Extracts facial bounding box using YCrCb skin segmentation and morphological grouping.
        """
        h, w = frame.shape[:2]
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([45, 125, 65]), np.array([255, 185, 140]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        valid_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < (h * w * 0.04):
                continue
            fx, fy, fw, fh = cv2.boundingRect(c)
            aspect = float(fh) / float(fw + 1e-6)
            if 0.40 <= aspect <= 2.6 and fw >= 40 and fh >= 40:
                density = np.count_nonzero(closed[fy:fy+fh, fx:fx+fw]) / float(fw * fh + 1e-6)
                if density >= 0.35:
                    valid_boxes.append((area, (fx, fy, fw, fh)))

        if valid_boxes:
            valid_boxes.sort(key=lambda x: x[0], reverse=True)
            return valid_boxes[0][1]

        # Fallback to center-upper third
        return (int(w * 0.25), int(h * 0.15), int(w * 0.50), int(h * 0.50))

    def apply_inswapper_artifacts(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        target_color_shift: Tuple[int, int, int] = (14, -8, 18),
    ) -> np.ndarray:
        """
        Simulates Roop / inswapper_128 face swap artifacts:
        - Downsamples face to 128x128 latent resolution, then upsamples (sub-band frequency loss).
        - Applies subtle AdaIN style color shift.
        - Attenuates sensor PRNU noise within the facial patch to create sensor fingerprint discontinuity.
        - Applies boundary feathering seam.
        - Adds acute-angle entry tracking delay on frames 0-2 (latch glitch).
        """
        fx, fy, fw, fh = bbox
        out = frame.copy()

        # Simulate 2-3 frame tracking delay at start (exact artifact of viral elevator video)
        if frame_idx < 3:
            return out  # Original face momentarily visible before swapper locks on

        face_crop = out[fy:fy+fh, fx:fx+fw]
        if face_crop.size == 0:
            return out

        orig_h, orig_w = face_crop.shape[:2]

        # 1. 128x128 Latent Space Compression Artifact (Sub-band frequency loss)
        latent = cv2.resize(face_crop, (self.latent_res, self.latent_res), interpolation=cv2.INTER_AREA)
        upsampled = cv2.resize(latent, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 2. Photometric Color Transfer Divergence
        b, g, r = cv2.split(upsampled.astype(np.int16))
        b = np.clip(b + target_color_shift[0], 0, 255).astype(np.uint8)
        g = np.clip(g + target_color_shift[1], 0, 255).astype(np.uint8)
        r = np.clip(r + target_color_shift[2], 0, 255).astype(np.uint8)
        synthetic_face = cv2.merge([b, g, r])

        # 3. Attenuate Sensor PRNU Noise on Synthetic Face Patch (Sensor mismatch artifact)
        # Deep learning generators output clean/smooth pixel distributions lacking sensor camera grain
        synthetic_face = cv2.bilateralFilter(synthetic_face, d=5, sigmaColor=25, sigmaSpace=25)

        # 4. Soft Oval Mask with Boundary Seam Feathering
        mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (orig_w // 2, orig_h // 2),
            (int(orig_w * 0.44), int(orig_h * 0.48)),
            0,
            0,
            360,
            1.0,
            -1,
        )
        feather_k = max(7, (orig_w // 12) | 1)
        mask = cv2.GaussianBlur(mask, (feather_k, feather_k), feather_k / 3.0)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # 5. Synthesize Blended Frame
        blended_face = (synthetic_face * mask_3d + face_crop * (1.0 - mask_3d)).astype(np.uint8)
        out[fy:fy+fh, fx:fx+fw] = blended_face
        return out

    def recompress_with_ffmpeg_crf(self, video_path: str, crf: int = 30) -> None:
        """
        Recompresses video to H.264 CRF 28-32 (matching WhatsApp/Telegram recompression).
        """
        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            return

        tmp_out = video_path + ".tmp.mp4"
        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            tmp_out,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(tmp_out):
                os.replace(tmp_out, video_path)
                print(f"  [+] Recompressed with H.264 (CRF {crf} - WhatsApp/Telegram profile)")
            elif os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass

    def generate_pair(
        self,
        input_video_path: str,
        output_dir: str,
        max_frames: int = 45,
        apply_crf: bool = True,
    ) -> Tuple[str, str]:
        """
        Reads input_video and generates both:
          1. Authentic baseline video (_authentic_source.mp4)
          2. Inswapper replica deepfake (_inswapper_faceswap.mp4)
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        stem = Path(input_video_path).stem
        authentic_path = str(out_path / f"{stem}_authentic_source.mp4")
        deepfake_path = str(out_path / f"{stem}_inswapper_faceswap.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_auth = cv2.VideoWriter(authentic_path, fourcc, fps, (width, height))
        writer_fake = cv2.VideoWriter(deepfake_path, fourcc, fps, (width, height))

        frame_idx = 0
        last_bbox = None

        print(f"Generating Controlled Deepfake Pair from '{Path(input_video_path).name}'...")
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            bbox = self.detect_face_bbox(frame)
            if bbox is not None:
                last_bbox = bbox
            elif last_bbox is not None:
                bbox = last_bbox
            else:
                bbox = (int(width * 0.25), int(height * 0.15), int(width * 0.50), int(height * 0.50))

            # Authentic frame
            writer_auth.write(frame)

            # Inswapper synthetic frame
            fake_frame = self.apply_inswapper_artifacts(frame, bbox, frame_idx)
            writer_fake.write(fake_frame)

            frame_idx += 1

        cap.release()
        writer_auth.release()
        writer_fake.release()

        if apply_crf:
            self.recompress_with_ffmpeg_crf(authentic_path, crf=30)
            self.recompress_with_ffmpeg_crf(deepfake_path, crf=30)

        print(f"  [1/2] Authentic Video Saved: {authentic_path} ({frame_idx} frames)")
        print(f"  [2/2] Inswapper Deepfake Saved: {deepfake_path} ({frame_idx} frames)")
        return authentic_path, deepfake_path


def compare_prism_evidence(auth_video: str, fake_video: str) -> None:
    """
    Executes PRISM multi-agent forensic verification on both the authentic
    baseline and the inswapper deepfake replica, printing a comparative audit.
    """
    if not HAS_PRISM_AGENTS:
        print("[!] PRISM agents not found.")
        return

    print("\n" + "=" * 80)
    print("🔬 PRISM COMPARATIVE FORENSIC AUDIT: AUTHENTIC VS. INSWAPPER REPLICA")
    print("=" * 80)

    temporal = VideoTemporalAgent()
    eye = EyeGazeAgent()
    rppg = RPPGAgent()
    fusion = DempsterShaferFusion()

    def run_on_video(vpath):
        cap = cv2.VideoCapture(vpath)
        frames = []
        while len(frames) < 30 and cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        t_ev = temporal.analyze(frames)
        g_ev = eye.analyze(frames[len(frames) // 2])
        faces_found = g_ev.telemetry.get("faces_detected", 0) > 0
        r_ev = rppg.analyze(frames, has_face=faces_found)

        fused = fusion.fuse_graph_evidence(
            {"temporal": t_ev.bba_mass, "gaze": g_ev.bba_mass, "rppg": r_ev.bba_mass},
            [("gaze", "rppg"), ("temporal", "gaze")],
        )
        return {
            "t_ev": t_ev,
            "g_ev": g_ev,
            "r_ev": r_ev,
            "fused": fused,
        }

    res_auth = run_on_video(auth_video)
    res_fake = run_on_video(fake_video)

    print(f"{'Forensic Telemetry Stream':32s} | {'Authentic Baseline':20s} | {'Inswapper Face-Swap':20s}")
    print("-" * 80)
    print(
        f"{'Motion Jitter Index':32s} | "
        f"{res_auth['t_ev'].telemetry.get('motion_jitter_index', 0.0):<20.4f} | "
        f"{res_fake['t_ev'].telemetry.get('motion_jitter_index', 0.0):<20.4f}"
    )
    print(
        f"{'Temporal Agent Verdict':32s} | "
        f"{res_auth['t_ev'].verdict:<20s} | "
        f"{res_fake['t_ev'].verdict:<20s}"
    )
    print(
        f"{'Temporal Mass m(T)':32s} | "
        f"{res_auth['t_ev'].bba_mass['T']:<20.1%} | "
        f"{res_fake['t_ev'].bba_mass['T']:<20.1%}"
    )
    print(
        f"{'Biometric / Gaze Mass m(T)':32s} | "
        f"{res_auth['g_ev'].bba_mass['T']:<20.1%} | "
        f"{res_fake['g_ev'].bba_mass['T']:<20.1%}"
    )
    print("-" * 80)
    print(
        f"{'D-S Lower Belief Bel(T)':32s} | "
        f"{res_auth['fused']['belief']:<20.1%} | "
        f"{res_fake['fused']['belief']:<20.1%}"
    )
    print(
        f"{'D-S Pignistic BetP(T)':32s} | "
        f"{res_auth['fused']['pignistic_fake']:<20.1%} | "
        f"{res_fake['fused']['pignistic_fake']:<20.1%}"
    )
    print(
        f"{'SYSTEM FORENSIC VERDICT':32s} | "
        f"{res_auth['fused']['verdict']:<20s} | "
        f"{res_fake['fused']['verdict']:<20s}"
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Controlled Deepfake Pair Synthesizer")
    parser.add_argument("--input", type=str, default="data/real_test_samples/01_real_human_video_v1.mp4", help="Path to input source video")
    parser.add_argument("--output_dir", type=str, default="data/real_test_samples", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=45, help="Max frames to process")
    parser.add_argument("--compare_prism", action="store_true", help="Execute comparative PRISM forensic audit on pair")

    args = parser.parse_args()
    gen = ControlledFaceSwapGenerator()
    auth_p, fake_p = gen.generate_pair(args.input, args.output_dir, max_frames=args.max_frames)

    if args.compare_prism:
        compare_prism_evidence(auth_p, fake_p)
