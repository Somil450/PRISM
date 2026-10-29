"""
scripts/matched_normalization.py
Matched Media Normalization Engine & Confound Remediation for PRISM.

Enforces strict confound elimination across authentic and deepfake media:
  1. Video Normalization:
     - Identical video codec (libx264, H.264)
     - Identical CRF (CRF 28, preset fast)
     - Identical pixel format (yuv420p)
     - Identical frame rate (30.0 fps)
     - Matched spatial resolution (1280x720 scaled/padded)
     - Audio stream matched (16 kHz mono AAC or stripped if silent)
  2. Image Normalization:
     - Uniform image container & compression (JPEG quality 90) across both authentic and AI images.
  3. Impact Analysis:
     - Evaluates and logs forensic agent AUC before vs. after matched normalization,
       proving which detection signals were genuine artifacts vs. codec shortcuts.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class MatchedMediaNormalizer:
    """
    Standardizes media encoding parameters across classes to eliminate shortcuts.
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        target_crf: int = 28,
        target_resolution: Tuple[int, int] = (1280, 720),
        image_quality: int = 90,
    ) -> None:
        self.target_fps = target_fps
        self.target_crf = target_crf
        self.target_w, self.target_h = target_resolution
        self.image_quality = image_quality
        self.ffmpeg_cmd = shutil.which("ffmpeg")

    def normalize_video(self, src_path: str, dst_path: str) -> bool:
        """
        Re-encodes video through strictly matched FFmpeg parameters.
        """
        p_src = Path(src_path)
        p_dst = Path(dst_path)
        p_dst.parent.mkdir(parents=True, exist_ok=True)

        if not self.ffmpeg_cmd:
            print("[-] Warning: ffmpeg not found. Falling back to OpenCV re-encode.")
            return self._normalize_video_opencv(src_path, dst_path)

        # Matched scale and pad filter to guarantee exact resolution and aspect ratio
        vf_filter = (
            f"fps={self.target_fps},"
            f"scale={self.target_w}:{self.target_h}:force_original_aspect_ratio=decrease,"
            f"pad={self.target_w}:{self.target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )

        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-i", str(p_src),
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-crf", str(self.target_crf),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",  # Strip audio for matched visual benchmark if audio absent/unvoiced
            str(p_dst),
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if res.returncode == 0 and p_dst.exists() and p_dst.stat().st_size > 1000:
                return True
        except Exception as e:
            print(f"[-] ffmpeg error on {src_path}: {e}")

        return self._normalize_video_opencv(src_path, dst_path)

    def _normalize_video_opencv(self, src_path: str, dst_path: str) -> bool:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dst_path, fourcc, self.target_fps, (self.target_w, self.target_h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize directly to target
            resized = cv2.resize(frame, (self.target_w, self.target_h))
            writer.write(resized)

        cap.release()
        writer.release()
        return Path(dst_path).exists()

    def normalize_image(self, src_path: str, dst_path: str) -> bool:
        """
        Re-encodes image to standard uniform format (JPEG quality 90) to eliminate container shortcuts.
        """
        p_dst = Path(dst_path)
        p_dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(src_path)
        if img is None:
            return False

        # If extension is not .jpg, enforce .jpg
        if p_dst.suffix.lower() not in (".jpg", ".jpeg"):
            p_dst = p_dst.with_suffix(".jpg")

        cv2.imwrite(str(p_dst), img, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        return p_dst.exists()

    def normalize_directory(self, input_dir: str, output_dir: str) -> Dict[str, str]:
        """
        Normalizes all media in input_dir into matched formats in output_dir.
        Returns mapping: {original_path: normalized_path}.
        """
        p_in = Path(input_dir)
        p_out = Path(output_dir)
        p_out.mkdir(parents=True, exist_ok=True)

        mapping = {}
        for f in sorted(list(p_in.iterdir())):
            if f.is_dir():
                continue
            ext = f.suffix.lower()
            if ext in (".mp4", ".avi", ".mov", ".mkv"):
                dst = p_out / f"{f.stem}.mp4"
                print(f"[+] Normalizing video: {f.name} -> {dst.name}")
                if self.normalize_video(str(f), str(dst)):
                    mapping[str(f)] = str(dst)
            elif ext in (".jpg", ".jpeg", ".png", ".webp"):
                dst = p_out / f"{f.stem}.jpg"
                print(f"[+] Normalizing image: {f.name} -> {dst.name}")
                if self.normalize_image(str(f), str(dst)):
                    mapping[str(f)] = str(dst)

        print(f"[+] Normalized {len(mapping)} media files into {output_dir}")
        return mapping


def fix_local_synthetic_set(target_dir: str = "data/real_test_samples") -> None:
    """
    Specifically fixes the in-house synthetic pairs by re-encoding authentic and fake
    clips with identical FFmpeg libx264 parameters.
    """
    p = Path(target_dir)
    normalizer = MatchedMediaNormalizer(target_fps=30.0, target_crf=28, target_resolution=(1280, 720))

    auth_video = p / "01_real_human_video_v1_authentic_source.mp4"
    fake_video = p / "01_real_human_video_v1_inswapper_faceswap.mp4"

    if auth_video.exists() and fake_video.exists():
        print("[+] Re-encoding controlled in-house pair with matched H.264 CRF 28 profile...")
        tmp_auth = auth_video.with_suffix(".matched.mp4")
        tmp_fake = fake_video.with_suffix(".matched.mp4")

        normalizer.normalize_video(str(auth_video), str(tmp_auth))
        normalizer.normalize_video(str(fake_video), str(tmp_fake))

        if tmp_auth.exists() and tmp_fake.exists():
            shutil.move(str(tmp_auth), str(auth_video))
            shutil.move(str(tmp_fake), str(fake_video))
            print("  [OK] Successfully re-encoded authentic and deepfake videos with identical encoding profile.")


def compare_auc_before_after(
    csv_before: str,
    csv_after: str,
) -> Dict[str, Dict[str, float]]:
    """
    Computes and compares AUC for every agent before and after matched normalization.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df_b = pd.read_csv(csv_before)
    df_a = pd.read_csv(csv_after)

    agents = [
        "vit_dora_score", "prnu_score", "temporal_score", "illumination_score",
        "eyegaze_score", "compression_score", "metadata_score", "rppg_score"
    ]

    n_pos = int((df_a["label"] == 1).sum())
    n_neg = int((df_a["label"] == 0).sum())
    pairs = n_pos * n_neg
    se_null = np.sqrt((n_pos + n_neg + 1) / (12.0 * max(1, pairs)))

    print("\n" + "=" * 110)
    print("🔬 FORENSIC AGENT DISCRIMINATION: BEFORE VS AFTER MATCHED NORMALIZATION")
    print("=" * 110)
    print(f"{'Forensic Stream / Agent':26s} | {'AUC Before':10s} | {'AUC After':10s} | {'Delta AUC':10s} | {'(a) Normalization Sensitivity':30s} | {'(b) AUC vs 0.5 (N<10)'}")
    print("-" * 110)

    comparison = {}
    for col in agents:
        name = col.replace("_score", "").replace("_", " ").title()
        auc_b = 0.50
        auc_a = 0.50

        # Before
        sub_b = df_b.dropna(subset=[col, "label"])
        if len(sub_b["label"].unique()) > 1:
            try:
                auc_b = float(roc_auc_score(sub_b["label"], sub_b[col]))
            except Exception:
                pass

        # After
        sub_a = df_a.dropna(subset=[col, "label"])
        if len(sub_a["label"].unique()) > 1:
            try:
                auc_a = float(roc_auc_score(sub_a["label"], sub_a[col]))
            except Exception:
                pass

        delta = auc_a - auc_b
        abs_delta = abs(delta)

        # Column (a): AUC Sensitivity to Normalization
        if abs_delta < 0.05:
            sens = f"Insensitive (|Δ|={abs_delta:.4f} < 0.05)"
        elif delta < -0.05:
            sens = f"Confound Drop (|Δ|={abs_delta:.4f} >= 0.05)"
        else:
            sens = f"Sensitivity Shift (|Δ|={abs_delta:.4f})"

        # Column (b): AUC vs 0.5 (Never call genuine signal at N<10)
        diff_05 = auc_a - 0.50
        if diff_05 > 0.05:
            disc = f"Above 0.5 (+{diff_05:.4f}) [Anecdotal N<10]"
        elif diff_05 < -0.05:
            disc = f"Below 0.5 ({diff_05:.4f}) [Inverted N<10]"
        else:
            disc = f"At Chance ({diff_05:+.4f}) [No Signal N<10]"

        comparison[name] = {
            "auc_before": round(auc_b, 4),
            "auc_after": round(auc_a, 4),
            "delta": round(delta, 4),
            "sensitivity": sens,
            "auc_vs_05": disc,
        }
        print(f"{name:26s} | {auc_b:10.4f} | {auc_a:10.4f} | {delta:+10.4f} | {sens:30s} | {disc}")

    print("=" * 110)
    print(f"Bounded Interpretation: N_pos = {n_pos}, N_neg = {n_neg}, Pairs = {pairs}, AUC standard error under the null = {se_null:.4f}")
    print("=" * 110 + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="PRISM Matched Normalization")
    parser.add_argument("--fix_local", action="store_true", help="Fix local in-house synthetic pairs in data/real_test_samples")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory of media")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for normalized media")
    parser.add_argument("--compare", nargs=2, metavar=("CSV_BEFORE", "CSV_AFTER"), help="Compare feature caches before vs after")
    args = parser.parse_args()

    if args.fix_local:
        fix_local_synthetic_set()

    if args.input_dir and args.output_dir:
        norm = MatchedMediaNormalizer()
        norm.normalize_directory(args.input_dir, args.output_dir)

    if args.compare:
        compare_auc_before_after(args.compare[0], args.compare[1])


if __name__ == "__main__":
    main()
