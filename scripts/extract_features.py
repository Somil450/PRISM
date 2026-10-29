"""
scripts/extract_features.py
PRISM Multi-Agent Feature Extraction and Caching Engine.

Runs every PRISM forensic agent across a manifest or dataset directory and extracts
standardized forensic signals and metadata into a tabular feature cache (CSV).

One CSV row per media item:
  - id, path, label, identity, generator, compression
  - Per-agent scores & Dempster-Shafer masses m(T), m(A), m(U) for 10 agents:
      * vit_dora (Spatial ViT with DoRA)
      * prnu (Sensor PRNU noise residual)
      * temporal (Optical flow jitter)
      * illumination (Photometric gradient)
      * eyegaze (Pupil gaze disparity)
      * compression (ELA / DCT grid)
      * metadata (EXIF/C2PA/provenance)
      * rppg (Biological pulse signal)
      * audio (Acoustic vocoder artifacts)
      * lipsync (Cross-modal phoneme-viseme synchrony)
  - Container / Codec / Bitrate / Metadata features:
      * container, codec, bitrate_kbps, resolution_w, resolution_h, fps, duration_sec, file_size_bytes

Supports running on Kaggle dual-GPU/CPU and local workstations with resumption.
All downstream fusion models and shortcut analyses consume this cached CSV.
"""

import os
import sys
import argparse
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure src is on python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dataset_manifest import DatasetManifestLoader, create_manifest_entry
from preprocessor import PRISMPreprocessor
from fusion import DempsterShaferFusion
from model import PRISMNetwork
from agents import (
    MetadataAgent,
    EyeGazeAgent,
    CompressionWatermarkAgent,
    AudioVoiceAgent,
    VideoTemporalAgent,
    IlluminationAgent,
    LipSyncAgent,
    RPPGAgent,
)

FEATURE_CSV_COLUMNS = [
    # Provenance and identity metadata
    "id",
    "path",
    "label",
    "identity",
    "generator",
    "compression",
    # Agent 1: Spatial ViT (DoRA)
    "vit_dora_score", "vit_dora_m_T", "vit_dora_m_A", "vit_dora_m_U",
    # Agent 2: PRNU Sensor Noise
    "prnu_score", "prnu_m_T", "prnu_m_A", "prnu_m_U",
    # Agent 3: Video Temporal Jitter
    "temporal_score", "temporal_m_T", "temporal_m_A", "temporal_m_U",
    # Agent 4: Illumination Consistency
    "illumination_score", "illumination_m_T", "illumination_m_A", "illumination_m_U",
    # Agent 5: Eye Gaze Disparity
    "eyegaze_score", "eyegaze_m_T", "eyegaze_m_A", "eyegaze_m_U",
    # Agent 6: Compression & ELA
    "compression_score", "compression_m_T", "compression_m_A", "compression_m_U",
    # Agent 7: Container Metadata
    "metadata_score", "metadata_m_T", "metadata_m_A", "metadata_m_U",
    # Agent 8: RPPG Biological Pulse
    "rppg_score", "rppg_m_T", "rppg_m_A", "rppg_m_U",
    # Agent 9: Audio Voice Artifacts
    "audio_score", "audio_m_T", "audio_m_A", "audio_m_U",
    # Agent 10: Lip-Sync Cross-Modal
    "lipsync_score", "lipsync_m_T", "lipsync_m_A", "lipsync_m_U",
    # Container / Codec shortcut features
    "container",
    "codec",
    "bitrate_kbps",
    "resolution_w",
    "resolution_h",
    "fps",
    "duration_sec",
    "file_size_bytes",
]


class FeatureExtractor:
    """
    Orchestrates all 10 forensic agents and extracts metadata attributes.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[+] Initializing PRISM Feature Extractor on device: {self.device}")

        # Core components
        self.preprocessor = PRISMPreprocessor()
        self.fusion = DempsterShaferFusion()

        # Load ViT model if weights exist
        self.model: Optional[PRISMNetwork] = None
        candidate_weights = [
            weights_path,
            str(REPO_ROOT / "prism_model_final.pth"),
            str(REPO_ROOT / "prism_model_final (2).pth"),
            str(REPO_ROOT / "models" / "checkpoints" / "prism_model_best.pth"),
        ]
        w_path = None
        for cw in candidate_weights:
            if cw and os.path.exists(cw):
                w_path = cw
                break

        if w_path is not None:
            try:
                self.model = PRISMNetwork(pretrained=False)
                state = torch.load(w_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state, strict=False)
                self.model.to(self.device)
                self.model.eval()
                print(f"[+] Loaded PRISM ViT checkpoint: {w_path}")
            except Exception as e:
                print(f"[-] Warning: Failed to load weights from {w_path}: {e}")
                self.model = None
        else:
            print("[-] Notice: Checkpoint not found. Running with neutral ViT priors.")

        # Standalone Agents
        self.meta_agent = MetadataAgent()
        self.gaze_agent = EyeGazeAgent()
        self.comp_agent = CompressionWatermarkAgent()
        self.audio_agent = AudioVoiceAgent()
        self.temporal_agent = VideoTemporalAgent()
        self.illum_agent = IlluminationAgent()
        self.lipsync_agent = LipSyncAgent()
        self.rppg_agent = RPPGAgent()

    def probe_metadata(self, media_path: str) -> Dict[str, Any]:
        """Extracts container, codec, bitrate, resolution, and fps metrics."""
        p = Path(media_path)
        ext = p.suffix.lower().lstrip(".")
        file_size = p.stat().st_size if p.exists() else 0

        info = {
            "container": ext if ext else "unknown",
            "codec": "unknown",
            "bitrate_kbps": 0.0,
            "resolution_w": 0,
            "resolution_h": 0,
            "fps": 0.0,
            "duration_sec": 0.0,
            "file_size_bytes": file_size,
        }

        if ext in ("mp4", "avi", "mov", "mkv", "webm"):
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = float(n_frames / fps) if fps > 0 else 0.0
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec_tag = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

                bitrate = 0.0
                if dur > 0 and file_size > 0:
                    bitrate = float((file_size * 8.0) / (dur * 1000.0))

                info["resolution_w"] = w
                info["resolution_h"] = h
                info["fps"] = round(fps, 2)
                info["duration_sec"] = round(dur, 3)
                info["bitrate_kbps"] = round(bitrate, 1)
                info["codec"] = codec_tag if codec_tag else "h264"
                cap.release()
        elif ext in ("jpg", "jpeg", "png", "webp", "bmp", "tiff"):
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                info["resolution_w"] = w
                info["resolution_h"] = h
                info["codec"] = f"image_{ext}"
                info["fps"] = 0.0
                info["duration_sec"] = 0.0
                info["bitrate_kbps"] = 0.0

        return info

    def extract_audio_stream(self, video_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Extracts mono 16 kHz audio waveform from video via ffmpeg."""
        import shutil
        import subprocess
        import io
        from scipy.io import wavfile

        ffmpeg_cmd = shutil.which("ffmpeg")
        if ffmpeg_cmd and Path(video_path).exists():
            try:
                proc = subprocess.run(
                    [ffmpeg_cmd, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=15,
                )
                if proc.stdout and len(proc.stdout) > 44:
                    sr, data = wavfile.read(io.BytesIO(proc.stdout))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if data.dtype == np.int16:
                        wave = (data / 32768.0).astype(np.float32)
                    elif data.dtype == np.int32:
                        wave = (data / 2147483648.0).astype(np.float32)
                    else:
                        wave = data.astype(np.float32)
                    return wave, int(sr)
            except Exception:
                pass
        return None, 16000

    def process_item(self, manifest_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs all 10 forensic agents and metadata extraction on a single manifest entry.
        """
        path_str = manifest_entry["path"]
        p = Path(path_str)
        if not p.is_absolute():
            p = REPO_ROOT / p

        label = int(manifest_entry.get("label", 0))
        item_id = manifest_entry.get("id", p.name)
        identity = manifest_entry.get("identity", p.stem.split("_")[0])
        generator = manifest_entry.get("generator", "original" if label == 0 else "unknown")
        compression = manifest_entry.get("compression", "unknown")

        meta_info = self.probe_metadata(str(p))

        # Default results dictionary
        res: Dict[str, Any] = {
            "id": item_id,
            "path": str(p.as_posix()),
            "label": label,
            "identity": identity,
            "generator": generator,
            "compression": compression,
            # Metadata features
            "container": meta_info["container"],
            "codec": meta_info["codec"],
            "bitrate_kbps": meta_info["bitrate_kbps"],
            "resolution_w": meta_info["resolution_w"],
            "resolution_h": meta_info["resolution_h"],
            "fps": meta_info["fps"],
            "duration_sec": meta_info["duration_sec"],
            "file_size_bytes": meta_info["file_size_bytes"],
        }

        # Initialize all agent scores & masses to default/abstaining
        for agent_prefix in [
            "vit_dora", "prnu", "temporal", "illumination", "eyegaze",
            "compression", "metadata", "rppg", "audio", "lipsync"
        ]:
            res[f"{agent_prefix}_score"] = 0.50
            res[f"{agent_prefix}_m_T"] = 0.0
            res[f"{agent_prefix}_m_A"] = 0.0
            res[f"{agent_prefix}_m_U"] = 1.0

        if not p.exists():
            return res

        ext = p.suffix.lower().lstrip(".")
        is_video = ext in ("mp4", "avi", "mov", "mkv", "webm")
        is_image = ext in ("jpg", "jpeg", "png", "webp", "bmp")

        # 1. Metadata Agent (analyzes in-memory bytes only, strictly zero path leakage)
        try:
            with open(p, "rb") as f:
                raw_bytes = f.read()
            m_ev = self.meta_agent.analyze(raw_bytes)
            res["metadata_score"] = float(round(m_ev.prob_tampered, 4))
            res["metadata_m_T"] = float(round(m_ev.bba_mass.get("T", 0.0), 4))
            res["metadata_m_A"] = float(round(m_ev.bba_mass.get("A", 0.0), 4))
            res["metadata_m_U"] = float(round(m_ev.bba_mass.get("U", 1.0), 4))
        except Exception:
            pass

        frames: List[np.ndarray] = []
        keyframe: Optional[np.ndarray] = None

        if is_image:
            img = cv2.imread(str(p))
            if img is not None:
                keyframe = img
                frames = [img]
        elif is_video:
            cap = cv2.VideoCapture(str(p))
            while len(frames) < 30 and cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f)
            cap.release()
            if frames:
                keyframe = frames[len(frames) // 2]

        if keyframe is not None:
            # 2. Spatial ViT (DoRA) & 3. PRNU Sensor Noise
            prob_vit = 0.50
            prob_prnu = 0.50
            try:
                rgb_t, noise_t = self.preprocessor.process_frame(keyframe)
                if self.model is not None:
                    rgb_b = rgb_t.unsqueeze(0).to(self.device)
                    noise_b = noise_t.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.model(rgb_b, noise_b)
                    prob_vit = float(torch.sigmoid(out["logit"]).cpu().item())
                else:
                    prob_vit = 0.50

                raw_noise = float(torch.mean(torch.abs(noise_t)).item())
                if raw_noise < 0.015:
                    prob_prnu = 0.70
                elif raw_noise > 0.45:
                    prob_prnu = 0.75
                else:
                    prob_prnu = 0.12
            except Exception:
                prob_vit = 0.50
                prob_prnu = 0.50

            vit_bba = self.fusion.create_mass(evidence_T=prob_vit * 8.8, evidence_A=(1.0 - prob_vit) * 8.8)
            prnu_bba = self.fusion.create_mass(evidence_T=prob_prnu * 6.0, evidence_A=(1.0 - prob_prnu) * 6.0)

            res["vit_dora_score"] = float(round(prob_vit, 4))
            res["vit_dora_m_T"] = float(round(vit_bba.get("T", 0.0), 4))
            res["vit_dora_m_A"] = float(round(vit_bba.get("A", 0.0), 4))
            res["vit_dora_m_U"] = float(round(vit_bba.get("U", 1.0), 4))

            res["prnu_score"] = float(round(prob_prnu, 4))
            res["prnu_m_T"] = float(round(prnu_bba.get("T", 0.0), 4))
            res["prnu_m_A"] = float(round(prnu_bba.get("A", 0.0), 4))
            res["prnu_m_U"] = float(round(prnu_bba.get("U", 1.0), 4))

            # 4. Compression Watermark Agent
            try:
                c_ev = self.comp_agent.analyze(keyframe)
                res["compression_score"] = float(round(c_ev.prob_tampered, 4))
                res["compression_m_T"] = float(round(c_ev.bba_mass.get("T", 0.0), 4))
                res["compression_m_A"] = float(round(c_ev.bba_mass.get("A", 0.0), 4))
                res["compression_m_U"] = float(round(c_ev.bba_mass.get("U", 1.0), 4))
            except Exception:
                pass

            # 5. Eye Gaze Agent
            has_face = False
            try:
                g_ev = self.gaze_agent.analyze(keyframe)
                has_face = g_ev.telemetry.get("faces_detected", 0) > 0
                res["eyegaze_score"] = float(round(g_ev.prob_tampered, 4))
                res["eyegaze_m_T"] = float(round(g_ev.bba_mass.get("T", 0.0), 4))
                res["eyegaze_m_A"] = float(round(g_ev.bba_mass.get("A", 0.0), 4))
                res["eyegaze_m_U"] = float(round(g_ev.bba_mass.get("U", 1.0), 4))
            except Exception:
                pass

            # 6. Illumination Agent
            try:
                i_ev = self.illum_agent.analyze(keyframe)
                res["illumination_score"] = float(round(i_ev.prob_tampered, 4))
                res["illumination_m_T"] = float(round(i_ev.bba_mass.get("T", 0.0), 4))
                res["illumination_m_A"] = float(round(i_ev.bba_mass.get("A", 0.0), 4))
                res["illumination_m_U"] = float(round(i_ev.bba_mass.get("U", 1.0), 4))
            except Exception:
                pass

        # 7. Video Temporal Agent & 8. RPPG Agent
        if len(frames) >= 5:
            try:
                t_ev = self.temporal_agent.analyze(frames)
                res["temporal_score"] = float(round(t_ev.prob_tampered, 4))
                res["temporal_m_T"] = float(round(t_ev.bba_mass.get("T", 0.0), 4))
                res["temporal_m_A"] = float(round(t_ev.bba_mass.get("A", 0.0), 4))
                res["temporal_m_U"] = float(round(t_ev.bba_mass.get("U", 1.0), 4))
            except Exception:
                pass

            try:
                r_ev = self.rppg_agent.analyze(frames, has_face=has_face)
                res["rppg_score"] = float(round(r_ev.prob_tampered, 4))
                res["rppg_m_T"] = float(round(r_ev.bba_mass.get("T", 0.0), 4))
                res["rppg_m_A"] = float(round(r_ev.bba_mass.get("A", 0.0), 4))
                res["rppg_m_U"] = float(round(r_ev.bba_mass.get("U", 1.0), 4))
            except Exception:
                pass

        # 9. Audio Voice Agent & 10. Lip Sync Agent
        if is_video:
            wave, sr = self.extract_audio_stream(str(p))
            has_speech = False
            if wave is not None and len(wave) >= 1600:
                rms = float(np.sqrt(np.mean(wave**2)))
                has_speech = (rms >= 0.005)

            if has_speech and wave is not None:
                try:
                    a_ev = self.audio_agent.analyze((wave, sr))
                    res["audio_score"] = float(round(a_ev.prob_tampered, 4))
                    res["audio_m_T"] = float(round(a_ev.bba_mass.get("T", 0.0), 4))
                    res["audio_m_A"] = float(round(a_ev.bba_mass.get("A", 0.0), 4))
                    res["audio_m_U"] = float(round(a_ev.bba_mass.get("U", 1.0), 4))
                except Exception:
                    pass

                if len(frames) >= 10:
                    try:
                        l_ev = self.lipsync_agent.analyze((frames, (wave, sr)))
                        res["lipsync_score"] = float(round(l_ev.prob_tampered, 4))
                        res["lipsync_m_T"] = float(round(l_ev.bba_mass.get("T", 0.0), 4))
                        res["lipsync_m_A"] = float(round(l_ev.bba_mass.get("A", 0.0), 4))
                        res["lipsync_m_U"] = float(round(l_ev.bba_mass.get("U", 1.0), 4))
                    except Exception:
                        pass

        return res


def extract_features_from_manifest(
    manifest: List[Dict[str, Any]],
    output_csv_path: str,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    resume: bool = True,
    max_samples: Optional[int] = None,
) -> str:
    """
    Processes all items in manifest and outputs standardized feature cache CSV.
    """
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = set()
    if resume and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_ids.add(row.get("id", ""))
            print(f"[+] Found {len(processed_ids)} previously processed entries in {output_csv_path}. Resuming...")
        except Exception:
            processed_ids = set()

    extractor = FeatureExtractor(weights_path=weights_path, device=device)

    items_to_process = [m for m in manifest if m.get("id", Path(m["path"]).name) not in processed_ids]
    if max_samples is not None:
        items_to_process = items_to_process[:max_samples]

    print(f"[+] Starting feature extraction for {len(items_to_process)} items -> {output_csv_path}")

    write_header = not out_path.exists() or len(processed_ids) == 0
    mode = "a" if (resume and out_path.exists() and len(processed_ids) > 0) else "w"

    with open(out_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for idx, entry in enumerate(items_to_process, 1):
            t0 = time.time()
            row = extractor.process_item(entry)
            writer.writerow(row)
            f.flush()
            dt = time.time() - t0
            if idx % 5 == 0 or idx == len(items_to_process):
                print(f"[{idx}/{len(items_to_process)}] Processed: {row['id']} ({dt:.2f}s) | Label: {row['label']}")

    print(f"[+] [SUCCESS] Feature cache successfully written to: {output_csv_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="PRISM Feature Extraction & Caching Engine")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV or JSON")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory of media to scan")
    parser.add_argument("--output_csv", type=str, default="data/cache/features_cache.csv", help="Path to output CSV")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to trained ViT weights")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda' or 'cpu')")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--no_resume", action="store_true", help="Overwrite existing output CSV")
    args = parser.parse_args()

    manifest: List[Dict[str, Any]] = []

    if args.manifest and os.path.exists(args.manifest):
        print(f"[+] Loading manifest from: {args.manifest}")
        manifest = DatasetManifestLoader.load_from_file(args.manifest)
    elif args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"[+] Scanning dataset directory: {args.dataset_dir}")
        p = Path(args.dataset_dir)
        for ext in ("*.mp4", "*.avi", "*.mov", "*.jpg", "*.png", "*.jpeg"):
            for f in sorted(list(p.rglob(ext))):
                is_fake = 1 if any(k in f.name.lower() for k in ("fake", "swap", "synth", "inswapper", "deepface")) else 0
                stem = f.stem
                ident = stem.split("_")[0]
                manifest.append(create_manifest_entry(
                    path=str(f),
                    label=is_fake,
                    identity=ident,
                    generator="deepfake" if is_fake else "original",
                    compression="unknown",
                ))
    else:
        # Default fallback: aggregate local test cohorts
        print("[+] No manifest or directory specified. Aggregating local evaluation cohorts...")
        samples_dir = REPO_ROOT / "data" / "real_test_samples"
        if samples_dir.exists():
            for ext in ("*.mp4", "*.jpg", "*.png"):
                for f in sorted(list(samples_dir.glob(ext))):
                    is_fake = 1 if any(k in f.name.lower() for k in ("fake", "swap", "synth", "inswapper")) else 0
                    stem = f.stem
                    ident = stem.split("_")[0]
                    manifest.append(create_manifest_entry(
                        path=str(f),
                        label=is_fake,
                        identity=ident,
                        generator="faceswap_inswapper" if is_fake else "original",
                        compression="unknown",
                    ))

    if not manifest:
        print("[-] Error: No media files found to process.")
        sys.exit(1)

    extract_features_from_manifest(
        manifest=manifest,
        output_csv_path=args.output_csv,
        weights_path=args.weights_path,
        device=args.device,
        resume=not args.no_resume,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
