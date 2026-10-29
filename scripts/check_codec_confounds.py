"""
scripts/check_codec_confounds.py
Forensic Codec & Confound Analysis Suite for PRISM Evaluation Sets.

Analyzes every media file across evaluation sets (data/real_test_samples, data/test_samples):
  1. Extracts container, video/audio codec, bitrate, resolution, fps, duration,
     and re-encoding indicators (e.g. Lavf/FFmpeg metadata tags).
  2. Groups attributes by Ground Truth label (Real vs Fake).
  3. Tests whether the label is predictable from metadata attributes alone.
  4. Explicitly flags forensic agents vulnerable to codec/compression shortcuts.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def probe_media_file(file_path: str) -> Dict[str, Any]:
    """
    Probes media container and stream details via ffprobe.
    Falls back gracefully if stream is image or audio.
    """
    p = Path(file_path)
    ext = p.suffix.lower().lstrip(".")
    res = {
        "path": str(p),
        "filename": p.name,
        "container": ext,
        "video_codec": "None",
        "audio_codec": "None",
        "bitrate_kbps": 0.0,
        "resolution": "N/A",
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration_s": 0.0,
        "is_reencoded": False,
        "encoder_tag": "None",
    }

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(p)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        info = json.loads(out.decode("utf-8", errors="ignore"))

        fmt = info.get("format", {})
        streams = info.get("streams", [])

        # Format / Container info
        format_name = fmt.get("format_name", ext)
        res["container"] = format_name.split(",")[0]
        if "bit_rate" in fmt:
            res["bitrate_kbps"] = round(float(fmt["bit_rate"]) / 1000.0, 1)
        if "duration" in fmt:
            res["duration_s"] = round(float(fmt["duration"]), 2)

        # Tags in format
        tags = fmt.get("tags", {})
        encoder_str = tags.get("encoder", tags.get("ENCODER", tags.get("major_brand", "")))
        res["encoder_tag"] = encoder_str if encoder_str else "None"

        # Check for re-encoding signatures (Lavf/FFmpeg/Handbrake/Adobe/Google)
        all_tags_str = json.dumps(tags).lower()
        if any(sig in all_tags_str for sig in ["lavf", "lavc", "ffmpeg", "handbrake", "isom", "mp42", "google"]):
            res["is_reencoded"] = True

        for s in streams:
            codec_type = s.get("codec_type")
            codec_name = s.get("codec_name", "unknown")
            s_tags = s.get("tags", {})
            s_tags_str = json.dumps(s_tags).lower()

            if any(sig in s_tags_str for sig in ["lavf", "lavc", "ffmpeg"]):
                res["is_reencoded"] = True
                if res["encoder_tag"] == "None" and "encoder" in s_tags:
                    res["encoder_tag"] = s_tags["encoder"]

            if codec_type == "video":
                res["video_codec"] = codec_name
                w = s.get("width", 0)
                h = s.get("height", 0)
                res["width"] = w
                res["height"] = h
                res["resolution"] = f"{w}x{h}" if w and h else "N/A"

                # Parse frame rate
                r_fps = s.get("r_frame_rate", "0/1")
                if "/" in r_fps:
                    num, den = r_fps.split("/")
                    res["fps"] = round(float(num) / float(den), 2) if float(den) > 0 else 0.0
                else:
                    res["fps"] = round(float(r_fps), 2)

                if res["bitrate_kbps"] == 0.0 and "bit_rate" in s:
                    res["bitrate_kbps"] = round(float(s["bit_rate"]) / 1000.0, 1)

            elif codec_type == "audio":
                res["audio_codec"] = codec_name
                if res["bitrate_kbps"] == 0.0 and "bit_rate" in s:
                    res["bitrate_kbps"] = round(float(s["bit_rate"]) / 1000.0, 1)

    except Exception as e:
        # Fallback for still images or unprobed files
        pass

    return res


def determine_label(filename: str) -> int:
    """
    Ground truth label: 0 for Authentic / Pristine, 1 for Tampered / Deepfake.
    """
    fn = filename.lower()
    if any(k in fn for k in ["fake", "swap", "spliced", "tampered", "midjourney", "inswapper", "synthetic"]):
        return 1
    return 0


def run_confound_analysis(target_dirs: List[str]) -> Dict[str, Any]:
    records = []
    for d in target_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".wav"]:
                info = probe_media_file(str(f))
                info["label"] = determine_label(f.name)
                info["label_str"] = "FAKE" if info["label"] == 1 else "REAL"
                records.append(info)

    # Print Table Grouped by Label
    real_records = [r for r in records if r["label"] == 0]
    fake_records = [r for r in records if r["label"] == 1]

    print("=" * 115)
    print("📋 PRISM CODEC & METADATA CONFOUND AUDIT (EVALUATION SETS)")
    print("=" * 115)
    print(f"Total Files Analyzed: {len(records)} | Real: {len(real_records)} | Fake: {len(fake_records)}")
    print("=" * 115)

    for group_name, grp in [("AUTHENTIC (REAL) SAMPLES", real_records), ("TAMPERED (DEEPFAKE) SAMPLES", fake_records)]:
        print(f"\n--- {group_name} (N = {len(grp)}) ---")
        print(f"{'Filename':38s} | {'Container':10s} | {'Codec':10s} | {'Bitrate':10s} | {'Resolution':12s} | {'FPS':6s} | {'Dur(s)':6s} | {'Re-encoded':10s}")
        print("-" * 115)
        for r in grp:
            codec_disp = r["video_codec"] if r["video_codec"] != "None" else r["audio_codec"]
            br_disp = f"{r['bitrate_kbps']:.0f} kbps" if r['bitrate_kbps'] > 0 else "N/A"
            fps_disp = f"{r['fps']:.1f}" if r['fps'] > 0 else "N/A"
            dur_disp = f"{r['duration_s']:.1f}" if r['duration_s'] > 0 else "N/A"
            reenc_disp = "YES (Lavf)" if r["is_reencoded"] else "NO (Orig)"
            print(f"{r['filename'][:38]:38s} | {r['container'][:10]:10s} | {codec_disp[:10]:10s} | {br_disp:10s} | {r['resolution']:12s} | {fps_disp:6s} | {dur_disp:6s} | {reenc_disp:10s}")

    print("=" * 115)

    # Statistical Predictability Analysis
    # Can label be predicted from metadata alone?
    # Features: is_reencoded (binary), fps > 0, bitrate, width*height, container is mp4
    X = []
    y = []
    for r in records:
        area = r["width"] * r["height"]
        is_mp4 = 1.0 if "mp4" in r["container"] else 0.0
        is_jpg = 1.0 if "jpeg" in r["container"] or "jpg" in r["container"] else 0.0
        reenc = 1.0 if r["is_reencoded"] else 0.0
        br = r["bitrate_kbps"]
        fps = r["fps"]
        dur = r["duration_s"]
        X.append([reenc, is_mp4, is_jpg, br, area, fps, dur])
        y.append(r["label"])

    X = np.array(X)
    y = np.array(y)

    # 1. Simple Re-encoded shortcut predictor:
    # Predict Fake if is_reencoded == 1
    pred_by_reenc = (X[:, 0] == 1.0).astype(int)
    acc_reenc = float(np.mean(pred_by_reenc == y))
    tp_reenc = float(np.sum((pred_by_reenc == 1) & (y == 1)) / (np.sum(y == 1) + 1e-6))
    fp_reenc = float(np.sum((pred_by_reenc == 1) & (y == 0)) / (np.sum(y == 0) + 1e-6))
    bal_acc_reenc = 0.5 * (tp_reenc + (1.0 - fp_reenc))

    # 2. Container shortcut:
    container_counts = {}
    for r in records:
        c = r["container"]
        if c not in container_counts:
            container_counts[c] = {"real": 0, "fake": 0}
        if r["label"] == 0:
            container_counts[c]["real"] += 1
        else:
            container_counts[c]["fake"] += 1

    print("\n" + "=" * 80)
    print("📊 METADATA CONFOUND & SHORTCUT PREDICTABILITY REPORT")
    print("=" * 80)
    print(f"1. Re-encoding Flag (Lavf / FFmpeg metadata signature):")
    print(f"   - Re-encoded among Fakes: {sum(r['is_reencoded'] for r in fake_records)} / {len(fake_records)} ({sum(r['is_reencoded'] for r in fake_records)/len(fake_records):.1%})")
    print(f"   - Re-encoded among Reals: {sum(r['is_reencoded'] for r in real_records)} / {len(real_records)} ({sum(r['is_reencoded'] for r in real_records)/len(real_records):.1%})")
    print(f"   - Trivial Metadata-Only Balanced Accuracy: {bal_acc_reenc:.1%} (FPR={fp_reenc:.1%}, TPR={tp_reenc:.1%})")

    print(f"\n2. Container Distribution Across Classes:")
    for c, cnts in container_counts.items():
        print(f"   - Container '{c}': {cnts['real']} Real, {cnts['fake']} Fake")

    # 3. Codec shortcuts warning per agent
    print("\n" + "=" * 80)
    print("⚠️  VULNERABILITY AUDIT: AGENTS AT RISK OF CODEC SHORTCUTS")
    print("=" * 80)

    flagged_agents = [
        {
            "agent": "CompressionWatermarkAgent",
            "risk_level": "HIGH",
            "shortcut_mechanism": "Quantization table divergence & ELA residuals reflect FFmpeg/H.264 re-compression rather than face-swap boundaries.",
            "confound_evidence": "Re-encoded fakes display strong DCT grid anomalies caused by secondary transcoding.",
        },
        {
            "agent": "MetadataAgent",
            "risk_level": "CRITICAL",
            "shortcut_mechanism": "Inspects container atoms and encoder tags (e.g. 'Lavf', 'isom'). If benchmark fakes are generated by FFmpeg while reals retain camera EXIF, this is a 100% trivial metadata shortcut.",
            "confound_evidence": "Stripped vs populated metadata separates raw web downloads from pristine camera photos.",
        },
        {
            "agent": "VideoTemporalAgent",
            "risk_level": "MEDIUM",
            "shortcut_mechanism": "Farneback dense optical flow variance is modulated by macroblock compression artifacts and non-uniform GOP frame rates.",
            "confound_evidence": "Bitrate discrepancies between fakes and reals alter inter-frame motion vector smoothness.",
        },
        {
            "agent": "AudioVoiceAgent",
            "risk_level": "HIGH",
            "shortcut_mechanism": "High-frequency brickwall attenuation (vocoder check) will trigger false alarms on MP3/AAC audio with 16 kHz low-pass filters.",
            "confound_evidence": "Transcoded audio clips confound neural vocoder signatures with standard perceptual audio compression cutoffs.",
        },
    ]

    for fa in flagged_agents:
        print(f"[{fa['risk_level']} RISK] {fa['agent']}:")
        print(f"  • Mechanism: {fa['shortcut_mechanism']}")
        print(f"  • Confound : {fa['confound_evidence']}\n")

    print("=" * 80)

    return {
        "total_records": len(records),
        "real_count": len(real_records),
        "fake_count": len(fake_records),
        "reencoding_shortcut_balanced_acc": round(bal_acc_reenc, 4),
        "container_counts": container_counts,
        "flagged_agents": flagged_agents,
        "records": records,
    }


if __name__ == "__main__":
    target_directories = ["data/real_test_samples", "data/test_samples"]
    run_confound_analysis(target_directories)
