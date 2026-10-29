"""
scripts/benchmark_compression_grid.py
Option D: In-House Video Pairs Evaluation Across H.264 Compression Grid (Raw, CRF 18, 23, 28, 35).

Measures forensic resilience and degradation of PRISM agents under progressive social-media
video compression.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agents import (
    VideoTemporalAgent,
    EyeGazeAgent,
    RPPGAgent,
    CompressionWatermarkAgent,
)
from fusion import DempsterShaferFusion


def transcode_crf(input_video: str, output_video: str, crf: int) -> bool:
    """Transcodes video using ffmpeg H.264 at specified CRF."""
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        return False
    cmd = [
        ffmpeg_cmd, "-y",
        "-i", input_video,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",
        output_video,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        print(f"Error during ffmpeg transcoding: {e}")
        return False


def run_prism_on_video(video_path: str, agents: dict, fusion: DempsterShaferFusion) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 30 and cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return {"error": "No frames"}

    t_ev = agents["temporal"].analyze(frames)
    mid_f = frames[len(frames) // 2]
    g_ev = agents["gaze"].analyze(mid_f)
    has_face = g_ev.telemetry.get("faces_detected", 0) > 0
    r_ev = agents["rppg"].analyze(frames, has_face=has_face)

    # Convert middle frame to PIL for compression agent
    from PIL import Image
    rgb_mid = cv2.cvtColor(mid_f, cv2.COLOR_BGR2RGB)
    c_ev = agents["comp"].analyze(Image.fromarray(rgb_mid))

    bba_dict = {
        "temporal": t_ev.bba_mass,
        "gaze": g_ev.bba_mass,
        "rppg": r_ev.bba_mass,
        "compression": c_ev.bba_mass,
    }
    edges = [("temporal", "gaze"), ("gaze", "rppg"), ("temporal", "compression")]
    fused = fusion.fuse_graph_evidence(bba_dict, edges)

    return {
        "temporal_prob": round(t_ev.prob_tampered, 4),
        "gaze_prob": round(g_ev.prob_tampered, 4),
        "rppg_prob": round(r_ev.prob_tampered, 4),
        "comp_prob": round(c_ev.prob_tampered, 4),
        "fused_betp": round(fused["pignistic_fake"], 4),
        "fused_belief": round(fused["belief"], 4),
        "fused_verdict": fused["verdict"],
    }


def evaluate_compression_grid() -> None:
    print("=" * 85)
    print("🔬 OPTION D: IN-HOUSE CONTROLLED PAIRS COMPRESSION GRID BENCHMARK")
    print("=" * 85)

    pairs = [
        {
            "pair_name": "Pair 1 (Inswapper Replica)",
            "auth": "data/real_test_samples/01_real_human_video_v1_authentic_source.mp4",
            "fake": "data/real_test_samples/01_real_human_video_v1_inswapper_faceswap.mp4",
        },
        {
            "pair_name": "Pair 2 (DeepFaceLab / Stanford DF1)",
            "auth": "data/real_test_samples/01_real_human_video_v1.mp4",
            "fake": "data/real_test_samples/03_deepfake_face_swap_vs1.mp4",
        },
    ]

    crf_levels = ["raw", 18, 23, 28, 35]

    agents = {
        "temporal": VideoTemporalAgent(),
        "gaze": EyeGazeAgent(),
        "rppg": RPPGAgent(),
        "comp": CompressionWatermarkAgent(),
    }
    fusion = DempsterShaferFusion()

    temp_dir = Path(tempfile.mkdtemp(prefix="prism_crf_grid_"))

    print(f"Testing {len(pairs)} authentic/fake pairs across {len(crf_levels)} compression levels:")
    print("CRF Levels: Raw (uncompressed), CRF 18 (visually lossless), CRF 23 (standard default),")
    print("            CRF 28 (social media / web), CRF 35 (heavy messaging / WhatsApp transcode)\n")

    grid_results = []

    try:
        for pair in pairs:
            print(f"--- Benchmarking: {pair['pair_name']} ---")
            p_auth = pair["auth"]
            p_fake = pair["fake"]

            if not os.path.exists(p_auth) or not os.path.exists(p_fake):
                print(f"[!] Pair files missing: {p_auth} or {p_fake}")
                continue

            for crf in crf_levels:
                if crf == "raw":
                    test_auth = p_auth
                    test_fake = p_fake
                else:
                    test_auth = str(temp_dir / f"auth_{Path(p_auth).stem}_crf{crf}.mp4")
                    test_fake = str(temp_dir / f"fake_{Path(p_fake).stem}_crf{crf}.mp4")
                    transcode_crf(p_auth, test_auth, crf)
                    transcode_crf(p_fake, test_fake, crf)

                res_auth = run_prism_on_video(test_auth, agents, fusion)
                res_fake = run_prism_on_video(test_fake, agents, fusion)

                betp_auth = res_auth.get("fused_betp", 0.5)
                betp_fake = res_fake.get("fused_betp", 0.5)
                separation = betp_fake - betp_auth
                correct_discrimination = (betp_fake > betp_auth) and (betp_fake > 0.40)

                grid_results.append({
                    "pair": pair["pair_name"],
                    "crf": str(crf),
                    "auth_betp": betp_auth,
                    "fake_betp": betp_fake,
                    "separation": separation,
                    "auth_verdict": res_auth.get("fused_verdict"),
                    "fake_verdict": res_fake.get("fused_verdict"),
                    "temporal_auth": res_auth.get("temporal_prob"),
                    "temporal_fake": res_fake.get("temporal_prob"),
                    "comp_auth": res_auth.get("comp_prob"),
                    "comp_fake": res_fake.get("comp_prob"),
                    "success": correct_discrimination,
                })

        print("=" * 85)
        print(f"{'Pair Name':25s} | {'CRF':6s} | {'Auth BetP':9s} | {'Fake BetP':9s} | {'Delta Gap':9s} | {'Auth Verdict':12s} | {'Fake Verdict'}")
        print("-" * 85)
        for r in grid_results:
            print(f"{r['pair']:25s} | {r['crf']:6s} | {r['auth_betp']:9.3f} | {r['fake_betp']:9.3f} | {r['separation']:+9.3f} | {r['auth_verdict']:12s} | {r['fake_verdict']}")
        print("=" * 85)

        # Quantitative synthesis of compression degradation
        print("\n--- Compression Degradation & Resilience Analysis ---")
        for crf in crf_levels:
            subset = [r for r in grid_results if r["crf"] == str(crf)]
            mean_sep = np.mean([r["separation"] for r in subset])
            mean_fake_betp = np.mean([r["fake_betp"] for r in subset])
            print(f"CRF {str(crf):5s} -> Mean Forensic Separation (Fake - Auth BetP): {mean_sep:+.4f} | Avg Fake BetP: {mean_fake_betp:.3f}")

        # Check threshold collapse at CRF 35
        crf_raw_sep = np.mean([r["separation"] for r in grid_results if r["crf"] == "raw"])
        crf_35_sep = np.mean([r["separation"] for r in grid_results if r["crf"] == "35"])
        decay = (crf_raw_sep - crf_35_sep) / (crf_raw_sep + 1e-6)
        print(f"\nForensic Separation Decay (Raw to CRF 35): {decay:6.1%}")
        if decay > 0.30:
            print("⚠️  Forensic Vulnerability Identified: Heavy social compression (CRF 35) significantly suppresses high-frequency artifacts, reducing forensic margin.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    evaluate_compression_grid()
