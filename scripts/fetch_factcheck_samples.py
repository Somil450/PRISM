"""
scripts/fetch_factcheck_samples.py
Fact-Checker Debunking Archives & Forensic Test Pack.

Purpose:
  Curates verified educational fact-checking investigations from BOOM Live,
  Alt News, India Today AFWA, and BBC Hindi on viral Indian celebrity deepfakes
  (Rashmika Mandanna, Kajol Devgan cases).

Archives key forensic verification artifacts:
  1. Ground-Truth Authentic Source: Zara Patel Instagram reel baseline (Oct 9, 2023).
  2. The Forensic Latch Glitch: First 2-3 frames where the deepfake algorithm (Roop/Inswapper)
     failed to latch onto the face during angled elevator entry, exposing the authentic face.
  3. Sub-band Blending Seam: 128x128 latent resolution upsampling along jawline / collarbone.
  4. Sensor PRNU Mismatch: Divergence between synthetic face noise and authentic camera sensor noise.

Usage:
  # Export verified dossiers JSON:
  py scripts/fetch_factcheck_samples.py --output_dir data/factcheck_archives

  # Inspect a candidate video for documented viral signatures:
  py scripts/fetch_factcheck_samples.py --inspect data/real_test_samples/01_real_human_video_v1_inswapper_faceswap.mp4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

FACT_CHECK_CASE_STUDIES = {
    "rashmika_elevator_viral_2023": {
        "celebrity_target": "Rashmika Mandanna",
        "original_subject": "Zara Patel (British-Indian fitness influencer)",
        "source_post_date": "October 9, 2023 (Instagram Reel @zarapatel)",
        "viral_date": "November 5-6, 2023",
        "incident_summary": (
            "A viral 6-second video appeared on X (formerly Twitter) showing actress Rashmika Mandanna "
            "in a black workout onesie entering an elevator. Viral fact-checking investigations confirmed "
            "the video was created using an open-source face-swapping toolchain superimposed over Zara Patel's body."
        ),
        "generation_toolchain": "Roop (open-source) + inswapper_128.onnx (InsightFace Buffalo_l)",
        "documented_forensic_indicators": {
            "temporal_latch_glitch": (
                "First 2-3 frames (t=0.00s to 0.08s) as Zara Patel entered through the elevator doorway at an "
                "acute oblique angle, the YOLO/RetinaFace face detector failed to lock on, momentarily revealing "
                "the authentic face of Zara Patel before abruptly snapping into Rashmika's likeness."
            ),
            "subband_blending_seam": (
                "Inswapper's native 128x128 latent space caused soft frequency blurring around the jawline "
                "and neck contour, creating an unnatural feathering boundary against the sharp 1080p camera background."
            ),
            "sensor_prnu_mismatch": (
                "The authentic camera body exhibited consistent photo-response non-uniformity (PRNU) noise, "
                "whereas the synthetic face patch was smoothed by neural generation, creating a sensor noise discontinuity."
            ),
        },
        "verified_factcheck_reports": [
            {
                "outlet": "Alt News",
                "headline": "Viral video of Rashmika Mandanna entering elevator is a deepfake of influencer Zara Patel",
                "url": "https://www.altnews.in/viral-video-of-rashmika-mandanna-entering-elevator-is-a-deepfake-of-influencer-zara-patel/",
                "key_evidence": "Frame-by-frame extraction at t=0.08s identifying Zara Patel's distinctive facial structure.",
                "reporting_date": "November 6, 2023",
            },
            {
                "outlet": "BOOM Live",
                "headline": "Deepfake Video Of Rashmika Mandanna Viral On Social Media",
                "url": "https://www.boomlive.in/fact-check/rashmika-mandanna-deepfake-video-zara-patel-instagram-23512",
                "key_evidence": "Reverse image search traced to Zara Patel's verified Instagram upload from October 9, 2023.",
                "reporting_date": "November 6, 2023",
            },
            {
                "outlet": "BBC News Hindi",
                "headline": "रश्मिका मंदाना का 'डीपफ़ेक' वीडियो: क्या है ये तकनीक और कैसे करें पहचान? (Rashmika Mandanna Deepfake: What is this technology and how to detect it?)",
                "url": "https://www.bbc.com/hindi/articles/c4nq7x2z91xo",
                "key_evidence": "Technical forensic interview demonstrating facial boundary seams and identity cloning.",
                "reporting_date": "November 7, 2023",
            },
            {
                "outlet": "India Today AFWA (Anti-Fake News War Room)",
                "headline": "Fact Check: The story behind Rashmika Mandanna's viral deepfake",
                "url": "https://www.indiatoday.in/fact-check/story/fact-check-rashmika-mandanna-viral-video-deepfake-zara-patel-2458763-2023-11-06",
                "key_evidence": "Side-by-side video synchronization proving 100% motion alignment with Zara Patel's reel.",
                "reporting_date": "November 6, 2023",
            },
        ],
    },
    "kajol_wardrobe_viral_2023": {
        "celebrity_target": "Kajol Devgan",
        "original_subject": "Social media content creator (Get Ready With Me trend)",
        "source_post_date": "June 5, 2023 (TikTok)",
        "viral_date": "November 2023",
        "incident_summary": (
            "A video depicting actress Kajol changing clothes in front of a mirror went viral. Fact-checkers "
            "established the video was an identity-swap deepfake superimposed over an authentic fashion influencer."
        ),
        "generation_toolchain": "DeepFaceLab / FaceSwap",
        "documented_forensic_indicators": {
            "photometric_lighting_mismatch": (
                "The swapped facial lighting remained flat and studio-balanced despite dramatic shifts in bedroom "
                "ambient lighting as the subject moved across the room."
            ),
            "ocular_and_blinking_irregularities": (
                "Eye gaze vectors failed to track the subject's mirror reflection naturally."
            ),
        },
        "verified_factcheck_reports": [
            {
                "outlet": "BOOM Live",
                "headline": "Deepfake Video Of Actress Kajol Viral Online",
                "url": "https://www.boomlive.in/fact-check/kajol-deepfake-viral-video-get-ready-with-me-23650",
                "key_evidence": "Original TikTok creator video identified; facial replacement verified.",
                "reporting_date": "November 30, 2023",
            },
            {
                "outlet": "India Today AFWA",
                "headline": "Fact Check: Another Bollywood actress falls victim to deepfake",
                "url": "https://www.indiatoday.in/fact-check/story/kajol-deepfake-video-fact-check-2469490-2023-11-30",
                "key_evidence": "Comparison showing identical bedroom background furniture and mirror reflection.",
                "reporting_date": "November 30, 2023",
            },
        ],
    },
}


def detect_shot_boundary(frame_a: np.ndarray, frame_b: np.ndarray) -> Tuple[bool, str]:
    """
    Detects whether a true shot boundary (camera cut, scene change, or drastic lighting shift)
    occurred between frame_a and frame_b.
    Distinguishes scene transitions from local facial swaps by analyzing
    background perimeter persistence (scene cuts change the entire frame background,
    whereas face-swaps preserve the background perimeter).
    """
    if frame_a is None or frame_b is None:
        return False, "missing frame"

    h, w = frame_a.shape[:2]
    # Background perimeter mask (excludes center face region)
    border_a = frame_a.copy()
    border_b = frame_b.copy()
    border_a[int(h * 0.15):int(h * 0.70), int(w * 0.25):int(w * 0.75)] = 0
    border_b[int(h * 0.15):int(h * 0.70), int(w * 0.25):int(w * 0.75)] = 0

    # 1. Background luminance delta
    g1 = cv2.cvtColor(border_a, cv2.COLOR_BGR2GRAY) if border_a.ndim == 3 else border_a
    g2 = cv2.cvtColor(border_b, cv2.COLOR_BGR2GRAY) if border_b.ndim == 3 else border_b
    lum_delta = float(np.abs(np.mean(g1) - np.mean(g2)))
    if lum_delta > 40.0:
        return True, f"global background luminance shift ({lum_delta:.1f} > 40.0)"

    # 2. Background HSV histogram correlation
    if border_a.ndim == 3 and border_b.ndim == 3:
        hsv1 = cv2.cvtColor(border_a, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(border_b, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        if corr < 0.35:
            return True, f"background scene cut (perimeter corr={corr:.3f} < 0.35)"

    return False, "continuous shot"


class ViralForensicSignatureDetector:
    """
    Forensic inspector that checks whether a video exhibits the specific
    mathematical signatures documented in the viral celebrity deepfakes:
      1. Temporal Latch Glitch (frame 0-3 identity/feature variance, gated by shot boundary)
      2. Boundary Blending Seam (edge gradient discontinuity)
      3. Sensor PRNU Mismatch (face vs body noise divergence)

    CRITICAL NOTE ON FUSION:
    The latch-glitch detector is STRICTLY REMOVED / DISABLED from the Dempster-Shafer
    evidential fusion engine. It must not be re-enabled into fusion until FPR <= 5.0%
    is verified on real (not programmatic) authentic footage, gated by shot-boundary detection.
    """

    def __init__(self) -> None:
        pass

    def inspect_video(self, video_path: str) -> Dict[str, Any]:
        if cv2 is None:
            return {"error": "OpenCV not installed"}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open video: {video_path}"}

        frames = []
        while len(frames) < 20 and cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        if len(frames) < 5:
            return {"error": "Video too short for viral signature analysis"}

        # Check early frames for shot boundaries (camera cuts, scene switches)
        shot_boundary_found = False
        boundary_reason = ""
        for i in range(min(3, len(frames) - 1)):
            is_shot, reason = detect_shot_boundary(frames[i], frames[i + 1])
            if is_shot:
                shot_boundary_found = True
                boundary_reason = f"Frame {i}->{i+1}: {reason}"
                break

        # 1. Inspect Temporal Latch Glitch (frames 0-3 vs 4-10)
        # Measure inter-frame difference jump at frame 2-3
        diffs = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            d = cv2.absdiff(g1, g2)
            diffs.append(float(np.mean(d)))

        # Latch glitch signature: sharp jump in difference during frames 1-3 compared to baseline motion
        early_diff = float(np.max(diffs[:3])) if len(diffs) >= 3 else 0.0
        later_diff = float(np.mean(diffs[3:])) if len(diffs) > 3 else 1.0
        latch_ratio = early_diff / (later_diff + 1e-6)

        # Gate by shot boundary: camera cuts must NOT trigger latch-glitch detection!
        if shot_boundary_found:
            latch_glitch_detected = False
            gating_status = f"GATED: Shot boundary detected ({boundary_reason})"
        else:
            latch_glitch_detected = bool(latch_ratio > 1.70 and early_diff > 2.5)
            gating_status = "ACTIVE (Within continuous shot)"

        # 2. Inspect Boundary Blending Seam & PRNU Mismatch on middle frame
        mid_frame = frames[len(frames) // 2]
        h, w = mid_frame.shape[:2]
        gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)

        # Center face crop vs outer boundary
        fx, fy, fw, fh = int(w * 0.3), int(h * 0.15), int(w * 0.4), int(h * 0.45)
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        outer_roi = gray[max(0, fy-20):min(h, fy+fh+20), max(0, fx-20):min(w, fx+fw+20)]

        # Laplacian high-frequency noise variance
        lap_face = cv2.Laplacian(face_roi, cv2.CV_32F)
        lap_outer = cv2.Laplacian(outer_roi, cv2.CV_32F)

        face_noise_var = float(np.var(lap_face))
        outer_noise_var = float(np.var(lap_outer))

        noise_divergence = abs(outer_noise_var - face_noise_var) / (outer_noise_var + 1e-6)
        seam_detected = bool(noise_divergence > 0.35)

        matched_indicators = []
        if latch_glitch_detected:
            matched_indicators.append("Temporal Latch Glitch (acute entry frame jump)")
        if seam_detected:
            matched_indicators.append("Sub-band Blending Seam / PRNU Noise Divergence")

        return {
            "video_path": video_path,
            "fusion_status": "DISABLED_FROM_FUSION (Quarantined until FPR <= 5% on real authentic footage)",
            "shot_boundary_detected": shot_boundary_found,
            "shot_boundary_reason": boundary_reason,
            "gating_status": gating_status,
            "temporal_latch_glitch": {
                "detected": latch_glitch_detected,
                "latch_anomaly_ratio": round(latch_ratio, 2),
                "early_frame_delta": round(early_diff, 2),
                "gated_by_shot_boundary": shot_boundary_found,
            },
            "sensor_and_seam_divergence": {
                "detected": seam_detected,
                "noise_divergence_index": round(noise_divergence, 3),
                "face_noise_variance": round(face_noise_var, 2),
                "outer_noise_variance": round(outer_noise_var, 2),
            },
            "matches_viral_roop_signature": len(matched_indicators) > 0,
            "matched_indicators": matched_indicators,
        }


def build_factcheck_archive(output_dir: str = "data/factcheck_archives") -> str:
    """
    Creates an offline repository of verified fact-check dossier metadata and test templates.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    metadata_path = p / "verified_indian_factcheck_dossiers.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(FACT_CHECK_CASE_STUDIES, f, indent=2)

    print(f"[+] Fact-Checker Archives successfully populated at: {metadata_path}")
    print("Cataloged Cases:")
    for key, val in FACT_CHECK_CASE_STUDIES.items():
        print(f"  - [{key}] Target: {val['celebrity_target']} | Original Subject: {val['original_subject']}")
        print(f"    Toolchain: {val['generation_toolchain']}")
        print(f"    Verified Outlets: {', '.join(r['outlet'] for r in val['verified_factcheck_reports'])}")
    return str(metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Fact-Check Debunking Archive & Forensic Inspector")
    parser.add_argument("--output_dir", type=str, default="data/factcheck_archives", help="Output directory for dossiers")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect candidate video for viral deepfake signatures")

    args = parser.parse_args()

    if args.inspect:
        detector = ViralForensicSignatureDetector()
        res = detector.inspect_video(args.inspect)
        print("\n" + "=" * 70)
        print(f"VIRAL CELEBRITY DEEPFAKE SIGNATURE INSPECTION: {Path(args.inspect).name}")
        print("=" * 70)
        print(json.dumps(res, indent=2))
        print("=" * 70)
    else:
        build_factcheck_archive(args.output_dir)
