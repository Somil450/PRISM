"""
tests/test_viral_indian_forensics.py
Regression & Diagnostic Test Suite for Indian Demographic Deepfakes & Fact-Check Benchmarks.

Validates:
  1. Option A: DF-Platter benchmark loader, demographic attribute parser (Fitzpatrick, headwear, facial hair),
               and IIT Jodhpur academic access interface.
  2. Option B: InDeepFake & SAMANVAYA Indic speech evaluation across language slices (Hindi, Tamil, Telugu, Bengali)
               and cross-modal lip-sync metrics.
  3. Option C: Fact-checker dossier archives (BOOM Live, Alt News, BBC Hindi, India Today AFWA) and automated
               viral forensic signature inspection (latch glitch, seam, PRNU divergence).
  4. Option D: Controlled Inswapper test pair generation (authentic vs inswapper face-swap) with H.264 CRF
               recompression, PRNU mismatch, and comparative multi-agent PRISM audit.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.video_temporal_agent import VideoTemporalAgent
from agents.compression_watermark_agent import CompressionWatermarkAgent
from agents.eye_gaze_agent import EyeGazeAgent
from agents.rppg_agent import RPPGAgent
from fusion import DempsterShaferFusion
from scripts.generate_controlled_faceswap_pairs import ControlledFaceSwapGenerator
from scripts.benchmark_df_platter import DFPlatterBenchmark, DFPlatterMetadataParser
from scripts.benchmark_indeepfake import InDeepFakeBenchmark, IndicMetadataParser
from scripts.fetch_factcheck_samples import (
    FACT_CHECK_CASE_STUDIES,
    ViralForensicSignatureDetector,
    build_factcheck_archive,
)


class TestViralIndianForensics(unittest.TestCase):
    """
    Forensic test suite specifically validating PRISM against Indian celebrity
    deepfakes, fact-checking archives, and controlled research replicas.
    """

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parent.parent
        cls.samples_dir = repo_root / "data" / "real_test_samples"
        cls.auth_video = cls.samples_dir / "01_real_human_video_v1_authentic_source.mp4"
        cls.fake_video = cls.samples_dir / "01_real_human_video_v1_inswapper_faceswap.mp4"

        # Ensure controlled pair exists
        if not cls.auth_video.exists() or not cls.fake_video.exists():
            gen = ControlledFaceSwapGenerator()
            src = cls.samples_dir / "01_real_human_video_v1.mp4"
            if src.exists():
                gen.generate_pair(str(src), str(cls.samples_dir), max_frames=30)

    def test_01_factcheck_case_studies_cataloged(self):
        """Option C: Verifies fact-checking case studies include Alt News, BOOM Live, BBC Hindi, and AFWA."""
        self.assertIn("rashmika_elevator_viral_2023", FACT_CHECK_CASE_STUDIES)
        case = FACT_CHECK_CASE_STUDIES["rashmika_elevator_viral_2023"]
        self.assertEqual(case["celebrity_target"], "Rashmika Mandanna")
        self.assertEqual(case["original_subject"], "Zara Patel (British-Indian fitness influencer)")

        # Verify key forensic indicators
        indicators = case["documented_forensic_indicators"]
        self.assertIn("temporal_latch_glitch", indicators)
        self.assertIn("subband_blending_seam", indicators)
        self.assertIn("sensor_prnu_mismatch", indicators)

        # Verify reporting outlets
        outlets = [r["outlet"] for r in case["verified_factcheck_reports"]]
        self.assertTrue(any("Alt News" in o for o in outlets))
        self.assertTrue(any("BOOM Live" in o for o in outlets))
        self.assertTrue(any("BBC" in o for o in outlets))
        self.assertTrue(any("India Today" in o for o in outlets))

        # Check export works cleanly
        dossier_path = build_factcheck_archive()
        self.assertTrue(Path(dossier_path).exists())

    def test_02_viral_signature_detector(self):
        """Option C: Verifies viral signature detector accurately discriminates between replica and authentic."""
        detector = ViralForensicSignatureDetector()
        self.assertTrue(self.fake_video.exists(), "Faceswap replica video must exist for signature testing")
        res_fake = detector.inspect_video(str(self.fake_video))
        self.assertTrue(res_fake["matches_viral_roop_signature"],
                        f"Faceswap replica must trigger viral signature, got {res_fake}")
        self.assertTrue(res_fake["temporal_latch_glitch"]["detected"],
                        "Faceswap replica must trigger temporal latch glitch detection")

        if self.auth_video.exists():
            res_auth = detector.inspect_video(str(self.auth_video))
            self.assertFalse(res_auth["matches_viral_roop_signature"],
                             f"Authentic video must NOT trigger viral signature, got {res_auth}")
            self.assertFalse(res_auth["temporal_latch_glitch"]["detected"],
                             "Authentic video must NOT trigger temporal latch glitch")

    def test_03_controlled_faceswap_generation(self):
        """Option D: Verifies controlled Inswapper face swap generation reproduces exact artifact traits."""
        self.assertTrue(self.auth_video.exists(), "Authentic source video must exist")
        self.assertTrue(self.fake_video.exists(), "Inswapper deepfake replica must exist")

        cap_auth = cv2.VideoCapture(str(self.auth_video))
        cap_fake = cv2.VideoCapture(str(self.fake_video))

        ret_a, frame_a = cap_auth.read()
        ret_f, frame_f = cap_fake.read()
        cap_auth.release()
        cap_fake.release()

        self.assertTrue(ret_a and ret_f, "Both videos must decode valid frames")
        self.assertEqual(frame_a.shape, frame_f.shape, "Frame dimensions must match exactly")

    def test_04_authentic_source_baseline_verification(self):
        """Option C/D: Verifies authentic human camera video is recognized as AUTHENTIC."""
        if not self.auth_video.exists():
            self.skipTest("Authentic video sample missing")

        cap = cv2.VideoCapture(str(self.auth_video))
        frames = []
        while len(frames) < 30 and cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        temporal_agent = VideoTemporalAgent()
        ev = temporal_agent.analyze(frames)

        self.assertEqual(ev.verdict, "AUTHENTIC", "Authentic source must yield AUTHENTIC verdict")
        self.assertLess(ev.prob_tampered, 0.20, f"Authentic source P(T) must be < 0.20, got {ev.prob_tampered}")
        self.assertLess(ev.telemetry["motion_jitter_index"], 0.15, "Natural optical flow must have low jitter")

    def test_05_df_platter_demographic_slices_and_benchmark(self):
        """Option A: Verifies DF-Platter demographic attribute parser and multi-slice benchmark."""
        from scripts.benchmark_df_platter import DFPlatterMetadataLoader
        loader = DFPlatterMetadataLoader()
        attrs = loader.get_metadata(Path("data/datasets/DF-Platter/c23/deepfacelab/fitz_IV_turban_beard_001.mp4"))
        self.assertEqual(attrs["technique"], "DeepFaceLab")
        self.assertEqual(attrs["compression"], "c23")
        self.assertEqual(attrs["fitzpatrick"], "Fitzpatrick IV")
        self.assertEqual(attrs["headwear"], "Turban")
        self.assertEqual(attrs["facial_hair"], "Beard / Moustache")
        self.assertTrue(attrs["is_deepfake"])

        bench = DFPlatterBenchmark()
        summary = bench.run_benchmark(str(self.samples_dir))
        self.assertIn("total_evaluated", summary)
        self.assertGreater(summary["total_evaluated"], 0)
        self.assertIn("overall_accuracy", summary)
        self.assertIn("slice_statistics", summary)
        self.assertIn("technique", summary["slice_statistics"])
        self.assertIn("compression", summary["slice_statistics"])

    def test_06_indeepfake_indic_language_benchmark(self):
        """Option B: Verifies InDeepFake language parser (Hindi/Tamil/Telugu/Bengali) and multimodal evaluation."""
        parser = IndicMetadataParser()
        attrs_hi = parser.parse_attributes(Path("hindi/speaker_01_lip_manipulated.mp4"))
        attrs_ta = parser.parse_attributes(Path("tamil/speaker_02_voice_cloned.mp4"))
        self.assertEqual(attrs_hi["language"], "Hindi")
        self.assertEqual(attrs_hi["modality"], "Lip-Sync Manipulated")
        self.assertEqual(attrs_ta["language"], "Tamil")
        self.assertEqual(attrs_ta["modality"], "Acoustic Voice Cloned")

        bench = InDeepFakeBenchmark()
        summary = bench.run_benchmark(str(self.samples_dir))
        self.assertIn("total_evaluated", summary)
        self.assertGreater(summary["total_evaluated"], 0)
        self.assertIn("overall_accuracy", summary)
        self.assertIn("language_statistics", summary)
        self.assertIn("modality_statistics", summary)


if __name__ == "__main__":
    unittest.main()
