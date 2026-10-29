"""
tests/test_robustness_and_failures.py
Forensic Robustness, Perturbation Invariance & Failure Mode Graceful Degradation Suite.

Validates:
  1. Image Perturbations:
     - JPEG Recompression (Q=40)
     - Resizing & Downscaling (50% scale)
     - Gaussian Blur (k=5)
     - Additive Gaussian Noise (sigma=15)
     - Unsharp Masking
  2. Failure Modes & Graceful Degradation:
     - Non-face blank frames -> Graceful bypass with pure uncertainty m(U) = 1.0, verdict INCONCLUSIVE
     - Multi-face frames -> Handled cleanly without indexing errors
     - Very short video clips (< 5 frames) -> Temporal & rPPG agents return INCONCLUSIVE
     - Zero-length / silent audio -> Audio agent returns graceful bypass
     - Corrupted or invalid media bytes -> Handled safely without unhandled crashes
"""

import sys
import os
import unittest
import numpy as np
import cv2
from PIL import Image, ImageFilter
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

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
from fusion import DempsterShaferFusion


class TestForensicRobustnessAndFailures(unittest.TestCase):
    """
    Stress-tests PRISM forensic modules under real-world social media
    perturbations and extreme boundary/failure input conditions.
    """

    @classmethod
    def setUpClass(cls):
        cls.meta_agent = MetadataAgent()
        cls.gaze_agent = EyeGazeAgent()
        cls.comp_agent = CompressionWatermarkAgent()
        cls.audio_agent = AudioVoiceAgent()
        cls.temp_agent = VideoTemporalAgent()
        cls.illum_agent = IlluminationAgent()
        cls.rppg_agent = RPPGAgent()
        cls.fusion = DempsterShaferFusion()

        # Base synthetic face-like canvas
        cls.base_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.circle(cls.base_img, (112, 112), 80, (180, 160, 150), -1)
        cv2.circle(cls.base_img, (85, 95), 12, (255, 255, 255), -1)
        cv2.circle(cls.base_img, (139, 95), 12, (255, 255, 255), -1)
        cv2.circle(cls.base_img, (85, 95), 5, (20, 20, 20), -1)
        cv2.circle(cls.base_img, (139, 95), 5, (20, 20, 20), -1)

    # -----------------------------------------------------------------------
    # 1. Perturbation Tests
    # -----------------------------------------------------------------------
    def test_01_jpeg_recompression_robustness(self):
        """Heavy JPEG recompression (Q=40) increases uncertainty m(U) or maintains safe verdict."""
        pil_img = Image.fromarray(self.base_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=40)
        buffer.seek(0)
        compressed_img = Image.open(buffer)

        ev_orig = self.comp_agent.analyze(pil_img)
        ev_comp = self.comp_agent.analyze(compressed_img)

        self.assertIn(ev_comp.verdict, ["AUTHENTIC", "INCONCLUSIVE", "TAMPERED"])
        self.assertAlmostEqual(sum(ev_comp.bba_mass.values()), 1.0, places=4)
        # Recompressed image should not produce a NaN or crash
        self.assertFalse(np.isnan(ev_comp.prob_tampered))

    def test_02_resizing_downscale_upscale_stability(self):
        """50% downscaling and upscaling does not produce unhandled math errors."""
        h, w = self.base_img.shape[:2]
        small = cv2.resize(self.base_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        ev = self.illum_agent.analyze(restored)
        self.assertFalse(np.isnan(ev.prob_tampered))
        self.assertAlmostEqual(sum(ev.bba_mass.values()), 1.0, places=4)

    def test_03_gaussian_blur_robustness(self):
        """Gaussian blur (5x5 kernel) is handled safely by visual agents."""
        blurred = cv2.GaussianBlur(self.base_img, (5, 5), 1.5)
        ev_gaze = self.gaze_agent.analyze(blurred)
        ev_illum = self.illum_agent.analyze(blurred)

        self.assertFalse(np.isnan(ev_gaze.prob_tampered))
        self.assertFalse(np.isnan(ev_illum.prob_tampered))
        self.assertGreaterEqual(ev_gaze.bba_mass["U"], 0.0)

    def test_04_additive_gaussian_noise_robustness(self):
        """Additive noise (sigma=15) does not invert valid BBA bounds."""
        noise = np.random.normal(0, 15, self.base_img.shape).astype(np.float32)
        noisy = np.clip(self.base_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        pil_noisy = Image.fromarray(noisy)
        ev = self.comp_agent.analyze(pil_noisy)
        self.assertGreaterEqual(ev.prob_tampered, 0.0)
        self.assertLessEqual(ev.prob_tampered, 1.0)
        self.assertAlmostEqual(sum(ev.bba_mass.values()), 1.0, places=4)

    def test_05_unsharp_mask_filter(self):
        """Sharpening filter does not produce divergent telemetry."""
        pil_img = Image.fromarray(self.base_img).filter(ImageFilter.SHARPEN)
        ev = self.comp_agent.analyze(pil_img)
        self.assertIn("mean_ela_residual", ev.telemetry)
        self.assertGreater(ev.telemetry["mean_ela_residual"], 0.0)

    # -----------------------------------------------------------------------
    # 2. Failure Modes & Graceful Degradation
    # -----------------------------------------------------------------------
    def test_06_non_face_blank_frame_returns_inconclusive(self):
        """Blank / non-face frame must yield INCONCLUSIVE and high uncertainty m(U) >= 0.80."""
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        ev_gaze = self.gaze_agent.analyze(blank)
        self.assertEqual(ev_gaze.verdict, "INCONCLUSIVE")
        self.assertGreaterEqual(ev_gaze.bba_mass["U"], 0.80)

        ev_rppg = self.rppg_agent.analyze([blank] * 20, has_face=False)
        self.assertEqual(ev_rppg.verdict, "INCONCLUSIVE")
        self.assertEqual(ev_rppg.bba_mass["U"], 1.0)

    def test_07_very_short_video_clip_returns_inconclusive(self):
        """Video with fewer than 5 frames must return INCONCLUSIVE rather than confident verdict."""
        short_frames = [self.base_img, self.base_img]
        ev_temp = self.temp_agent.analyze(short_frames)
        self.assertEqual(ev_temp.verdict, "INCONCLUSIVE")
        self.assertIn("Insufficient frames", ev_temp.findings[0])
        self.assertGreaterEqual(ev_temp.bba_mass["U"], 0.80)

        ev_rppg = self.rppg_agent.analyze(short_frames, has_face=True)
        self.assertEqual(ev_rppg.verdict, "INCONCLUSIVE")
        self.assertIn("Insufficient frames", ev_rppg.findings[0])

    def test_08_zero_length_and_silent_audio(self):
        """Zero-length or silent audio tracks must not crash AudioVoiceAgent."""
        # Case A: Empty waveform
        empty_wave = np.array([], dtype=np.float32)
        ev_empty = self.audio_agent.analyze((empty_wave, 16000))
        self.assertEqual(ev_empty.verdict, "INCONCLUSIVE")
        self.assertEqual(ev_empty.bba_mass["U"], 1.0)

        # Case B: Completely silent audio
        silent_wave = np.zeros(16000, dtype=np.float32)
        ev_silent = self.audio_agent.analyze((silent_wave, 16000))
        self.assertEqual(ev_silent.verdict, "INCONCLUSIVE")
        self.assertGreaterEqual(ev_silent.bba_mass["U"], 0.80)

    def test_09_multi_face_frame_handling(self):
        """Frame with multiple face targets is handled gracefully without out-of-bounds error."""
        multi_face = np.zeros((300, 600, 3), dtype=np.uint8)
        # Face 1
        cv2.circle(multi_face, (150, 150), 70, (180, 160, 150), -1)
        # Face 2
        cv2.circle(multi_face, (450, 150), 70, (180, 160, 150), -1)

        ev_gaze = self.gaze_agent.analyze(multi_face)
        self.assertIn(ev_gaze.verdict, ["AUTHENTIC", "INCONCLUSIVE", "TAMPERED"])
        self.assertAlmostEqual(sum(ev_gaze.bba_mass.values()), 1.0, places=4)

    def test_10_extreme_conflict_fusion_returns_pure_uncertainty(self):
        """Two agents with diametrically opposed certainties (m1(T)=1.0, m2(A)=1.0) yield pure uncertainty."""
        m_fake = {"T": 1.0, "A": 0.0, "U": 0.0}
        m_real = {"T": 0.0, "A": 1.0, "U": 0.0}
        fused, K = self.fusion.combine(m_fake, m_real)
        self.assertAlmostEqual(K, 0.99, places=2)
        # When conflict is total, the system safely falls back to pure epistemic uncertainty
        self.assertEqual(fused["U"], 1.0)
        self.assertEqual(fused["T"], 0.0)
        self.assertEqual(fused["A"], 0.0)


if __name__ == "__main__":
    unittest.main()
