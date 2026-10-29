"""
tests/test_all_agents.py
Comprehensive End-to-End Test Suite for PRISM.

Tests all 8 forensic modules, neural backbone, mathematical fusion,
and PDF report exporter under isolated test conditions.
"""

import sys
import os
import io
import unittest
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import PRISMNetwork
from preprocessor import PRISMPreprocessor
from fusion import DempsterShaferFusion
from report_exporter import ForensicReportExporter
from agents import (
    MetadataAgent,
    EyeGazeAgent,
    CompressionWatermarkAgent,
    AudioVoiceAgent,
    VideoTemporalAgent,
    IlluminationAgent,
    LipSyncAgent,
)


class TestPRISMSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prep = PRISMPreprocessor()
        cls.fusion = DempsterShaferFusion()
        cls.exporter = ForensicReportExporter()
        cls.meta_ag = MetadataAgent()
        cls.gaze_ag = EyeGazeAgent()
        cls.comp_ag = CompressionWatermarkAgent()
        cls.audio_ag = AudioVoiceAgent()
        cls.vid_ag = VideoTemporalAgent()
        cls.illum_ag = IlluminationAgent()
        cls.lip_ag = LipSyncAgent()

    def test_01_preprocessor_shape(self):
        dummy_bgr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        rgb_t, noise_t = self.prep.process_frame(dummy_bgr)
        self.assertEqual(tuple(rgb_t.shape), (3, 224, 224))
        self.assertEqual(tuple(noise_t.shape), (1, 224, 224))

    def test_02_metadata_provenance(self):
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.info["Software"] = "ChatGPT / DALL-E 3"
        ev = self.meta_ag.analyze(img)
        self.assertEqual(ev.verdict, "TAMPERED")
        self.assertGreater(ev.bba_mass["T"], 0.80)

    def test_03_corneal_eye_gaze(self):
        # Non-face image should gracefully return pure uncertainty m_U = 1.0
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        ev = self.gaze_ag.analyze(blank)
        self.assertEqual(ev.verdict, "INCONCLUSIVE")
        self.assertEqual(ev.bba_mass["U"], 1.0)

    def test_04_compression_watermark(self):
        # Case A: Authentic clean image with low ELA residual
        clean_img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        ev_clean = self.comp_ag.analyze(clean_img)
        self.assertEqual(ev_clean.verdict, "AUTHENTIC", "Uniform clean image must be AUTHENTIC")
        self.assertLess(ev_clean.prob_tampered, 0.35, "Clean image must have low prob_tampered")
        self.assertLess(ev_clean.telemetry["mean_ela_residual"], 0.05, "Clean image ELA residual must be near zero")

        # Case B: Tampered image with high-contrast localized splice boundary
        # Inject artificial sharp localized block splicing
        tampered_arr = np.full((224, 224, 3), 128, dtype=np.uint8)
        tampered_arr[60:140, 60:140] = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        tampered_img = Image.fromarray(tampered_arr)
        ev_tampered = self.comp_ag.analyze(tampered_img)
        self.assertGreater(ev_tampered.prob_tampered, ev_clean.prob_tampered,
                           "Tampered splice must produce higher tampering probability than clean image")
        self.assertGreater(ev_tampered.telemetry["mean_ela_residual"], ev_clean.telemetry["mean_ela_residual"],
                           "Tampered splice must have higher ELA residual than clean image")
        self.assertAlmostEqual(sum(ev_tampered.bba_mass.values()), 1.0, places=4)

    def test_05_audio_voice_antispoof(self):
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        # Pure tone (steep cutoff)
        synth_wave = (np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        ev = self.audio_ag.analyze((synth_wave, sr))
        self.assertIsNotNone(ev.visual_artifact)
        self.assertGreater(ev.prob_tampered, 0.50)

    def test_06_video_temporal_consistency(self):
        # Continuous smooth pan video
        pan_frames = []
        canvas = np.random.randint(50, 200, (260, 260, 3), dtype=np.uint8)
        for i in range(8):
            pan_frames.append(canvas[i * 2:i * 2 + 224, i * 2:i * 2 + 224])
        ev = self.vid_ag.analyze(pan_frames)
        self.assertEqual(ev.verdict, "AUTHENTIC")
        self.assertLess(ev.prob_tampered, 0.35)

    def test_07_illumination_consistency(self):
        # Consistent linear gradient
        h, w = 224, 224
        y, x = np.mgrid[0:h, 0:w]
        grad_rgb = cv2.cvtColor((np.clip(255 - (x * 0.5 + y * 0.5), 20, 240)).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        ev = self.illum_ag.analyze(grad_rgb)
        self.assertEqual(ev.verdict, "AUTHENTIC")
        self.assertLess(ev.telemetry["angular_divergence_deg"], 35.0)

    def test_08_lip_sync_cross_modal(self):
        num_frames = 20
        sr = 16000
        fps = 30.0
        t_audio = np.linspace(0, num_frames / fps, int(sr * (num_frames / fps)))
        audio_env = np.exp(-((t_audio - 0.3)**2) / (2 * 0.08**2))
        audio_wave = (np.sin(2 * np.pi * 300 * t_audio) * audio_env).astype(np.float32)

        synced_frames = []
        for i in range(num_frames):
            canvas = np.zeros((224, 224, 3), dtype=np.uint8)
            open_val = int(255 * np.exp(-(((i / fps) - 0.3)**2) / (2 * 0.08**2)))
            canvas[150:150 + max(4, open_val // 10), 90:134] = open_val
            synced_frames.append(canvas)

        ev = self.lip_ag.analyze((synced_frames, (audio_wave, sr)))
        self.assertEqual(ev.verdict, "AUTHENTIC")
        self.assertLess(abs(ev.telemetry["temporal_lag_ms"]), 45.0)

    def test_09_dempster_shafer_bounds(self):
        m1 = self.fusion.create_mass(evidence_T=0.90 * 10.0 * 0.85, evidence_A=(1.0 - 0.90) * 10.0 * 0.85)
        m2 = self.fusion.create_mass(evidence_T=0.80 * 10.0 * 0.75, evidence_A=(1.0 - 0.80) * 10.0 * 0.75)
        fused = self.fusion.fuse_multimodal_evidence([m1, m2])
        self.assertEqual(fused["verdict"], "TAMPERED")
        self.assertLessEqual(fused["belief"], fused["plausibility"])
        self.assertAlmostEqual(fused["T"] + fused["A"] + fused["U"], 1.0, places=4)

    def test_10_pdf_report_exporter(self):
        dummy_masses = {"Agent_1": {"T": 0.9, "A": 0.05, "U": 0.05}}
        pdf_bytes = self.exporter.generate_pdf(
            filename="test_case_001.png",
            file_bytes=b"dummy_test_bytes",
            verdict="TAMPERED",
            score=0.92,
            belief=0.90,
            plausibility=0.95,
            agent_masses=dummy_masses,
            llm_report_text="Certified forensic lab report explanation.",
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_11_trained_weights_loading(self):
        checkpoints_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints"))
        final_pth = os.path.join(checkpoints_dir, "prism_model_final.pth")
        if not os.path.isfile(final_pth): raise unittest.SkipTest("Trained weights not found locally. Run py scripts/download_weights.py")
        
        import torch
        model = PRISMNetwork(pretrained=False)
        state = torch.load(final_pth, map_location="cpu")
        cleaned_state = {k.replace("module.", ""): v for k, v in state.items()}
        res = model.load_state_dict(cleaned_state, strict=False)
        self.assertEqual(len(res.missing_keys), 0)
        self.assertEqual(len(res.unexpected_keys), 0)

        # Smoke test forward
        rgb = torch.randn(1, 3, 224, 224)
        noise = torch.randn(1, 1, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(rgb, noise)
        self.assertEqual(out["logit"].shape, (1, 1))
        self.assertEqual(out["mask"].shape, (1, 1, 224, 224))

    def test_12_adaptive_reasoner_no_ollama(self):
        from reasoner import PRISMReasoner
        reasoner = PRISMReasoner()
        report = reasoner.generate_report(
            score=0.88,
            mask_area="15.0% of frame",
            noise_status="Anomalous PRNU",
            system_verdict="TAMPERED",
            vit_mass=0.85,
            prnu_mass=0.75,
            ela_mass=0.70,
            meta_mass=0.60,
            gaze_mass=0.65,
            belief=0.82,
            plausibility=0.94,
        )
        self.assertIn("### 1. Case Summary", report)
        self.assertIn("### 2. Methodology & Findings per Evidence Stream", report)
        self.assertIn("### 3. Statistical Confidence", report)
        self.assertIn("### 4. Examiner Conclusion", report)
        self.assertNotIn("LLM reasoning layer is currently unavailable", report)


if __name__ == "__main__":
    unittest.main()

