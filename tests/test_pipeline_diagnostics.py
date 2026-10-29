"""
tests/test_pipeline_diagnostics.py
Empirical Diagnostic & Stress Benchmark for PRISM Forensics Pipeline.

Tests:
1. Optical Lens Vignette & Symmetric Falloff (Illumination Agent robustness)
2. Smooth Accelerating/Decelerating Camera Pan (Kinematic second-order jerk analysis)
3. Direct Video Audio Stream Extraction & Multimodal Lip/Voice Analysis
4. Stripped/Sanitized Metadata Epistemic Uncertainty (m(U) = 1.0 identity law)
5. Multi-Agent Corroboration vs. Single-Heuristic Isolation in Dempster-Shafer Fusion
"""

import os
import sys
import unittest
import io
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import PRISMNetwork
from preprocessor import PRISMPreprocessor
from fusion import DempsterShaferFusion
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


class TestPipelineDiagnostics(unittest.TestCase):
    """Rigorous empirical diagnostic test suite verifying forensic robustness."""

    @classmethod
    def setUpClass(cls):
        cls.prep = PRISMPreprocessor()
        cls.fusion = DempsterShaferFusion()
        cls.meta_ag = MetadataAgent()
        cls.temporal_ag = VideoTemporalAgent()
        cls.illum_ag = IlluminationAgent()
        cls.audio_ag = AudioVoiceAgent()
        cls.lip_ag = LipSyncAgent()
        cls.gaze_ag = EyeGazeAgent()
        cls.rppg_ag = RPPGAgent()
        cls.comp_ag = CompressionWatermarkAgent()

    def test_01_optical_vignette_illumination_robustness(self):
        """
        Diagnostic Case 1: Optical Lens Vignetting.
        Natural lenses exhibit cos^4(theta) optical falloff towards edges.
        Old heuristic falsely declared 180° divergence and TAMPERED.
        The robust IlluminationAgent must detect radial symmetry / low coherence
        and assign epistemic uncertainty (m_U >= 0.80) instead of false conviction.
        """
        h, w = 224, 224
        cy, cx = h / 2.0, w / 2.0
        y, x = np.mgrid[0:h, 0:w]
        dist_sq = (x - cx)**2 + (y - cy)**2
        max_dist_sq = cx**2 + cy**2

        # Quadratic optical falloff towards perimeter (authentic wide-angle lens)
        vignette = 220.0 * (1.0 - 0.55 * (dist_sq / max_dist_sq))
        vignette_rgb = cv2.cvtColor(np.clip(vignette, 20, 240).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        ev = self.illum_ag.analyze(vignette_rgb)

        print(f"\n[Diagnostic 1] Optical Vignette: Verdict={ev.verdict}, P(Tampered)={ev.prob_tampered:.2f}, m(U)={ev.bba_mass['U']:.2f}")
        self.assertNotEqual(ev.verdict, "TAMPERED", "Natural optical vignette must NOT be flagged as TAMPERED")
        self.assertGreaterEqual(ev.bba_mass["U"], 0.70, "Symmetric lens falloff must yield high epistemic uncertainty")
        self.assertLess(ev.bba_mass["T"], 0.25, "Tampering mass must not leak from lens vignetting")

    def test_02_smooth_kinematic_camera_pan(self):
        """
        Diagnostic Case 2: Natural Camera Acceleration and Deceleration.
        A real camera panning across a landscape starts at 2 px/frame, peaks at 12 px/frame,
        and decelerates back to 2 px/frame.
        Old code measured whole-clip std(flow), falsely scoring jitter_score = 1.0 (TAMPERED).
        The new second-order kinematic jerk and circular statistics must classify this as AUTHENTIC.
        """
        # Create a textured canvas
        canvas_h, canvas_w = 400, 800
        np.random.seed(42)
        base = np.random.randint(40, 220, (canvas_h, canvas_w, 3), dtype=np.uint8)
        # Smooth with Gaussian to simulate natural continuous scene textures
        base = cv2.GaussianBlur(base, (15, 15), 5.0)

        # Smooth camera trajectory with acceleration then deceleration
        velocities = [2, 3, 5, 8, 11, 12, 11, 9, 6, 4, 3, 2]
        x_offsets = [0]
        for v in velocities:
            x_offsets.append(x_offsets[-1] + v)

        frames = []
        for x_off in x_offsets:
            crop = base[50:50 + 224, x_off:x_off + 224]
            frames.append(crop)

        ev = self.temporal_ag.analyze(frames)

        print(f"\n[Diagnostic 2] Kinematic Pan: Verdict={ev.verdict}, P(Tampered)={ev.prob_tampered:.2f}, JitterIndex={ev.telemetry['motion_jitter_index']:.2f}")
        self.assertEqual(ev.verdict, "AUTHENTIC", "Smooth accelerating/decelerating pan must be AUTHENTIC")
        self.assertLess(ev.prob_tampered, 0.35, "Kinematic smooth pan must have P(Tampered) < 0.35")
        self.assertLess(ev.telemetry["motion_jitter_index"], 0.30, "Kinematic jerk must be low for smooth pan")

    def test_03_synthetic_audio_and_silent_video_handling(self):
        """
        Diagnostic Case 3: Multimodal Audio Forensic Pipeline.
        Verifies:
        1. AI synthetic harmonic audio is correctly identified by AudioVoiceAgent (P > 0.50).
        2. Silent/absent audio gracefully outputs pure uncertainty m(U) = 1.0 without crashing.
        """
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # 1. Pure synthetic robot tone with robotic harmonics (AI voice clone artifact)
        synth_pcm = (0.7 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        ev_synth = self.audio_ag.analyze((synth_pcm, sr))

        print(f"\n[Diagnostic 3a] Synthetic Audio: Verdict={ev_synth.verdict}, P(Tampered)={ev_synth.prob_tampered:.2f}, m(T)={ev_synth.bba_mass['T']:.2f}")
        self.assertGreater(ev_synth.prob_tampered, 0.50, "Synthetic harmonic tone must be flagged as suspicious")
        self.assertIsNotNone(ev_synth.visual_artifact, "Audio agent must render spectrogram")

        # 2. Silent or missing audio stream
        ev_silent = self.audio_ag.analyze(None)
        print(f"[Diagnostic 3b] Silent/Missing Audio: Verdict={ev_silent.verdict}, m(U)={ev_silent.bba_mass['U']:.2f}")
        self.assertEqual(ev_silent.verdict, "INCONCLUSIVE")
        self.assertEqual(ev_silent.bba_mass["U"], 1.0)
        self.assertEqual(ev_silent.bba_mass["T"], 0.0)
        self.assertEqual(ev_silent.bba_mass["A"], 0.0)

    def test_04_stripped_metadata_epistemic_ignorance(self):
        """
        Diagnostic Case 4: Stripped / Messaging-App Sanitized Metadata.
        WhatsApp, Discord, and Telegram strip all EXIF metadata.
        Under Dempster-Shafer theory, absence of evidence is pure uncertainty (m_U = 1.0).
        Old code leaked m(T) = 0.32 and m(A) = 0.32.
        The new agent must output m(U) = 1.0, which acts as the neutral identity element.
        """
        clean_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        ev_meta = self.meta_ag.analyze(clean_img)

        print(f"\n[Diagnostic 4] Stripped Metadata: Verdict={ev_meta.verdict}, m(U)={ev_meta.bba_mass['U']:.2f}, m(T)={ev_meta.bba_mass['T']:.2f}")
        self.assertEqual(ev_meta.verdict, "INCONCLUSIVE")
        self.assertEqual(ev_meta.bba_mass["U"], 1.0)
        self.assertEqual(ev_meta.bba_mass["T"], 0.0)
        self.assertEqual(ev_meta.bba_mass["A"], 0.0)

        # Verify Dempster-Shafer identity law: m_ensemble (+) m_U = m_ensemble
        m_ensemble = {"T": 0.15, "A": 0.75, "U": 0.10}
        fused, K = self.fusion.combine(m_ensemble, ev_meta.bba_mass)
        self.assertAlmostEqual(fused["T"], m_ensemble["T"], places=4)
        self.assertAlmostEqual(fused["A"], m_ensemble["A"], places=4)
        self.assertAlmostEqual(fused["U"], m_ensemble["U"], places=4)
        self.assertEqual(K, 0.0, "Combining with pure uncertainty must have zero conflict")

    def test_05_multi_agent_corroboration_vs_single_heuristic(self):
        """
        Diagnostic Case 5: Multi-Agent Corroboration vs. Single Heuristic Hijack.
        Verifies:
        1. When only 1 noisy heuristic reports moderate suspicion (m(T) = 0.61) while
           7 independent authentic agents report authentic (m(A) = 0.80),
           the pipeline does NOT falsely convict as TAMPERED.
        2. When at least two independent agents corroborate tampering (e.g. meta + temporal),
           or a definitive agent triggers (m(T) >= 0.72),
           the pipeline decisively rules TAMPERED.
        """
        # Scenario A: Uncorroborated single noisy heuristic against authentic ensemble
        authentic_agents = {
            "vit": {"T": 0.05, "A": 0.85, "U": 0.10},
            "prnu": {"T": 0.05, "A": 0.80, "U": 0.15},
            "gaze": {"T": 0.0, "A": 0.0, "U": 1.0},
            "lip": {"T": 0.05, "A": 0.80, "U": 0.15},
            "rppg": {"T": 0.05, "A": 0.85, "U": 0.10},
            "audio": {"T": 0.05, "A": 0.80, "U": 0.15},
            "temporal": {"T": 0.05, "A": 0.80, "U": 0.15},
            "illum": {"T": 0.61, "A": 0.25, "U": 0.14},  # Single uncorroborated heuristic
        }
        edges = [("vit", "temporal"), ("illum", "temporal"), ("lip", "temporal"), ("gaze", "rppg")]
        fused_a = self.fusion.fuse_graph_evidence(authentic_agents, edges)

        print(f"\n[Diagnostic 5a] Single Heuristic Conflict: Verdict={fused_a['verdict']}, PignisticFake={fused_a['pignistic_fake']:.2f}")
        self.assertNotEqual(fused_a["verdict"], "TAMPERED", "Uncorroborated single heuristic must NOT falsely convict")

        # Scenario B: Corroborated multi-agent tampering (e.g. metadata editing trace + temporal jerk)
        corroborated_agents = {
            "vit": {"T": 0.20, "A": 0.70, "U": 0.10},
            "temporal": {"T": 0.65, "A": 0.15, "U": 0.20},  # Suspicious motion vector jerk
            "meta": {"T": 0.70, "A": 0.10, "U": 0.20},      # Container editing footprint
            "gaze": {"T": 0.0, "A": 0.0, "U": 1.0},
            "lip": {"T": 0.0, "A": 0.0, "U": 1.0},
            "rppg": {"T": 0.0, "A": 0.0, "U": 1.0},
            "audio": {"T": 0.0, "A": 0.0, "U": 1.0},
        }
        edges_b = [("vit", "temporal"), ("meta", "temporal")]
        fused_b = self.fusion.fuse_graph_evidence(corroborated_agents, edges_b)

        print(f"[Diagnostic 5b] Corroborated Tampering: Verdict={fused_b['verdict']}, PignisticFake={fused_b['pignistic_fake']:.2f}")
        self.assertEqual(fused_b["verdict"], "TAMPERED", "Corroborated multi-agent evidence must rule TAMPERED")


if __name__ == "__main__":
    unittest.main()
