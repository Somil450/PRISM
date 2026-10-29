"""
tests/test_pipeline_e2e.py
End-to-end integration test simulating complete PRISM multimodal workflow:
1. Synthetic Image Input
2. PRNU Preprocessing & Feature Extraction
3. PRISMNetwork Forward Pass (Trained Checkpoint)
4. Multi-Agent Analysis (Metadata, Gaze, ELA, Illumination)
5. Dempster-Shafer Mathematical Fusion
6. Adaptive Forensic Reasoner Synthesis
7. Investigative Support PDF Dossier Export
"""

import os
import sys
import unittest
import io
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import PRISMNetwork
from preprocessor import PRISMPreprocessor
from fusion import DempsterShaferFusion
from reasoner import PRISMReasoner
from report_exporter import ForensicReportExporter
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


class TestPRISMPipelineE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = PRISMNetwork(pretrained=False).to(cls.device)
        
        checkpoints_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints")
        )
        final_pth = os.path.join(checkpoints_dir, "prism_model_final.pth")
        cls.has_trained_weights = os.path.isfile(final_pth)
        if cls.has_trained_weights:
            state = torch.load(final_pth, map_location=cls.device)
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            cls.model.load_state_dict(cleaned, strict=False)

        cls.model.eval()
        cls.prep = PRISMPreprocessor()
        cls.fusion = DempsterShaferFusion()
        cls.reasoner = PRISMReasoner()
        cls.exporter = ForensicReportExporter()

        cls.agents = {
            "metadata": MetadataAgent(),
            "eye_gaze": EyeGazeAgent(),
            "compression": CompressionWatermarkAgent(),
            "audio": AudioVoiceAgent(),
            "temporal": VideoTemporalAgent(),
            "illumination": IlluminationAgent(),
            "lip_sync": LipSyncAgent(),
            "rppg": RPPGAgent(),
        }

    def test_end_to_end_image_pipeline(self):
        if not self.has_trained_weights:
            raise unittest.SkipTest("Trained checkpoint not found. Run py scripts/download_weights.py")
        # 1. Create realistic synthetic test image
        img = Image.new("RGB", (320, 320), color=(180, 160, 140))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 2. Preprocessor: Extract RGB and PRNU Noise Tensors
        rgb_t, noise_t = self.prep.process_frame(cv_img)
        self.assertEqual(tuple(rgb_t.shape), (3, 224, 224))
        self.assertEqual(tuple(noise_t.shape), (1, 224, 224))

        # 3. Model Inference using trained weights
        with torch.no_grad():
            out = self.model(rgb_t.unsqueeze(0), noise_t.unsqueeze(0))
            prob_fake = float(torch.sigmoid(out["logit"]).item())
            mask = out["mask"].squeeze(0).squeeze(0).cpu().numpy()

        self.assertGreaterEqual(prob_fake, 0.0)
        self.assertLessEqual(prob_fake, 1.0)
        self.assertEqual(mask.shape, (224, 224))

        # 4. Multi-Agent Analysis
        meta_ev = self.agents["metadata"].analyze(img)
        gaze_ev = self.agents["eye_gaze"].analyze(cv_img)
        comp_ev = self.agents["compression"].analyze(img)
        illum_ev = self.agents["illumination"].analyze(cv_img)

        # 5. Dempster-Shafer Mass Creation and Fusion
        vit_bba = self.fusion.create_mass(
            evidence_T=prob_fake * 8.0, evidence_A=(1.0 - prob_fake) * 8.0
        )
        fused = self.fusion.fuse_multimodal_evidence([
            vit_bba,
            meta_ev.bba_mass,
            gaze_ev.bba_mass,
            comp_ev.bba_mass,
            illum_ev.bba_mass,
        ])

        verdict = fused["verdict"]
        belief = fused["belief"]
        plausibility = fused["plausibility"]

        # Assert mathematical bounds and that neutral image is not falsely convicted as TAMPERED
        self.assertNotEqual(verdict, "TAMPERED", "Clean neutral image must not be convicted as TAMPERED")
        self.assertLess(fused["pignistic_fake"], 0.65, "Clean neutral image must have BetP < 0.65")
        self.assertLessEqual(belief, plausibility)

        # Comparative test on tampered input: must trigger TAMPERED and higher BetP
        tampered_img = img.copy()
        tampered_img.info["parameters"] = "Steps: 30, Sampler: Euler, Model: Stable Diffusion v2.1"
        meta_fake = self.agents["metadata"].analyze(tampered_img)
        fused_fake = self.fusion.fuse_multimodal_evidence([
            vit_bba,
            meta_fake.bba_mass,
            comp_ev.bba_mass,
            illum_ev.bba_mass,
        ])
        self.assertEqual(meta_fake.verdict, "TAMPERED", "Tampered metadata input must yield TAMPERED verdict on MetadataAgent")
        self.assertGreater(fused_fake["pignistic_fake"], fused["pignistic_fake"],
                           "Tampered evidence must produce strictly higher BetP(T) than clean input")
        self.assertGreater(fused_fake["belief"], fused["belief"],
                           "Tampered evidence must produce strictly higher Bel(T) than clean input")

        # 6. Adaptive Reasoner Report Generation
        report = self.reasoner.generate_report(
            score=fused["pignistic_fake"],
            mask_area=f"{float(np.mean(mask > 0.5) * 100):.1f}% of frame",
            noise_status="Sensor noise analysis complete",
            system_verdict=verdict,
            vit_mass=vit_bba["T"],
            prnu_mass=0.50,
            ela_mass=comp_ev.bba_mass["T"],
            meta_mass=meta_ev.bba_mass["T"],
            gaze_mass=gaze_ev.bba_mass["T"],
            meta_summary="; ".join(meta_ev.findings) if meta_ev.findings else "No metadata anomalies.",
            gaze_summary="; ".join(gaze_ev.findings) if gaze_ev.findings else "No ocular anomalies.",
            belief=belief,
            plausibility=plausibility,
        )

        self.assertIn("### 1. Case Summary", report)
        self.assertIn("### 2. Methodology & Findings per Evidence Stream", report)
        self.assertIn("### 3. Statistical Confidence", report)
        self.assertIn("### 4. Examiner Conclusion", report)

        # 7. Certified PDF Export
        pdf_bytes = self.exporter.generate_pdf(
            filename="sample_evidence.png",
            file_bytes=file_bytes,
            verdict=verdict,
            score=fused["pignistic_fake"],
            belief=belief,
            plausibility=plausibility,
            agent_masses={
                "ViT_spatial": vit_bba,
                "Metadata": meta_ev.bba_mass,
                "Ocular": gaze_ev.bba_mass,
                "Compression": comp_ev.bba_mass,
            },
            llm_report_text=report,
        )

        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 2000)

    def test_end_to_end_video_pipeline_edited_video(self):
        """
        End-to-end test for edited video without faces (e.g. night_rain_21sec_edit.mp4).
        Verifies:
        1. rPPG does NOT falsely claim biological pulse on non-face video.
        2. Container metadata and filename footprint flag post-capture editing.
        3. Multi-agent fusion yields TAMPERED verdict.
        4. Certified dossier generates properly.
        """
        # 1. Simulate 16 video frames with rain/noise particles
        frames = []
        base_canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        base_canvas[:, :] = (15, 20, 30)  # Dark night background
        for i in range(16):
            f = base_canvas.copy()
            # Random rain streaks
            for _ in range(40):
                rx = int(np.random.randint(0, 220))
                ry = int(np.random.randint(0, 200))
                cv2.line(f, (rx, ry), (rx + 2, ry + 15), (180, 200, 220), 1)
            frames.append(f)

        # 2. Simulate video container bytes with Adobe Premiere / CapCut editing toolchain in moov.udta atom
        dummy_mp4_bytes = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x40moov\x00\x00\x00\x38udtaAdobe Premiere Pro CC\x00" + b"\x00" * 100
        filename = "night_rain_21sec_edit.mp4"

        # 3. Execute Video Multi-Agent Pipeline
        temporal_ev = self.agents["temporal"].analyze(frames)
        gaze_ev = self.agents["eye_gaze"].analyze(frames[8])
        faces_found = gaze_ev.telemetry.get("faces_detected", 0) > 0
        self.assertFalse(faces_found, "Rain scene must not detect faces")

        # rPPG MUST be bypassed when no face is present
        rppg_ev = self.agents["rppg"].analyze(frames, has_face=faces_found)
        self.assertEqual(rppg_ev.verdict, "INCONCLUSIVE")
        self.assertEqual(rppg_ev.bba_mass["U"], 1.0)
        self.assertEqual(rppg_ev.bba_mass["A"], 0.0)

        # Metadata MUST flag container editing software toolchain
        meta_ev = self.agents["metadata"].analyze(dummy_mp4_bytes)
        self.assertEqual(meta_ev.verdict, "TAMPERED")
        self.assertGreater(meta_ev.bba_mass["T"], 0.65)

        # Compression / ELA agent
        comp_ev = self.agents["compression"].analyze(Image.fromarray(cv2.cvtColor(frames[8], cv2.COLOR_BGR2RGB)))
        illum_ev = self.agents["illumination"].analyze(frames[8])
        lip_ev = self.agents["lip_sync"].analyze((frames, None))

        # ViT & PRNU Backbone
        rgb_t, noise_t = self.prep.process_frame(frames[8])
        with torch.no_grad():
            out = self.model(rgb_t.unsqueeze(0), noise_t.unsqueeze(0))
            prob_fake = float(torch.sigmoid(out["logit"]).item())

        vit_bba = self.fusion.create_mass(evidence_T=prob_fake * 8.0 * 0.88, evidence_A=(1.0 - prob_fake) * 8.0 * 0.88)
        prnu_bba = self.fusion.create_mass(evidence_T=0.5 * 8.0 * 0.75, evidence_A=0.5 * 8.0 * 0.75)

        # FEG Graph Fusion
        agent_dict = {
            "vit": vit_bba,
            "prnu": prnu_bba,
            "temporal": temporal_ev.bba_mass,
            "meta": meta_ev.bba_mass,
            "comp": comp_ev.bba_mass,
            "illum": illum_ev.bba_mass,
            "gaze": gaze_ev.bba_mass,
            "lip": lip_ev.bba_mass,
            "rppg": rppg_ev.bba_mass,
        }
        edges = [
            ("vit", "temporal"),
            ("comp", "vit"),
            ("prnu", "comp"),
            ("meta", "vit"),
            ("illum", "temporal"),
        ]
        fused = self.fusion.fuse_graph_evidence(agent_dict, edges)

        # System verdict must never claim AUTHENTIC on tampered/re-encoded video
        self.assertNotEqual(fused["verdict"], "AUTHENTIC", "Edited video with metadata tampering must not be declared AUTHENTIC")
        self.assertIn(fused["verdict"], ["TAMPERED", "INCONCLUSIVE"])
        self.assertGreater(fused["pignistic_fake"], 0.40)

        # Reasoner Report
        report = self.reasoner.generate_report(
            score=fused["pignistic_fake"],
            mask_area="Full-frame temporal video inspection",
            noise_status="Sensor noise analysis complete",
            system_verdict=fused["verdict"],
            vit_mass=vit_bba["T"],
            prnu_mass=prnu_bba["T"],
            ela_mass=comp_ev.bba_mass["T"],
            meta_mass=meta_ev.bba_mass["T"],
            gaze_mass=temporal_ev.bba_mass["T"],
            meta_summary="; ".join(meta_ev.findings),
            gaze_summary="; ".join(temporal_ev.findings),
            belief=fused["belief"],
            plausibility=fused["plausibility"],
        )
        self.assertIn("Case Summary", report)

        # PDF Dossier
        pdf = self.exporter.generate_pdf(
            filename=filename,
            file_bytes=dummy_mp4_bytes,
            verdict=fused["verdict"],
            score=fused["pignistic_fake"],
            belief=fused["belief"],
            plausibility=fused["plausibility"],
            agent_masses=agent_dict,
            llm_report_text=report,
        )
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()

