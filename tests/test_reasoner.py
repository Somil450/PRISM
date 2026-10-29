"""
tests/test_reasoner.py
Unit tests for the upgraded PRISMReasoner layer.
Verifies that the reasoner produces complete, certified, publication-quality forensic
reports without requiring an external Ollama daemon.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from reasoner import PRISMReasoner


class TestPRISMReasoner(unittest.TestCase):

    def setUp(self):
        self.reasoner = PRISMReasoner()

    def test_01_engine_detection(self):
        engine_name = self.reasoner.active_engine
        self.assertIsInstance(engine_name, str)
        self.assertGreater(len(engine_name), 0)

    def test_02_tampered_report_generation(self):
        report = self.reasoner.generate_report(
            score=0.92,
            mask_area="18.5% of frame (concentrated in facial region)",
            noise_status="Anomalous -- PRNU pattern deviation detected",
            system_verdict="TAMPERED",
            vit_mass=0.88,
            prnu_mass=0.81,
            ela_mass=0.74,
            meta_mass=0.65,
            gaze_mass=0.59,
            meta_summary="Software signature indicates Photoshop/Wav2Lip post-processing.",
            gaze_summary="Corneal specular highlight vectors exhibit non-convergent lighting.",
            belief=0.89,
            plausibility=0.96,
        )

        self.assertIsInstance(report, str)
        self.assertIn("### 1. Case Summary", report)
        self.assertIn("### 2. Methodology & Findings per Evidence Stream", report)
        self.assertIn("### 3. Statistical Confidence", report)
        self.assertIn("### 4. Examiner Conclusion", report)
        self.assertIn("TAMPERED", report)
        self.assertNotIn("LLM reasoning layer is currently unavailable", report)
        self.assertNotIn("Failed", report)

    def test_03_authentic_report_generation(self):
        report = self.reasoner.generate_report(
            score=0.08,
            mask_area="0.0% of frame",
            noise_status="Normal -- sensor noise consistent with camera signature",
            system_verdict="AUTHENTIC",
            vit_mass=0.05,
            prnu_mass=0.04,
            ela_mass=0.06,
            meta_mass=0.02,
            gaze_mass=0.03,
            meta_summary="Original camera EXIF tags intact.",
            gaze_summary="Gaze vectors and illumination angles consistent.",
            belief=0.94,
            plausibility=0.99,
        )

        self.assertIn("### 1. Case Summary", report)
        self.assertIn("AUTHENTIC", report)
        self.assertNotIn("LLM reasoning layer is currently unavailable", report)


if __name__ == "__main__":
    unittest.main()
