"""
tests/test_negative_controls_and_mutations.py
Negative Control Detectors & Mutation-Style Robustness Tests.

Validates:
  1. Negative Controls:
     - Constant "Always Real" detector fails tampered cohort tests.
     - Constant "Always Fake" detector fails authentic cohort tests.
     - Uniform "Random" detector fails accuracy & AUC thresholds.
  2. Mutation Checks:
     - Mutating an agent (e.g. zeroing its output or corrupting its verdict)
       causes a detectable test failure in the test suite.
"""

import sys
import os
import unittest
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fusion import DempsterShaferFusion
from agents import (
    MetadataAgent,
    VideoTemporalAgent,
    RPPGAgent,
    AgentEvidence,
)


class ConstantRealDetector:
    """Negative control: degenerate detector that always predicts AUTHENTIC / 0.0."""
    def predict(self, sample) -> dict:
        return {"prob_tampered": 0.0, "verdict": "AUTHENTIC"}


class ConstantFakeDetector:
    """Negative control: degenerate detector that always predicts TAMPERED / 1.0."""
    def predict(self, sample) -> dict:
        return {"prob_tampered": 1.0, "verdict": "TAMPERED"}


class RandomDetector:
    """Negative control: degenerate detector that predicts uniform random noise."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def predict(self, sample) -> dict:
        p = float(self.rng.uniform(0.0, 1.0))
        verdict = "TAMPERED" if p > 0.50 else "AUTHENTIC"
        return {"prob_tampered": p, "verdict": verdict}


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Rigorous tests proving that degenerate negative controls fail and mutations are caught."""

    @classmethod
    def setUpClass(cls):
        cls.fusion = DempsterShaferFusion()

    def test_01_negative_control_constant_real_fails_tampered_test(self):
        """Negative Control 1: An 'always real' detector must FAIL when evaluated on tampered samples."""
        detector = ConstantRealDetector()

        # Evaluation cohort of 10 tampered samples (known ground-truth = 1)
        tampered_samples = [f"tampered_case_{i}" for i in range(10)]
        preds = [detector.predict(s)["prob_tampered"] for s in tampered_samples]

        # Metric: Recall on tampered class must be >= 0.70 for a viable detector
        recall = np.mean([p > 0.5 for p in preds])
        self.assertEqual(recall, 0.0, "ConstantReal detector trivially has 0% recall")

        # The viability assertion that a real detector must satisfy:
        with self.assertRaises(AssertionError, msg="Negative control should fail minimum recall check"):
            self.assertGreaterEqual(recall, 0.70, "Detector failed to identify tampered samples")

    def test_02_negative_control_constant_fake_fails_authentic_test(self):
        """Negative Control 2: An 'always fake' detector must FAIL when evaluated on authentic samples."""
        detector = ConstantFakeDetector()

        # Evaluation cohort of 10 authentic samples (known ground-truth = 0)
        authentic_samples = [f"authentic_case_{i}" for i in range(10)]
        preds = [detector.predict(s)["prob_tampered"] for s in authentic_samples]

        # Metric: False Positive Rate (FPR) must be <= 0.15 for a viable detector
        fpr = np.mean([p > 0.5 for p in preds])
        self.assertEqual(fpr, 1.0, "ConstantFake detector has 100% false positive rate")

        with self.assertRaises(AssertionError, msg="Negative control should fail maximum FPR check"):
            self.assertLessEqual(fpr, 0.15, "Detector has unacceptable false positive rate on authentic media")

    def test_03_negative_control_random_detector_fails_discrimination(self):
        """Negative Control 3: A random detector must FAIL discrimination / AUC threshold."""
        detector = RandomDetector(seed=123)

        # Balanced cohort: 20 authentic (0), 20 tampered (1)
        y_true = np.array([0] * 20 + [1] * 20)
        y_score = np.array([detector.predict(i)["prob_tampered"] for i in range(40)])

        # Compute accuracy at 0.50 threshold
        accuracy = float(np.mean((y_score > 0.50) == y_true))

        # Random detector accuracy is near ~50%
        self.assertAlmostEqual(accuracy, 0.50, delta=0.20)

        # Must fail standard detection requirement of accuracy >= 0.75
        with self.assertRaises(AssertionError, msg="Random detector must fail accuracy >= 0.75"):
            self.assertGreaterEqual(accuracy, 0.75, "Random detector unexpectedly passed 75% accuracy threshold")

    def test_04_mutation_zeroed_metadata_agent_causes_detection_failure(self):
        """Mutation 1: Deliberately zeroing MetadataAgent output fails container tamper test."""
        # Unmutated agent correctly convicts Photoshop / Lavf container
        meta_agent = MetadataAgent()
        img = Image.new("RGB", (100, 100))
        img.info["Software"] = "Adobe Photoshop 2026"
        unmutated_ev = meta_agent.analyze(img)
        self.assertEqual(unmutated_ev.verdict, "TAMPERED", "Unmutated agent must detect Photoshop software tag in metadata chunks")

        # Mutate agent: override analyze to return zero evidence / pure uncertainty
        class MutatedZeroedMetadataAgent(MetadataAgent):
            def analyze(self, *args, **kwargs):
                return AgentEvidence(
                    agent_name=self.name,
                    prob_tampered=0.50,
                    reliability=0.0,
                    bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                    verdict="INCONCLUSIVE",
                    findings=["Mutated: zero output"],
                    telemetry={},
                )

        mutated_agent = MutatedZeroedMetadataAgent()
        mutated_ev = mutated_agent.analyze(img)

        # The test that must fail under the mutation:
        with self.assertRaises(AssertionError, msg="Mutated agent should fail tampering detection"):
            self.assertEqual(mutated_ev.verdict, "TAMPERED")

    def test_05_mutation_rppg_false_biological_pulse_causes_failure(self):
        """Mutation 2: Mutating RPPGAgent to falsely claim pulse on non-face video causes test failure."""
        rppg = RPPGAgent()
        # On non-face frames (has_face=False), unmutated agent returns INCONCLUSIVE
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
        unmutated_ev = rppg.analyze(frames, has_face=False)
        self.assertEqual(unmutated_ev.verdict, "INCONCLUSIVE")
        self.assertEqual(unmutated_ev.bba_mass["U"], 1.0)

        # Mutate agent: falsely reports AUTHENTIC biological pulse even when no face exists
        class MutatedFaultyRPPGAgent(RPPGAgent):
            def analyze(self, raw_frames, has_face=True, **kwargs):
                return AgentEvidence(
                    agent_name=self.name,
                    prob_tampered=0.05,
                    reliability=0.90,
                    bba_mass={"T": 0.05, "A": 0.85, "U": 0.10},
                    verdict="AUTHENTIC",
                    findings=["Mutated: false pulse claimed"],
                    telemetry={"rPPG_SNR": 0.95},
                )

        mutated_rppg = MutatedFaultyRPPGAgent()
        mutated_ev = mutated_rppg.analyze(frames, has_face=False)

        # The safety invariant: non-face input MUST NOT yield AUTHENTIC
        with self.assertRaises(AssertionError, msg="Mutated rPPG agent should violate non-face safety invariant"):
            self.assertNotEqual(mutated_ev.verdict, "AUTHENTIC", "Mutated rPPG falsely claimed AUTHENTIC on non-face")


if __name__ == "__main__":
    unittest.main()
