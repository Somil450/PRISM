"""
tests/test_fusion_properties.py
Formal Property & Invariant Tests for Dempster-Shafer Evidential Fusion.

Validates the mathematical properties of Dempster-Shafer theory on Frame Omega = {T, A, U}:
  1. Mass Conservation: For all combinations, m(T) + m(A) + m(U) = 1.0
  2. Conflict Bounds: Conflict coefficient K in [0.0, 1.0]
  3. Evidential Ordering: Bel(T) <= BetP(T) <= Pl(T) for all valid BBAs
  4. Neutral Identity Law: Vacuous evidence m(U) = 1.0 leaves beliefs unchanged: m (+) m_U = m
  5. Commutativity: combine(m1, m2) == combine(m2, m1)
  6. Total Conflict Singularity Handling: K -> 1.0 does not raise ZeroDivisionError and yields pure uncertainty
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from fusion import DempsterShaferFusion


class TestDempsterShaferProperties(unittest.TestCase):
    """Rigorous property tests validating mathematical invariants of Dempster's Rule of Combination."""

    @classmethod
    def setUpClass(cls):
        cls.fusion = DempsterShaferFusion()

    def _generate_random_bba(self) -> dict:
        """Generates a random valid BBA where m(T) + m(A) + m(U) == 1.0 and all m >= 0."""
        raw = np.random.exponential(scale=1.0, size=3)
        normalized = raw / np.sum(raw)
        return {"T": float(normalized[0]), "A": float(normalized[1]), "U": float(normalized[2])}

    def test_01_mass_conservation_invariant(self):
        """Property 1: Sum of focal masses must equal 1.0 under all valid combinations."""
        np.random.seed(101)
        for _ in range(100):
            m1 = self._generate_random_bba()
            m2 = self._generate_random_bba()
            fused, _ = self.fusion.combine(m1, m2)
            total_mass = fused["T"] + fused["A"] + fused["U"]
            self.assertAlmostEqual(total_mass, 1.0, places=5,
                                   msg=f"Mass conservation violated: {fused}")
            self.assertGreaterEqual(fused["T"], -1e-7)
            self.assertGreaterEqual(fused["A"], -1e-7)
            self.assertGreaterEqual(fused["U"], -1e-7)

    def test_02_conflict_bounds_invariant(self):
        """Property 2: Conflict coefficient K must strictly reside in [0.0, 1.0]."""
        np.random.seed(102)
        for _ in range(100):
            m1 = self._generate_random_bba()
            m2 = self._generate_random_bba()
            _, K = self.fusion.combine(m1, m2)
            self.assertGreaterEqual(K, 0.0, f"Conflict K < 0.0: {K}")
            self.assertLessEqual(K, 1.0, f"Conflict K > 1.0: {K}")

    def test_03_evidential_ordering_inequality(self):
        """Property 3: Lower Belief <= Pignistic BetP <= Upper Plausibility everywhere."""
        np.random.seed(103)
        for _ in range(100):
            m_list = [self._generate_random_bba() for _ in range(np.random.randint(2, 6))]
            fused = self.fusion.fuse_multimodal_evidence(m_list)
            bel = fused["belief"]
            betp = fused["pignistic_fake"]
            pl = fused["plausibility"]

            self.assertLessEqual(bel, betp + 1e-6,
                                 f"Bel(T) > BetP(T): Bel={bel}, BetP={betp}")
            self.assertLessEqual(betp, pl + 1e-6,
                                 f"BetP(T) > Pl(T): BetP={betp}, Pl={pl}")
            self.assertLessEqual(bel, pl + 1e-6,
                                 f"Bel(T) > Pl(T): Bel={bel}, Pl={pl}")

    def test_04_neutral_identity_element(self):
        """Property 4: Vacuous evidence m(U) = 1.0 acts as the neutral identity element (m (+) m_U = m)."""
        vacuous_mass = {"T": 0.0, "A": 0.0, "U": 1.0}
        np.random.seed(104)
        for _ in range(50):
            m = self._generate_random_bba()
            fused, K = self.fusion.combine(m, vacuous_mass)
            self.assertAlmostEqual(fused["T"], m["T"], places=5)
            self.assertAlmostEqual(fused["A"], m["A"], places=5)
            self.assertAlmostEqual(fused["U"], m["U"], places=5)
            self.assertEqual(K, 0.0, "Combining with vacuous evidence must yield zero conflict")

    def test_05_commutativity(self):
        """Property 5: Dempster combination is commutative: combine(m1, m2) == combine(m2, m1)."""
        np.random.seed(105)
        for _ in range(100):
            m1 = self._generate_random_bba()
            m2 = self._generate_random_bba()
            fused_12, K_12 = self.fusion.combine(m1, m2)
            fused_21, K_21 = self.fusion.combine(m2, m1)

            self.assertAlmostEqual(K_12, K_21, places=6, msg="Conflict K is non-commutative")
            self.assertAlmostEqual(fused_12["T"], fused_21["T"], places=6)
            self.assertAlmostEqual(fused_12["A"], fused_21["A"], places=6)
            self.assertAlmostEqual(fused_12["U"], fused_21["U"], places=6)

    def test_06_total_conflict_singularity_handling(self):
        """Property 6: Total conflict (K -> 1.0) must not raise ZeroDivisionError and yields pure uncertainty."""
        # Perfect contradiction: m1 is 100% Tampered, m2 is 100% Authentic
        m_tampered = {"T": 1.0, "A": 0.0, "U": 0.0}
        m_authentic = {"T": 0.0, "A": 1.0, "U": 0.0}

        try:
            fused, K = self.fusion.combine(m_tampered, m_authentic)
        except ZeroDivisionError:
            self.fail("ZeroDivisionError raised during total conflict combination!")

        # K capped at 0.99
        self.assertAlmostEqual(K, 0.99, places=3)
        # Total numerator is 0, so fused defaults to pure epistemic uncertainty
        self.assertEqual(fused["T"], 0.0)
        self.assertEqual(fused["A"], 0.0)
        self.assertEqual(fused["U"], 1.0)

    def test_07_extreme_edge_cases(self):
        """Property 7: Extreme edge cases (zero mass, high uncertainty, pure certainty)."""
        pure_uncertain = {"T": 0.0, "A": 0.0, "U": 1.0}
        pure_t = {"T": 1.0, "A": 0.0, "U": 0.0}

        # Combining pure certainty with pure uncertainty preserves pure certainty
        fused, K = self.fusion.combine(pure_t, pure_uncertain)
        self.assertAlmostEqual(fused["T"], 1.0, places=5)
        self.assertAlmostEqual(fused["A"], 0.0, places=5)
        self.assertAlmostEqual(fused["U"], 0.0, places=5)
        self.assertEqual(K, 0.0)


if __name__ == "__main__":
    unittest.main()
