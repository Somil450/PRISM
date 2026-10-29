"""
tests/test_roc_metrics.py
Unit tests for rigorous ROC / AUC / EER calculation, average-rank tie handling,
constant detector baseline AUC=0.5, and bootstrap CI sample size guards.
"""

import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import rankdata


def compute_roc_metrics(y_true: np.ndarray, y_score: np.ndarray):
    """
    Computes AUC, EER, and TPR@1%FPR with proper tie handling via average ranks
    and linear interpolation for EER.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.0

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5, 0.0

    # Constant detector baseline: all scores identical -> AUC = 0.5 exactly
    if len(np.unique(y_score)) <= 1:
        return 0.5, 0.5, 0.0

    try:
        auc = float(roc_auc_score(y_true, y_score))
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr

        # Exact EER with linear interpolation at zero crossing
        diffs = fpr - fnr
        zero_crossings = np.where(np.diff(np.sign(diffs)))[0]
        if len(zero_crossings) > 0:
            idx = zero_crossings[0]
            denom = (fpr[idx + 1] - fpr[idx]) - (fnr[idx + 1] - fnr[idx])
            if abs(denom) > 1e-9:
                t = (fnr[idx] - fpr[idx]) / denom
                t = np.clip(t, 0.0, 1.0)
                eer = float(fpr[idx] + t * (fpr[idx + 1] - fpr[idx]))
            else:
                eer = float((fpr[idx] + fnr[idx]) / 2.0)
        else:
            idx = int(np.argmin(np.abs(diffs)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)

        # TPR @ 1% FPR
        valid_fpr = np.where(fpr <= 0.01)[0]
        tpr1 = float(tpr[valid_fpr[-1]]) if len(valid_fpr) > 0 else 0.0

    except Exception:
        auc, eer, tpr1 = 0.5, 0.5, 0.0

    return round(auc, 4), round(eer, 4), round(tpr1, 4)


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_bootstraps: int = 1000, seed: int = 42):
    """
    Computes 95% bootstrap confidence intervals for AUC.
    Refuses to compute CI when either class has < 10 samples; returns sentinel indicator.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos < 10 or n_neg < 10:
        if n_pos < 10 and n_neg < 10:
            return None, "insufficient samples (<10/class)"
        elif n_pos < 10:
            return None, "insufficient positives"
        else:
            return None, "insufficient negatives"

    rng = np.random.RandomState(seed)
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idxs]
        ys_b = y_score[idxs]
        if len(np.unique(yt_b)) < 2:
            continue
        auc_b, _, _ = compute_roc_metrics(yt_b, ys_b)
        boot_aucs.append(auc_b)

    if not boot_aucs:
        return None, "insufficient variance in bootstrap"

    ci_low = round(float(np.percentile(boot_aucs, 2.5)), 3)
    ci_high = round(float(np.percentile(boot_aucs, 97.5)), 3)
    return (ci_low, ci_high), f"[{ci_low:.3f}, {ci_high:.3f}]"


class TestROCMetrics(unittest.TestCase):

    def test_01_constant_detector_yields_exact_point_five(self):
        """A detector returning constant scores must yield AUC = 0.5 exactly."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Constant zero
        auc, eer, tpr1 = compute_roc_metrics(y_true, np.zeros(8))
        self.assertEqual(auc, 0.5000, "Constant zeros failed to give AUC = 0.5")
        self.assertEqual(eer, 0.5000, "Constant zeros failed to give EER = 0.5")

        # Constant one (majority class predictor)
        auc, eer, tpr1 = compute_roc_metrics(y_true, np.ones(8))
        self.assertEqual(auc, 0.5000, "Constant ones failed to give AUC = 0.5")
        self.assertEqual(eer, 0.5000, "Constant ones failed to give EER = 0.5")

        # Constant 0.5
        auc, eer, tpr1 = compute_roc_metrics(y_true, np.full(8, 0.5))
        self.assertEqual(auc, 0.5000, "Constant 0.5 failed to give AUC = 0.5")

    def test_02_majority_class_baseline_imbalance(self):
        """Imbalanced classes with constant prediction must still yield AUC = 0.5 exactly."""
        # 3 positives, 7 negatives
        y_true = np.array([0] * 7 + [1] * 3)
        score_majority = np.zeros(10)  # Predicts majority (negatives)
        auc, eer, tpr1 = compute_roc_metrics(y_true, score_majority)
        self.assertEqual(auc, 0.5000, f"Majority baseline on imbalanced data gave {auc}, expected 0.5000")

    def test_03_tie_handling_matches_mann_whitney_average_rank(self):
        """Tied scores must use average ranks (Mann-Whitney U equivalent)."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.2, 0.5, 0.5, 0.8, 0.9])

        # Analytical Mann-Whitney U calculation
        r = rankdata(y_score, method="average")
        n_pos, n_neg = 3, 2
        u = np.sum(r[y_true == 1]) - n_pos * (n_pos + 1) / 2.0
        expected_auc = u / (n_pos * n_neg)

        auc, _, _ = compute_roc_metrics(y_true, y_score)
        self.assertAlmostEqual(auc, expected_auc, places=4)
        self.assertAlmostEqual(auc, 0.9167, places=4)

    def test_04_perfect_and_inverted_detectors(self):
        """Perfect detector -> AUC=1.0, Inverted -> AUC=0.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score_perf = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc_p, eer_p, tpr1_p = compute_roc_metrics(y_true, y_score_perf)
        self.assertEqual(auc_p, 1.0)
        self.assertEqual(eer_p, 0.0)
        self.assertEqual(tpr1_p, 1.0)

        y_score_inv = 1.0 - y_score_perf
        auc_i, eer_i, tpr1_i = compute_roc_metrics(y_true, y_score_inv)
        self.assertEqual(auc_i, 0.0)
        self.assertEqual(eer_i, 1.0)

    def test_05_bootstrap_ci_refuses_when_under_10_samples_per_class(self):
        """Refuses to report CI when either class has < 10 samples; prints 'insufficient positives'."""
        # Only 5 positives, 20 negatives
        y_true = np.array([0] * 20 + [1] * 5)
        y_score = np.linspace(0, 1, 25)
        ci_tuple, ci_msg = bootstrap_ci(y_true, y_score)
        self.assertIsNone(ci_tuple)
        self.assertEqual(ci_msg, "insufficient positives")

        # 20 positives, only 4 negatives
        y_true_rev = np.array([0] * 4 + [1] * 20)
        y_score_rev = np.linspace(0, 1, 24)
        ci_tuple, ci_msg = bootstrap_ci(y_true_rev, y_score_rev)
        self.assertIsNone(ci_tuple)
        self.assertEqual(ci_msg, "insufficient negatives")

        # Sufficient samples (15 positives, 15 negatives) -> CI must be computed
        rng = np.random.RandomState(42)
        y_true_suff = np.array([0] * 15 + [1] * 15)
        y_score_suff = y_true_suff * 0.5 + rng.normal(0, 0.3, 30)
        ci_tuple, ci_msg = bootstrap_ci(y_true_suff, y_score_suff, n_bootstraps=100)
        self.assertIsNotNone(ci_tuple)
        self.assertIsInstance(ci_tuple, tuple)
        self.assertLessEqual(ci_tuple[0], ci_tuple[1])


if __name__ == "__main__":
    unittest.main()
