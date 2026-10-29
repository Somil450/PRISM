"""
src/benchmark.py
Automated Multi-Agent Forensic Benchmark & Metric Evaluation Suite.

Evaluates individual agents and Dempster-Shafer fusion on test cohorts,
computing Accuracy, Precision, Recall, F1-Score, AUC-ROC, EER, and latency.
"""

from typing import Any, Dict, List, Tuple
import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

try:
    from .model import PRISMNetwork
    from .preprocessor import PRISMPreprocessor
    from .fusion import DempsterShaferFusion
    from .agents import (
        MetadataAgent,
        EyeGazeAgent,
        CompressionWatermarkAgent,
        AudioVoiceAgent,
        VideoTemporalAgent,
        IlluminationAgent,
        LipSyncAgent,
    )
except ImportError:
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
    )


class PRISMBenchmarkSuite:
    """
    Automated evaluation framework for measuring per-agent and fused system accuracy.
    """

    def __init__(self) -> None:
        self.preprocessor = PRISMPreprocessor()
        self.fusion = DempsterShaferFusion()
        self.agents = {
            "metadata": MetadataAgent(),
            "eye_gaze": EyeGazeAgent(),
            "compression": CompressionWatermarkAgent(),
            "audio": AudioVoiceAgent(),
            "temporal": VideoTemporalAgent(),
            "illumination": IlluminationAgent(),
            "lip_sync": LipSyncAgent(),
        }

    def _generate_synthetic_cohort(self, num_samples: int = 20) -> List[Dict[str, Any]]:
        """
        Generates a deterministic synthetic test cohort (50% Authentic, 50% Tampered)
        covering Images, Audio, and Video.
        """
        samples = []
        np.random.seed(42)

        for i in range(num_samples):
            is_tampered = (i % 2 == 1)
            sample_type = ["image", "audio", "video"][i % 3]

            if sample_type == "image":
                # Create base gradient image
                h, w = 224, 224
                y, x = np.mgrid[0:h, 0:w]
                base_img = (np.clip(255 - (x * 0.4 + y * 0.4), 30, 230)).astype(np.uint8)
                rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

                if is_tampered:
                    # Inject ELA high-frequency noise & inverted central patch
                    rgb[50:170, 50:170] = np.fliplr(rgb[50:170, 50:170])
                    rgb[60:120, 60:120] = np.clip(rgb[60:120, 60:120].astype(int) + np.random.randint(-40, 40, (60, 60, 3)), 0, 255).astype(np.uint8)

                pil_img = Image.fromarray(rgb)
                samples.append({
                    "id": f"cohort_img_{i:03d}",
                    "type": "image",
                    "data": pil_img,
                    "raw_rgb": rgb,
                    "ground_truth": 1.0 if is_tampered else 0.0,
                })

            elif sample_type == "audio":
                sr = 16000
                t = np.linspace(0, 1.0, sr)
                if is_tampered:
                    # Low-pass vocoder brickwall cutoff with harmonic buzz
                    wave = (0.5 * np.sin(2 * np.pi * 350 * t) + 0.3 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
                else:
                    # Natural continuous broadband sweep
                    wave = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))).astype(np.float32)

                samples.append({
                    "id": f"cohort_aud_{i:03d}",
                    "type": "audio",
                    "data": (wave, sr),
                    "ground_truth": 1.0 if is_tampered else 0.0,
                })

            elif sample_type == "video":
                frames = []
                base_canvas = np.random.randint(40, 210, (260, 260, 3), dtype=np.uint8)
                for step in range(8):
                    if is_tampered and step % 2 == 1:
                        # Deepfake boundary flicker jitter
                        frames.append(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                    else:
                        crop = base_canvas[step * 2:step * 2 + 224, step * 2:step * 2 + 224]
                        frames.append(crop)

                samples.append({
                    "id": f"cohort_vid_{i:03d}",
                    "type": "video",
                    "data": frames,
                    "ground_truth": 1.0 if is_tampered else 0.0,
                })

        return samples

    def run_benchmark(self, num_samples: int = 30) -> Dict[str, Any]:
        """
        Executes full benchmark suite across all agents and multi-agent fusion.
        """
        cohort = self._generate_synthetic_cohort(num_samples)
        
        y_true = []
        y_scores_fused = []
        agent_scores: Dict[str, List[float]] = {name: [] for name in self.agents.keys()}
        agent_y_true: Dict[str, List[float]] = {name: [] for name in self.agents.keys()}
        latencies = []

        for sample in cohort:
            start_t = time.perf_counter()
            gt = sample["ground_truth"]
            y_true.append(gt)

            stype = sample["type"]
            evidence_masses = []

            if stype == "image":
                pil_img = sample["data"]
                rgb = sample["raw_rgb"]

                comp_ev = self.agents["compression"].analyze(pil_img)
                meta_ev = self.agents["metadata"].analyze(pil_img)
                gaze_ev = self.agents["eye_gaze"].analyze(rgb)
                illum_ev = self.agents["illumination"].analyze(pil_img)

                evidence_masses.extend([comp_ev.bba_mass, meta_ev.bba_mass, gaze_ev.bba_mass, illum_ev.bba_mass])

                agent_scores["compression"].append(comp_ev.prob_tampered)
                agent_y_true["compression"].append(gt)

                agent_scores["illumination"].append(illum_ev.prob_tampered)
                agent_y_true["illumination"].append(gt)

                agent_scores["metadata"].append(meta_ev.prob_tampered)
                agent_y_true["metadata"].append(gt)

            elif stype == "audio":
                aud_ev = self.agents["audio"].analyze(sample["data"])
                evidence_masses.append(aud_ev.bba_mass)

                agent_scores["audio"].append(aud_ev.prob_tampered)
                agent_y_true["audio"].append(gt)

            elif stype == "video":
                vid_ev = self.agents["temporal"].analyze(sample["data"])
                evidence_masses.append(vid_ev.bba_mass)

                agent_scores["temporal"].append(vid_ev.prob_tampered)
                agent_y_true["temporal"].append(gt)

            # Dempster-Shafer Multi-Agent Combination
            fused_result = self.fusion.fuse_multimodal_evidence(evidence_masses)
            fused_score = fused_result["pignistic_fake"]
            y_scores_fused.append(fused_score)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)

        # Compute Metrics
        metrics = self._compute_metrics(np.array(y_true), np.array(y_scores_fused))
        metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 2)
        metrics["p95_latency_ms"] = round(float(np.percentile(latencies, 95)), 2)
        metrics["total_samples_evaluated"] = len(cohort)

        per_agent_metrics = {}
        for name, scores in agent_scores.items():
            if len(scores) > 0:
                per_agent_metrics[name] = self._compute_metrics(np.array(agent_y_true[name]), np.array(scores))

        return {
            "fused_system": metrics,
            "per_agent": per_agent_metrics,
        }

    def _compute_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """
        Computes Accuracy, Precision, Recall, F1, and EER.
        """
        y_pred = (y_score >= 0.50).astype(int)
        y_int = y_true.astype(int)

        tp = int(np.sum((y_pred == 1) & (y_int == 1)))
        tn = int(np.sum((y_pred == 0) & (y_int == 0)))
        fp = int(np.sum((y_pred == 1) & (y_int == 0)))
        fn = int(np.sum((y_pred == 0) & (y_int == 1)))

        acc = (tp + tn) / max(1, len(y_true))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * (prec * rec) / max(1e-6, (prec + rec))

        # Equal Error Rate (EER) approximation
        thresholds = np.linspace(0.01, 0.99, 100)
        fpr_list, fnr_list = [], []
        for th in thresholds:
            p = (y_score >= th).astype(int)
            fpr = np.sum((p == 1) & (y_int == 0)) / max(1, np.sum(y_int == 0))
            fnr = np.sum((p == 0) & (y_int == 1)) / max(1, np.sum(y_int == 1))
            fpr_list.append(fpr)
            fnr_list.append(fnr)

        diff = np.abs(np.array(fpr_list) - np.array(fnr_list))
        min_idx = int(np.argmin(diff))
        eer = float((fpr_list[min_idx] + fnr_list[min_idx]) / 2.0)

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "equal_error_rate_eer": round(float(eer), 4),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        }


# ---------------------------------------------------------------------------
# CLI Entrypoint & Evaluation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("PRISM MULTI-AGENT AUTOMATED BENCHMARK & EVALUATION SUITE")
    print("=" * 70)

    suite = PRISMBenchmarkSuite()
    print("Executing benchmark over synthetic multimodal cohort (N=30)...")
    results = suite.run_benchmark(num_samples=30)

    fused = results["fused_system"]
    print("\n" + "-" * 70)
    print("FUSED MULTI-AGENT SYSTEM PERFORMANCE (Dempster-Shafer):")
    print(f"  * Accuracy:              {fused['accuracy'] * 100:.1f}%")
    print(f"  * Precision:             {fused['precision'] * 100:.1f}%")
    print(f"  * Recall (Sensitivity):  {fused['recall'] * 100:.1f}%")
    print(f"  * F1-Score:              {fused['f1_score']:.4f}")
    print(f"  * Equal Error Rate (EER):{fused['equal_error_rate_eer'] * 100:.2f}%")
    print(f"  * Mean Latency / sample: {fused['mean_latency_ms']} ms (P95: {fused['p95_latency_ms']} ms)")
    print(f"  * Confusion Matrix:      TP={fused['true_positives']}, TN={fused['true_negatives']}, FP={fused['false_positives']}, FN={fused['false_negatives']}")
    print("-" * 70)

    print("\nPER-AGENT SPECIALIST PERFORMANCE BREAKDOWN:")
    for agent_name, m in results["per_agent"].items():
        print(f"  * {agent_name.title():<18}: Acc={m['accuracy']*100:>5.1f}% | Prec={m['precision']*100:>5.1f}% | Rec={m['recall']*100:>5.1f}% | F1={m['f1_score']:.3f} | EER={m['equal_error_rate_eer']*100:>4.1f}%")

    print("=" * 70)
    print("Benchmark suite execution completed successfully!")
