"""
scripts/benchmark_df_platter.py
DF-Platter Benchmark Evaluator (IIT Jodhpur, CVPR 2023).

Purpose:
  Evaluates PRISM's multi-agent forensic engine on DF-Platter or local test samples.
  Strictly reports whether the real academic dataset is present or if results
  are from local test samples only. Computes demographic fairness slices,
  TPR, FPR, AUC, 95% bootstrap confidence intervals, and fairness disparity gaps.

Reference:
  Kartik Narayan, Harsh Agarwal, Kartik Thakral, Surbhi Mittal, Mayank Vatsa, Richa Singh.
  "DF-Platter: Multi-Face Heterogeneous Dataset for Evaluating Deepfake Detection."
  IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023.
  Access guide: docs/DF_PLATTER_ACCESS.md
"""

import argparse
import os
import sys
import json
import re
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import cv2
    from agents.video_temporal_agent import VideoTemporalAgent
    from agents.compression_watermark_agent import CompressionWatermarkAgent
    from agents.eye_gaze_agent import EyeGazeAgent
    from agents.rppg_agent import RPPGAgent
    from agents.metadata_agent import MetadataAgent
    from fusion import DempsterShaferFusion
except ImportError as err:
    print(f"Error importing PRISM forensic components: {err}")
    sys.exit(1)


class DFPlatterMetadataLoader:
    """
    Robust metadata loader supporting:
      1. Official CSV / JSON annotations file
      2. Safe fallback parser with strict regex word boundaries to prevent false substring matches
    """
    def __init__(self, annotation_file: Optional[str] = None) -> None:
        self.annotations = {}
        if annotation_file and os.path.isfile(annotation_file):
            self._load_annotations(annotation_file)

    def _load_annotations(self, file_path: str) -> None:
        p = Path(file_path)
        if p.suffix.lower() == ".csv":
            with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row.get("video_id") or row.get("relative_path") or row.get("filename")
                    if key:
                        self.annotations[Path(key).name.lower()] = {
                            "technique": row.get("technique", "unknown"),
                            "compression": row.get("compression", "unknown"),
                            "fitzpatrick": row.get("fitzpatrick", "unknown"),
                            "headwear": row.get("headwear", "unknown"),
                            "facial_hair": row.get("facial_hair", "unknown"),
                            "is_deepfake": bool(int(row.get("is_deepfake", row.get("label", 0)))),
                        }
        elif p.suffix.lower() == ".json":
            with open(file_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    self.annotations[Path(k).name.lower()] = v

    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        fname = file_path.name.lower()
        if fname in self.annotations:
            return self.annotations[fname]

        # Safe Fallback Parser using strict regex word boundaries and path tokens
        path_str = str(file_path).lower().replace("\\", "/")
        tokens = [t.lower() for t in re.split(r"[/_\-\.\s]+", path_str)]

        # 1. Technique
        technique = "unknown"
        if "deepfacelab" in tokens or "dfl" in tokens:
            technique = "DeepFaceLab"
        elif "faceswap" in tokens or ("fs" in tokens and "c23" in tokens):
            technique = "FaceSwap"
        elif "diffusion" in tokens or "ldm" in tokens or "latent" in tokens:
            technique = "LatentDiffusion"
        elif "inswapper" in tokens or "roop" in tokens:
            technique = "FaceSwap"
        elif "authentic" in tokens or "real" in tokens:
            technique = "Authentic"

        # 2. Compression
        compression = "unknown"
        if "c40" in tokens:
            compression = "c40"
        elif "c23" in tokens:
            compression = "c23"
        elif "raw" in tokens:
            compression = "raw"

        # 3. Fitzpatrick
        fitz = "unknown"
        clean_path = path_str.replace("_", " ").replace("-", " ")
        if "fitzpatrick iii" in clean_path or "fitz iii" in clean_path or "fitz3" in tokens or ("fitzpatrick" in tokens and "iii" in tokens):
            fitz = "Fitzpatrick III"
        elif "fitzpatrick iv" in clean_path or "fitz iv" in clean_path or "fitz4" in tokens or ("fitzpatrick" in tokens and "iv" in tokens):
            fitz = "Fitzpatrick IV"
        elif "fitzpatrick v" in clean_path or "fitz v" in clean_path or "fitz5" in tokens or ("fitzpatrick" in tokens and "v" in tokens):
            fitz = "Fitzpatrick V"
        elif "fitzpatrick vi" in clean_path or "fitz vi" in clean_path or "fitz6" in tokens or ("fitzpatrick" in tokens and "vi" in tokens):
            fitz = "Fitzpatrick VI"

        # 4. Cultural Headwear (use exact word matching: 'turban', 'pagri', 'dupatta', 'hijab')
        headwear = "unknown"
        if any(tok in ["turban", "pagri", "dastar", "pagdi"] for tok in tokens):
            headwear = "Turban"
        elif any(tok in ["dupatta", "chunni", "dupatas"] for tok in tokens):
            headwear = "Dupatta"
        elif any(tok in ["hijab", "niqab"] for tok in tokens):
            headwear = "Hijab"
        elif "no_headwear" in tokens or "none" in tokens:
            headwear = "None"

        # 5. Facial Hair (strict matching to avoid substring collision with 'bear' or 'disobedient')
        facial_hair = "unknown"
        if any(tok in ["beard", "bearded", "daadhi", "moustache", "mustache", "mooch"] for tok in tokens):
            facial_hair = "Beard / Moustache"
        elif any(tok in ["cleanshaven", "clean_shaven", "shaven", "no_beard"] for tok in tokens):
            facial_hair = "Clean-shaven"

        # Ground truth
        is_deepfake = technique in ["DeepFaceLab", "FaceSwap", "LatentDiffusion"] or ("fake" in tokens and "authentic" not in tokens)

        return {
            "technique": technique,
            "compression": compression,
            "fitzpatrick": fitz,
            "headwear": headwear,
            "facial_hair": facial_hair,
            "is_deepfake": is_deepfake,
        }


# Backwards compatibility alias
DFPlatterMetadataParser = DFPlatterMetadataLoader



# ---------------------------------------------------------------------------
# Metric Calculations (Imported from eval_core for consistent tie & CI handling)
# ---------------------------------------------------------------------------
try:
    from eval_core import compute_roc_metrics, bootstrap_ci
except ImportError:
    try:
        from scripts.eval_core import compute_roc_metrics, bootstrap_ci
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_core import compute_roc_metrics, bootstrap_ci


class DFPlatterBenchmark:
    """
    Automated benchmark evaluator with dataset presence verification and demographic fairness auditing.
    """

    def __init__(self, annotation_file: Optional[str] = None) -> None:
        self.temporal_agent = VideoTemporalAgent()
        self.eye_agent = EyeGazeAgent()
        self.rppg_agent = RPPGAgent()
        self.comp_agent = CompressionWatermarkAgent()
        self.meta_agent = MetadataAgent()
        self.fusion = DempsterShaferFusion()
        self.metadata_loader = DFPlatterMetadataLoader(annotation_file)

    def is_real_df_platter_present(self, dataset_path: str) -> bool:
        """
        Checks if the provided path is actually the official DF-Platter dataset
        (contains official directory structure or minimum required academic scale > 500 videos).
        """
        p = Path(dataset_path)
        if not p.exists():
            return False

        # Real DF-Platter has thousands of videos and standard splits
        videos = list(p.rglob("*.mp4"))[:501]
        has_academic_scale = len(videos) > 500
        has_annotation_file = (p / "metadata.csv").exists() or (p / "annotations.json").exists()
        has_df_structure = (p / "raw").exists() and (p / "c23").exists() and (p / "c40").exists()

        return has_academic_scale or (has_df_structure and has_annotation_file)

    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < 30 and cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        if not frames:
            return {"verdict": "ERROR", "prob_tampered": 0.5, "pignistic_fake": 0.5}

        t_ev = self.temporal_agent.analyze(frames)
        g_ev = self.eye_agent.analyze(frames[len(frames) // 2])
        faces_found = g_ev.telemetry.get("faces_detected", 0) > 0
        r_ev = self.rppg_agent.analyze(frames, has_face=faces_found)

        fused = self.fusion.fuse_graph_evidence(
            {"temporal": t_ev.bba_mass, "gaze": g_ev.bba_mass, "rppg": r_ev.bba_mass},
            [("gaze", "rppg"), ("temporal", "gaze")],
        )

        return {
            "verdict": fused["verdict"],
            "pignistic_fake": fused["pignistic_fake"],
            "belief": fused["belief"],
            "plausibility": fused["plausibility"],
        }

    def run_benchmark(
        self,
        dataset_path: str,
        export_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        p = Path(dataset_path)
        is_real_dataset = self.is_real_df_platter_present(dataset_path)

        print("=" * 80)
        if not is_real_dataset:
            print("⚠️  DATASET PRESENCE AUDIT NOTICE:")
            print("   DF-Platter not found - results below are from local samples only and are NOT DF-Platter results.")
            print("   To obtain and evaluate official DF-Platter (133k videos), see: docs/DF_PLATTER_ACCESS.md")
        else:
            print("✅ OFFICIAL DF-PLATTER DATASET DETECTED (IIT Jodhpur / CVPR 2023)")
        print("=" * 80)

        if not p.exists():
            print(f"[!] Target directory does not exist: {dataset_path}")
            return {"total_evaluated": 0, "is_real_df_platter": False}

        videos = list(p.rglob("*.mp4")) + list(p.rglob("*.avi"))
        if not videos:
            print(f"[!] No video files found in {dataset_path}")
            return {"total_evaluated": 0, "is_real_df_platter": False}

        results = []
        slice_data = {
            "technique": {},
            "compression": {},
            "fitzpatrick": {},
            "headwear": {},
            "facial_hair": {},
        }

        y_true_all = []
        y_score_all = []

        print(f"Processing {len(videos)} media files in: {dataset_path}...")
        for idx, v in enumerate(videos, 1):
            attrs = self.metadata_loader.get_metadata(v)
            res = self.evaluate_video(str(v))
            gt = 1 if attrs["is_deepfake"] else 0
            score = res.get("pignistic_fake", 0.5)

            y_true_all.append(gt)
            y_score_all.append(score)

            # Record per slice
            for category in ["technique", "compression", "fitzpatrick", "headwear", "facial_hair"]:
                val = attrs[category]
                if val not in slice_data[category]:
                    slice_data[category][val] = {"y_true": [], "y_score": []}
                slice_data[category][val]["y_true"].append(gt)
                slice_data[category][val]["y_score"].append(score)

            results.append({
                "filename": v.name,
                "attributes": attrs,
                "verdict": res["verdict"],
                "pignistic_fake": score,
                "is_ground_truth_fake": bool(gt),
            })

        y_true_arr = np.array(y_true_all)
        y_score_arr = np.array(y_score_all)

        overall_auc, overall_eer, overall_tpr1 = compute_roc_metrics(y_true_arr, y_score_arr)
        overall_ci_tuple, overall_ci_str = bootstrap_ci(y_true_arr, y_score_arr)
        overall_acc = float(np.mean((y_score_arr > 0.50) == y_true_arr)) if len(y_true_arr) > 0 else 0.0

        # Build Slice Evaluation Tables
        slice_report = {}
        fairness_gaps = {}

        report_title = "OFFICIAL DF-PLATTER BENCHMARK" if is_real_dataset else "LOCAL SAMPLE BENCHMARK (NOT DF-PLATTER)"
        print("\n" + "=" * 90)
        print(f"📊 {report_title} - MULTI-AGENT FAIRNESS & DEMOGRAPHIC REPORT")
        print("=" * 90)
        print(f"Total Videos: {len(videos)} | Overall Acc: {overall_acc:.1%} | AUC: {overall_auc:.3f} (95% CI: {overall_ci_str})")
        print("=" * 90)

        for category, cat_data in slice_data.items():
            print(f"\n--- Slicing Category: {category.upper()} ---")
            print(f"{'Slice Value':24s} | {'N':5s} | {'Acc':7s} | {'TPR':7s} | {'FPR':7s} | {'AUC (95% CI)':20s} | {'Reliability'}")
            print("-" * 90)

            slice_aucs = []
            slice_report[category] = {}

            for val, d in cat_data.items():
                yt = np.array(d["y_true"])
                ys = np.array(d["y_score"])
                n_samples = len(yt)

                acc = float(np.mean((ys > 0.50) == yt)) if n_samples > 0 else 0.0
                n_pos = int(np.sum(yt == 1))
                n_neg = int(np.sum(yt == 0))

                tpr = float(np.sum((ys > 0.50) & (yt == 1)) / (n_pos + 1e-6)) if n_pos > 0 else 0.0
                fpr = float(np.sum((ys > 0.50) & (yt == 0)) / (n_neg + 1e-6)) if n_neg > 0 else 0.0

                auc_val, _, _ = compute_roc_metrics(yt, ys)
                ci_tuple_val, ci_str_val = bootstrap_ci(yt, ys)

                # Reliability label per user specification
                reliability = "✅ Valid" if n_samples >= 30 else "⚠️ Unreliable (N < 30)"

                if n_samples >= 2 and len(np.unique(yt)) >= 2:
                    slice_aucs.append(auc_val)

                slice_report[category][val] = {
                    "count": n_samples,
                    "accuracy": round(acc, 4),
                    "tpr": round(tpr, 4),
                    "fpr": round(fpr, 4),
                    "auc": auc_val,
                    "ci_95": ci_str_val,
                    "reliable": n_samples >= 30,
                }

                auc_str = f"{auc_val:.3f} ({ci_str_val})"
                print(f"{val:24s} | {n_samples:5d} | {acc:6.1%} | {tpr:6.1%} | {fpr:6.1%} | {auc_str:20s} | {reliability}")

            # Demographic fairness gap (max AUC - min AUC across slices in category)
            if len(slice_aucs) >= 2:
                gap = max(slice_aucs) - min(slice_aucs)
                fairness_gaps[category] = round(gap, 4)
                print(f"  --> Maximum Fairness Disparity Gap in {category}: {gap:.3f}")

        summary = {
            "is_real_df_platter": is_real_dataset,
            "disclaimer": "Local samples only - NOT DF-Platter" if not is_real_dataset else "Official DF-Platter",
            "total_evaluated": len(videos),
            "overall_accuracy": round(overall_acc, 4),
            "overall_auc": overall_auc,
            "overall_eer": overall_eer,
            "overall_ci_95": overall_ci_str,
            "fairness_gaps": fairness_gaps,
            "slice_statistics": slice_report,
            "videos": results,
        }

        if export_json:
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"\n[+] Detailed benchmark report exported to: {export_json}")

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM DF-Platter Benchmark Evaluator")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory of DF-Platter dataset")
    parser.add_argument("--sample_dir", type=str, default="data/real_test_samples", help="Local directory for test run")
    parser.add_argument("--metadata", type=str, default=None, help="Path to metadata CSV/JSON file")
    parser.add_argument("--export_json", type=str, default=None, help="Save evaluation summary to JSON")

    args = parser.parse_args()
    target_dir = args.dataset_dir if args.dataset_dir else args.sample_dir

    bench = DFPlatterBenchmark(annotation_file=args.metadata)
    bench.run_benchmark(target_dir, export_json=args.export_json)
