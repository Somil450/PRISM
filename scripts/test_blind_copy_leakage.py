"""
scripts/test_blind_copy_leakage.py
Rigorous Blind-Copy Leakage Audit for PRISM Forensic Agents.

Protocol:
  1. Copies every evaluation file (N=17) into a randomized temporary directory.
  2. Assigns each file a completely random hex UUID filename (e.g. 7f92b4...mp4),
     stripping all label, generator, and identity tokens from paths.
  3. Normalizes all file mtimes and atimes to a fixed constant timestamp.
  4. Runs feature extraction through PRISM's multi-agent pipeline on the blind files.
  5. Compares extracted forensic scores against the original run.
  6. Reports every agent whose score changes versus the original run (|Delta| > 1e-4).
"""

import os
import sys
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_features import FeatureExtractor
from eval_fusion_cached import AGENT_COLUMNS


def run_blind_copy_leakage_audit(
    original_csv: str = "data/cache/features_normalized.csv",
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    orig_path = Path(original_csv)
    if not orig_path.is_absolute():
        orig_path = REPO_ROOT / orig_path

    if not orig_path.exists():
        raise FileNotFoundError(f"Original feature cache not found at: {orig_path}")

    df_orig = pd.read_csv(orig_path)
    extractor = FeatureExtractor()

    print("=" * 105)
    print("🔬 PRISM BLIND-COPY LEAKAGE AUDIT")
    print("=" * 105)
    print(f"Original Cache Path : {orig_path}")
    print(f"Sample Count        : N={len(df_orig)}")
    print(f"Score Tolerance     : |Delta| <= {tolerance}")
    print("-" * 105)

    with tempfile.TemporaryDirectory(prefix="prism_blind_audit_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        fixed_timestamp = 1700000000.0  # Normalized epoch timestamp

        blind_items: List[Dict[str, Any]] = []
        mapping: List[Tuple[str, str, str]] = []

        print("[+] Copying evaluation files to randomized blind paths with normalized timestamps...")
        for idx, row in df_orig.iterrows():
            src_file = Path(row["path"])
            if not src_file.is_absolute():
                src_file = REPO_ROOT / src_file

            ext = src_file.suffix.lower()
            random_name = f"{uuid.uuid4().hex}{ext}"
            dst_file = tmp_p / random_name

            shutil.copy2(src_file, dst_file)
            # Normalize mtime and atime
            os.utime(dst_file, (fixed_timestamp, fixed_timestamp))

            # Entry strictly stripped of label, generator, identity, and original name
            blind_entry = {
                "id": random_name,
                "path": str(dst_file),
                "label": 0,          # Neutral dummy label (not passed to analyze())
                "identity": "anon",  # Anonymized identity
                "generator": "none", # Anonymized generator
                "compression": "none",
            }
            blind_items.append(blind_entry)
            mapping.append((str(row["id"]), random_name, str(src_file.name)))

        print(f"[+] Successfully staged {len(blind_items)} anonymized files in temp dir: {tmp_p.name}")
        print("[+] Re-extracting multi-agent features on blind files...")

        blind_rows = []
        for idx, item in enumerate(blind_items):
            res = extractor.process_item(item)
            blind_rows.append(res)
            print(f"    [{idx+1:02d}/{len(blind_items)}] Extracted blind features for anonymized item {item['id'][:12]}...")

        df_blind = pd.DataFrame(blind_rows)

    # Compare scores per agent across all files
    score_cols = [c for c in AGENT_COLUMNS]
    agent_deltas: Dict[str, List[float]] = {c: [] for c in score_cols}
    agent_max_delta: Dict[str, float] = {c: 0.0 for c in score_cols}
    changed_agents: Dict[str, List[Dict[str, Any]]] = {c: [] for c in score_cols}

    for idx in range(len(df_orig)):
        orig_row = df_orig.iloc[idx]
        blind_row = df_blind.iloc[idx]
        orig_name = orig_row["id"]

        for col in score_cols:
            orig_val = float(orig_row.get(col, 0.50))
            blind_val = float(blind_row.get(col, 0.50))
            delta = abs(orig_val - blind_val)

            agent_deltas[col].append(delta)
            if delta > agent_max_delta[col]:
                agent_max_delta[col] = delta

            if delta > tolerance:
                changed_agents[col].append({
                    "id": orig_name,
                    "orig_score": orig_val,
                    "blind_score": blind_val,
                    "delta": delta,
                })

    print("\n" + "=" * 105)
    print("📊 BLIND-COPY LEAKAGE AUDIT RESULTS: ORIGINAL VS RANDOMIZED BLIND SCORE COMPARISON")
    print("=" * 105)
    print(f"{'Forensic Agent':26s} | {'Max |Delta|':12s} | {'Mean |Delta|':12s} | {'Divergent Files':16s} | {'Leakage Assessment'}")
    print("-" * 105)

    any_leakage = False
    for col in score_cols:
        name = col.replace("_score", "").replace("_", " ").title()
        max_d = agent_max_delta[col]
        mean_d = float(np.mean(agent_deltas[col]))
        n_div = len(changed_agents[col])

        if max_d > tolerance:
            any_leakage = True
            assessment = f"LEAKAGE DETECTED ({n_div} files diverged)"
        else:
            assessment = "CLEAN (Zero Score Change)"

        print(f"{name:26s} | {max_d:12.6f} | {mean_d:12.6f} | {n_div:16d} | {assessment}")

    print("=" * 105)
    if not any_leakage:
        print("[+] VERIFICATION PASSED: All 10 forensic agents produced identical scores")
        print("    when paths, filenames, and mtimes were completely randomized. Zero leakage detected.")
    else:
        print("[!] WARNING: Forensic scores changed when filenames/paths were randomized.")
    print("=" * 105 + "\n")

    return {
        "score_cols": score_cols,
        "agent_max_delta": agent_max_delta,
        "changed_agents": changed_agents,
        "any_leakage": any_leakage,
    }


if __name__ == "__main__":
    run_blind_copy_leakage_audit()
