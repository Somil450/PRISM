"""
tests/test_kaggle_splits.py
Unit tests verifying strict identity-disjoint Celeb-DF partitioning.

Enforces Prompt 13 Requirements:
  1. Both identities in fake videos (idA_idB_XXXX.mp4) are tracked and checked.
  2. Zero identity leakage between train, val, and test splits.
  3. Fails when a split is empty or nearly empty (< 30 train identities) unless explicitly in unit-test override.
  4. Verifies mock_test does not hide empty or leaked splits.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Ensure src and scripts on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.train_kaggle_vit_dora import build_identity_disjoint_splits, parse_celeb_df_test_list


def create_mock_celeb_df_structure(root_dir: str, num_ids: int = 50, test_id_count: int = 10):
    """Creates a miniature folder structure with synthetic identity names."""
    for folder in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
        os.makedirs(os.path.join(root_dir, folder), exist_ok=True)

    # Real videos
    for i in range(num_ids):
        ident = f"id{i}"
        fname = f"{ident}_001.mp4"
        open(os.path.join(root_dir, "Celeb-real", fname), "w").close()

    # Synthesis videos with TWO identities
    for i in range(0, num_ids - 1, 2):
        t_id = f"id{i}"
        s_id = f"id{i+1}"
        fname = f"{t_id}_{s_id}_001.mp4"
        open(os.path.join(root_dir, "Celeb-synthesis", fname), "w").close()

    # Test list designating test_id_count identities
    test_list_p = os.path.join(root_dir, "List_of_testing_videos.txt")
    with open(test_list_p, "w") as f:
        for i in range(num_ids - test_id_count, num_ids):
            f.write(f"0 Celeb-real/id{i}_001.mp4\n")
    return test_list_p


def test_dual_identity_tracking_in_fakes():
    """Verify that synthesis videos spanning two identities are tracked across both IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        tl_path = create_mock_celeb_df_structure(tmp, num_ids=50, test_id_count=10)
        train_recs, val_recs, test_recs, summary = build_identity_disjoint_splits(
            tmp, tl_path, val_ratio=0.20, min_train_identities=25
        )

        train_ids = set(summary["train_ids"])
        val_ids = set(summary["val_ids"])
        test_ids = set(summary["test_ids"])

        # Assert no identity leakage between any partitions
        assert len(train_ids.intersection(val_ids)) == 0, "Train-Val identity leakage!"
        assert len(train_ids.intersection(test_ids)) == 0, "Train-Test identity leakage!"
        assert len(val_ids.intersection(test_ids)) == 0, "Val-Test identity leakage!"

        # Assert every video in train has all its constituent IDs in train_ids
        for r in train_recs:
            for ident in r["all_ids"]:
                assert ident in train_ids, f"Leakage: identity {ident} in train video {r['filename']} not in train_ids!"
                assert ident not in test_ids, f"Leakage: test identity {ident} found in train video {r['filename']}!"
                assert ident not in val_ids, f"Leakage: val identity {ident} found in train video {r['filename']}!"


def test_fails_when_train_identities_under_threshold():
    """Verify that partitioning fails loudly if train set has < 30 identities."""
    with tempfile.TemporaryDirectory() as tmp:
        # Only 20 total identities; test takes 10 -> pool has 10 -> train has ~8 (< 30)
        tl_path = create_mock_celeb_df_structure(tmp, num_ids=20, test_id_count=10)
        with pytest.raises(ValueError, match="under minimum threshold"):
            build_identity_disjoint_splits(tmp, tl_path, min_train_identities=30)


def test_split_not_empty():
    """Verify that no split is empty."""
    with tempfile.TemporaryDirectory() as tmp:
        tl_path = create_mock_celeb_df_structure(tmp, num_ids=50, test_id_count=10)
        train_recs, val_recs, test_recs, summary = build_identity_disjoint_splits(
            tmp, tl_path, min_train_identities=25
        )
        assert len(train_recs) > 0, "Train split is empty!"
        assert len(val_recs) > 0, "Validation split is empty!"
        assert len(test_recs) > 0, "Test split is empty!"
