"""
tests/test_df_platter_metadata.py
Unit tests verifying DFPlatterMetadataLoader against tricky filenames and fixture annotations.

Validates:
  1. Does not misclassify substrings (e.g. 'bear_dance' or 'disobedient' must NOT match 'beard').
  2. Does not misclassify 'fast_car' as 'fs' (FaceSwap).
  3. Handles mixed-case paths (e.g. 'FitzPatrick_IV', 'TURBAN', 'DeEpFaCeLaB').
  4. Missing attributes cleanly default to 'unknown'.
  5. Correctly loads and prioritizes CSV / JSON fixture annotations over filename heuristics.
"""

import os
import sys
import unittest
import tempfile
import json
import csv
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from benchmark_df_platter import DFPlatterMetadataLoader, DFPlatterBenchmark


class TestDFPlatterMetadataLoader(unittest.TestCase):

    def setUp(self):
        self.loader = DFPlatterMetadataLoader()

    def test_01_tricky_substrings_not_misclassified(self):
        """Tricky filenames containing substrings of keywords must NOT be misclassified."""
        # 'bear_dance.mp4' contains 'bear' but is NOT 'beard'
        meta = self.loader.get_metadata(Path("bear_dance.mp4"))
        self.assertEqual(meta["facial_hair"], "unknown", f"Expected 'unknown' for 'bear_dance.mp4', got {meta['facial_hair']}")

        # 'disobedient_actor.mp4' contains 'bed' or letters but NOT 'beard'
        meta2 = self.loader.get_metadata(Path("disobedient_actor.mp4"))
        self.assertEqual(meta2["facial_hair"], "unknown")

        # 'fast_car_highway.mp4' contains 'fs' substring in 'fast' -> must NOT match FaceSwap
        meta3 = self.loader.get_metadata(Path("fast_car_highway.mp4"))
        self.assertNotEqual(meta3["technique"], "FaceSwap")

    def test_02_mixed_case_handling(self):
        """Mixed-case and uppercase tags must be normalized correctly."""
        meta = self.loader.get_metadata(Path("C23/DeepFaceLab/FITZPATRICK_IV_TURBAN_BEARD_01.mp4"))
        self.assertEqual(meta["technique"], "DeepFaceLab")
        self.assertEqual(meta["compression"], "c23")
        self.assertEqual(meta["fitzpatrick"], "Fitzpatrick IV")
        self.assertEqual(meta["headwear"], "Turban")
        self.assertEqual(meta["facial_hair"], "Beard / Moustache")
        self.assertTrue(meta["is_deepfake"])

    def test_03_missing_tags_default_to_unknown(self):
        """Unannotated generic video paths must cleanly default to 'unknown'."""
        meta = self.loader.get_metadata(Path("unlabeled_video_clip_99.mp4"))
        self.assertEqual(meta["technique"], "unknown")
        self.assertEqual(meta["compression"], "unknown")
        self.assertEqual(meta["fitzpatrick"], "unknown")
        self.assertEqual(meta["headwear"], "unknown")
        self.assertEqual(meta["facial_hair"], "unknown")

    def test_04_csv_fixture_annotation_loading(self):
        """Official CSV metadata fixtures must be cleanly loaded and override heuristics."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            csv_path = f.name
            writer = csv.DictWriter(f, fieldnames=["video_id", "technique", "compression", "fitzpatrick", "headwear", "facial_hair", "is_deepfake"])
            writer.writeheader()
            writer.writerow({
                "video_id": "test_video_001.mp4",
                "technique": "LatentDiffusion",
                "compression": "c40",
                "fitzpatrick": "Fitzpatrick V",
                "headwear": "Dupatta",
                "facial_hair": "Clean-shaven",
                "is_deepfake": "1",
            })

        try:
            fixture_loader = DFPlatterMetadataLoader(annotation_file=csv_path)
            meta = fixture_loader.get_metadata(Path("path/to/test_video_001.mp4"))
            self.assertEqual(meta["technique"], "LatentDiffusion")
            self.assertEqual(meta["compression"], "c40")
            self.assertEqual(meta["fitzpatrick"], "Fitzpatrick V")
            self.assertEqual(meta["headwear"], "Dupatta")
            self.assertEqual(meta["facial_hair"], "Clean-shaven")
            self.assertTrue(meta["is_deepfake"])
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_05_real_dataset_presence_detection(self):
        """Evaluator must correctly report presence vs absence of official DF-Platter dataset."""
        bench = DFPlatterBenchmark()
        # Non-existent or small directory must report False
        self.assertFalse(bench.is_real_df_platter_present("data/real_test_samples"))
        self.assertFalse(bench.is_real_df_platter_present("non_existent_dir_123"))


if __name__ == "__main__":
    unittest.main()
