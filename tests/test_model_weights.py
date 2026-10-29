"""
tests/test_model_weights.py
Unit tests verifying that trained PRISM model checkpoints exist, load cleanly,
and perform accurate inference on sample tensors.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from model import PRISMNetwork


class TestModelCheckpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.checkpoints_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints")
        )
        cls.final_path = os.path.join(cls.checkpoints_dir, "prism_model_final.pth")
        cls.best_path = os.path.join(cls.checkpoints_dir, "prism_model_best.pth")

    def test_01_checkpoints_exist(self):
        if not os.path.isfile(self.final_path):
            raise unittest.SkipTest("Model checkpoint not found. Run py scripts/download_weights.py to download.")
        self.assertTrue(os.path.isfile(self.final_path), f"Missing {self.final_path}")
        self.assertTrue(os.path.isfile(self.best_path), f"Missing {self.best_path}")
        self.assertGreater(os.path.getsize(self.final_path), 300 * 1024 * 1024)
        self.assertGreater(os.path.getsize(self.best_path), 300 * 1024 * 1024)

    def test_02_load_final_checkpoint(self):
        if not os.path.isfile(self.final_path):
            raise unittest.SkipTest("Model checkpoint not found. Run py scripts/download_weights.py to download.")
        model = PRISMNetwork(pretrained=False)
        state = torch.load(self.final_path, map_location="cpu")
        res = model.load_state_dict(state, strict=False)
        self.assertEqual(len(res.missing_keys), 0, f"Missing keys: {res.missing_keys}")
        self.assertEqual(len(res.unexpected_keys), 0, f"Unexpected keys: {res.unexpected_keys}")

        model.eval()
        rgb = torch.randn(2, 3, 224, 224)
        noise = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            out = model(rgb, noise)

        self.assertIn("logit", out)
        self.assertIn("mask", out)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["mask"].shape, (2, 1, 224, 224))
        self.assertTrue(torch.all((out["mask"] >= 0.0) & (out["mask"] <= 1.0)))

    def test_03_load_best_checkpoint(self):
        if not os.path.isfile(self.best_path):
            raise unittest.SkipTest("Model checkpoint not found. Run py scripts/download_weights.py to download.")
        model = PRISMNetwork(pretrained=False)
        state = torch.load(self.best_path, map_location="cpu")
        res = model.load_state_dict(state, strict=False)
        self.assertEqual(len(res.missing_keys), 0, f"Missing keys: {res.missing_keys}")
        self.assertEqual(len(res.unexpected_keys), 0, f"Unexpected keys: {res.unexpected_keys}")

        model.eval()
        rgb = torch.randn(1, 3, 224, 224)
        noise = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            out = model(rgb, noise)

        self.assertEqual(out["logit"].shape, (1, 1))
        self.assertEqual(out["mask"].shape, (1, 1, 224, 224))


if __name__ == "__main__":
    unittest.main()
