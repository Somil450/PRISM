"""
tests/test_neural_backbone_tanush.py
Member 1: Tanush - Core Neural Backbone, Preprocessing & Dynamic Staging Tests
"""

import sys, os
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessor import PRISMPreprocessor
from dora_layer import DoRALinear
from model import SpatialMaskDecoder, SensorNoiseEncoder


def test_dora_layer_forward():
    dora = DoRALinear(in_features=64, out_features=64, rank=8, lora_alpha=16.0)
    x = torch.randn(2, 64)
    out = dora(x)
    assert out.shape == (2, 64), f"Expected shape (2, 64), got {out.shape}"


def test_spatial_decoder_upsampling():
    decoder = SpatialMaskDecoder()
    # 196 tokens corresponding to 14x14 patch grid
    tokens = torch.randn(2, 196, 768)
    mask = decoder(tokens)
    assert mask.shape == (2, 1, 224, 224), f"Expected mask shape (2, 1, 224, 224), got {mask.shape}"
    assert torch.all(mask >= 0.0) and torch.all(mask <= 1.0), "Mask outputs must be bounded in [0.0, 1.0]"


def test_preprocessor_prnu_extraction():
    prep = PRISMPreprocessor()
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rgb_t, noise_t = prep.process_frame(test_img)
    assert rgb_t.shape == (3, 224, 224)
    assert noise_t.shape == (1, 224, 224)
