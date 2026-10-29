"""
tests/test_forensic_vision_somil.py
Member 2: Somil - Physical, Temporal & Container Forensic Agents Tests
"""

import sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents import MetadataAgent, CompressionWatermarkAgent, VideoTemporalAgent, IlluminationAgent


def test_metadata_agent_sanitized_container():
    agent = MetadataAgent()
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    ev = agent.analyze(img, file_bytes=b"RIFF....WEBP")
    assert ev.bba_mass["U"] == 1.0, "Sanitized container must map to epistemic uncertainty m(U) = 1.0"
    assert ev.verdict == "INCONCLUSIVE"


def test_compression_ela_agent():
    agent = CompressionWatermarkAgent()
    img = Image.new("RGB", (128, 128), color=(200, 150, 100))
    ev = agent.analyze(img)
    assert "T" in ev.bba_mass and "A" in ev.bba_mass and "U" in ev.bba_mass
    assert abs(sum(ev.bba_mass.values()) - 1.0) < 1e-4


def test_video_temporal_static_screen_detection():
    agent = VideoTemporalAgent()
    # 10 identical stationary frames simulating desktop recording
    static_frames = [np.ones((224, 224, 3), dtype=np.uint8) * 100 for _ in range(10)]
    ev = agent.analyze(static_frames)
    assert ev.prob_tampered <= 0.20, f"Stationary screen must not trigger false jitter; got {ev.prob_tampered}"
    assert ev.verdict != "TAMPERED"


def test_illumination_diffuse_lighting():
    agent = IlluminationAgent()
    diffuse_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    ev = agent.analyze(diffuse_img)
    assert ev.verdict == "INCONCLUSIVE", "Diffuse uninformative lighting must yield INCONCLUSIVE"
