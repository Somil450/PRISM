"""
tests/test_leakage_static_check.py
Static checks and unit tests verifying zero label/generator/identity/filename leakage.

Enforces Prompt Requirements:
  1. No agent's analyze() accepts a path, filename, label, generator, or identity argument.
  2. Passing any forbidden argument or string path to analyze() raises ValueError.
  3. Forensic fusion models never read path, filename, label, generator, or identity CSV columns as features.
"""

import inspect
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents import (
    MetadataAgent,
    EyeGazeAgent,
    CompressionWatermarkAgent,
    AudioVoiceAgent,
    VideoTemporalAgent,
    IlluminationAgent,
    LipSyncAgent,
    RPPGAgent,
)
from agents.base_agent import BaseForensicAgent

ALL_AGENT_CLASSES = [
    MetadataAgent,
    EyeGazeAgent,
    CompressionWatermarkAgent,
    AudioVoiceAgent,
    VideoTemporalAgent,
    IlluminationAgent,
    LipSyncAgent,
    RPPGAgent,
]

FORBIDDEN_NAMES = {"path", "filename", "label", "generator", "identity"}


def test_agent_analyze_signatures_no_forbidden_parameters():
    """Verify that no agent's analyze() method declares path, filename, label, generator, or identity."""
    for cls in ALL_AGENT_CLASSES:
        sig = inspect.signature(cls.analyze)
        param_names = set(sig.parameters.keys())
        leakage_params = param_names.intersection(FORBIDDEN_NAMES)
        assert len(leakage_params) == 0, (
            f"Agent class {cls.__name__}.analyze() declares forbidden parameters: {leakage_params}"
        )


def test_agent_analyze_rejects_forbidden_kwargs():
    """Verify that passing any forbidden kwarg to any agent raises ValueError."""
    dummy_img = Image.new("RGB", (64, 64))
    dummy_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(8)]
    dummy_audio = (np.zeros(1600, dtype=np.float32), 16000)
    dummy_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    agents = [
        (MetadataAgent(), dummy_bytes),
        (EyeGazeAgent(), dummy_img),
        (CompressionWatermarkAgent(), dummy_img),
        (AudioVoiceAgent(), dummy_audio),
        (VideoTemporalAgent(), dummy_frames),
        (IlluminationAgent(), dummy_img),
        (LipSyncAgent(), (dummy_frames, dummy_audio)),
        (RPPGAgent(), dummy_frames),
    ]

    for agent, valid_input in agents:
        for forbidden in FORBIDDEN_NAMES:
            kwargs = {forbidden: "leakage_test_value"}
            with pytest.raises(ValueError, match="prohibits argument"):
                agent.analyze(valid_input, **kwargs)


def test_agent_analyze_rejects_string_paths():
    """Verify that media agents reject string paths to prevent implicit filename/path leakage."""
    string_rejecting_agents = [
        MetadataAgent(),
        CompressionWatermarkAgent(),
        EyeGazeAgent(),
        IlluminationAgent(),
        VideoTemporalAgent(),
        AudioVoiceAgent(),
        LipSyncAgent(),
        RPPGAgent(),
    ]

    fake_path = "data/normalized_test_samples/01_deepfake_face_swap.mp4"
    for agent in string_rejecting_agents:
        with pytest.raises(ValueError, match="does not accept"):
            agent.analyze(fake_path)


def test_fusion_never_reads_forbidden_columns():
    """Verify that fusion scripts strictly exclude path, filename, label, generator, identity, and id from features."""
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    from eval_fusion_cached import AGENT_COLUMNS

    forbidden_cols = {"path", "filename", "label", "generator", "identity", "id"}
    overlap = set(AGENT_COLUMNS).intersection(forbidden_cols)
    assert len(overlap) == 0, f"AGENT_COLUMNS contains forbidden metadata columns: {overlap}"

    # Verify all agent columns are strictly forensic score columns
    for col in AGENT_COLUMNS:
        assert col.endswith("_score"), f"Fusion column '{col}' is not an agent score column!"
