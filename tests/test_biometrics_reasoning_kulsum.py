"""
tests/test_biometrics_reasoning_kulsum.py
Member 3: Kulsum - Biometrics, Cross-Modal Forensics & Evidential Reasoning Tests
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agents import RPPGAgent, EyeGazeAgent, LipSyncAgent, AudioVoiceAgent
from fusion import DempsterShaferFusion
from reasoner import PRISMReasoner


def test_dempster_shafer_uncertainty_handling():
    fuser = DempsterShaferFusion()
    # If all agents are uncertain
    m1 = {"T": 0.0, "A": 0.0, "U": 1.0}
    m2 = {"T": 0.0, "A": 0.0, "U": 1.0}
    combined, K = fuser.combine(m1, m2)
    assert combined["U"] == 1.0
    assert K == 0.0


def test_zadeh_paradox_resolution():
    fuser = DempsterShaferFusion()
    # Primary ViT strongly authentic, but two heuristic agents report tamper
    agent_dict = {
        "vit": {"T": 0.012, "A": 0.815, "U": 0.173},
        "temporal": {"T": 0.810, "A": 0.04, "U": 0.15},
        "illum": {"T": 0.775, "A": 0.05, "U": 0.175},
        "comp": {"T": 0.05, "A": 0.85, "U": 0.10},
    }
    edges = [("vit", "temporal"), ("comp", "vit"), ("illum", "temporal")]
    fused = fuser.fuse_graph_evidence(agent_dict, edges)
    # Must NOT falsely convict as TAMPERED
    assert fused["verdict"] != "TAMPERED", "Zadeh's paradox must prevent heuristic false conviction"


def test_deterministic_reasoner_court_dossier():
    reasoner = PRISMReasoner()
    report = reasoner.generate_report(
        score=0.15,
        mask_area="< 0.5%",
        noise_status="Consistent with physical sensor floor",
        system_verdict="AUTHENTIC",
        vit_mass=0.012,
        prnu_mass=0.09,
        ela_mass=0.04,
        meta_mass=0.0,
    )
    assert "### 1. Case Summary" in report
    assert "AUTHENTIC" in report
    assert "### 4. Examiner Conclusion & Limitations" in report
