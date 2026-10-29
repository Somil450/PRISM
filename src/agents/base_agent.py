"""
src/agents/base_agent.py
Base contract and data structure for all PRISM forensic agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class AgentEvidence:
    """
    Standard evidence packet output by every PRISM forensic agent.
    """
    agent_name: str
    prob_tampered: float               # Raw tampering probability in [0.0, 1.0]
    reliability: float                 # Quality discount factor in [0.0, 1.0]
    bba_mass: Dict[str, float]         # Basic Belief Assignment {'T': mass_T, 'A': mass_A, 'U': mass_U}
    verdict: str                       # TAMPERED, AUTHENTIC, or INCONCLUSIVE
    findings: list[str]                # Human-readable bullet points
    telemetry: Dict[str, Any]          # Raw quantitative telemetry metrics
    visual_artifact: Optional[np.ndarray] = None  # Heatmap, spectrogram, or overlay image array


class BaseForensicAgent(ABC):
    """
    Abstract base class for all specialized forensic agents in PRISM.
    """

    def __init__(
        self,
        name: str,
        base_reliability: float = 0.85,
        is_positive_authenticator: bool = False,
        authenticity_factor: float = 0.15,
        evidence_scale: float = 10.0,
    ) -> None:
        self.name = name
        self.base_reliability = base_reliability
        self.is_positive_authenticator = is_positive_authenticator
        self.authenticity_factor = authenticity_factor
        self.evidence_scale = evidence_scale

    def create_bba(
        self,
        prob_tampered: float,
        custom_reliability: Optional[float] = None,
        is_positive_authenticator: Optional[bool] = None,
        authenticity_factor: Optional[float] = None,
        evidence_scale: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculates the Basic Belief Assignment (BBA) mass function over {T, A, U}.

        Asymmetric Forensic Evidential Mapping (Subjective Logic / Evidential Deep Learning):
        - For anomaly detectors (compression, temporal jitter, metadata, illumination, gaze):
          An observed defect (prob > 0.5) is positive evidence for Tampering (m_T).
          Crucially, absence of an anomaly (prob <= 0.5) is NOT positive proof of authenticity
          ("absence of evidence is not evidence of absence", NIST SP 800-86).
          It represents epistemic uncertainty (m_U). A configurable authenticity factor
          (default 15%) is routed to m_A, preventing the multiplicative suppression of true positive
          tampering evidence when un-triggered agents are fused.
        - For positive authenticators (e.g. verified living pulse, known camera PRNU match):
          Normal physiological/hardware presence directly certifies authenticity (m_A).

        Args:
            prob_tampered: Probability of tampering in [0, 1].
            custom_reliability: Optional override for agent reliability in [0, 1].
            is_positive_authenticator: Whether this agent serves as a positive validator of authenticity.
            authenticity_factor: Fraction of non-tamper evidence mapped to m(A) for anomaly detectors.
            evidence_scale: Dirichlet pseudo-observation multiplier (default 10.0 against prior W=2.0).
        """
        rel = custom_reliability if custom_reliability is not None else self.base_reliability
        prob = max(0.0, min(1.0, float(prob_tampered)))
        rel = max(0.0, min(1.0, float(rel)))

        pos_auth = (
            is_positive_authenticator
            if is_positive_authenticator is not None
            else getattr(self, "is_positive_authenticator", False)
        )
        auth_factor = (
            authenticity_factor
            if authenticity_factor is not None
            else getattr(self, "authenticity_factor", 0.15)
        )
        auth_factor = max(0.0, min(1.0, float(auth_factor)))

        ev_scale = (
            evidence_scale
            if evidence_scale is not None
            else getattr(self, "evidence_scale", 10.0)
        )
        ev_scale = max(0.1, float(ev_scale))

        if pos_auth:
            e_T = prob * ev_scale * rel
            e_A = (1.0 - prob) * ev_scale * rel
            W = 2.0
        else:
            e_T = prob * ev_scale * rel
            e_A = (1.0 - prob) * ev_scale * rel * auth_factor
            W = 2.0 + (1.0 - prob) * ev_scale * rel * (1.0 - auth_factor)

        S = e_T + e_A + W
        m_T = e_T / S
        m_A = e_A / S
        m_U = W / S

        return {"T": m_T, "A": m_A, "U": m_U}

    FORBIDDEN_KWARGS = frozenset({"path", "filename", "label", "generator", "identity"})

    def _validate_no_leakage_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Enforces zero label/generator/identity/path leakage:
        Rejects any attempt to pass path, filename, label, generator, or identity to analyze().
        """
        for forbidden in self.FORBIDDEN_KWARGS:
            if forbidden in kwargs:
                raise ValueError(
                    f"{self.__class__.__name__}.analyze() strictly prohibits argument '{forbidden}' "
                    f"to prevent label, identity, and filename metadata leakage."
                )

    @abstractmethod
    def analyze(self, **kwargs: Any) -> AgentEvidence:
        """
        Execute forensic analysis on the in-memory input payload and return an AgentEvidence packet.
        Must NOT accept path, filename, label, generator, or identity arguments.
        """
        pass
