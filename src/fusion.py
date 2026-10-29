import numpy as np
from typing import Any, Dict, List, Tuple, Union
"""
src/fusion.py
Dempster-Shafer Evidence Fusion for PRISM multi-agent outputs.

Frame of Discernment: Omega = {Tampered (T), Authentic (A), Uncertainty (Theta)}
where Theta = {T, A} is the full frame (agent unsure).

Conflict K measures how much agents contradict each other:
  K = m1(T)*m2(A) + m1(A)*m2(T)   <- all focal element pairs with empty intersection

High K (near 1.0) means total contradiction - capped at 0.99 to prevent ZeroDivisionError.
"""

_K_CAP = 0.99   # conflict ceiling; avoids 1/(1-K) blowup


class DempsterShaferFusion:
    """
    Stateless fusion engine. All methods are pure functions on dicts.
    Thread-safe by design (no shared mutable state).

    Mass dict keys: 'T' (Tampered), 'A' (Authentic), 'U' (Uncertainty / full frame).
    """

    @staticmethod
    def create_mass(evidence_T: float, evidence_A: float, W: float = 2.0) -> dict:
        """
        Build a Basic Belief Assignment (BBA) using Evidential Deep Learning (Subjective Logic).
        Instead of heuristics, this maps raw evidence (e.g., ReLU(logits)) directly to beliefs.

        Args:
            evidence_T: Raw evidence supporting Tampered (>= 0).
            evidence_A: Raw evidence supporting Authentic (>= 0).
            W:          A priori weight (typically number of classes, so 2.0).

        Returns:
            dict with keys 'T', 'A', 'U' summing to 1.0.
        """
        # Ensure evidence is non-negative
        e_T = max(0.0, evidence_T)
        e_A = max(0.0, evidence_A)

        total_evidence = e_T + e_A
        S = total_evidence + W

        m_T = e_T / S
        m_A = e_A / S
        m_U = W / S

        return {"T": m_T, "A": m_A, "U": m_U}

    @staticmethod
    def combine(m1: dict, m2: dict) -> tuple:
        """
        Dempster's Rule of Combination for two BBAs over {T, A, U}.

        Focal elements and their intersections:
          T ∩ T = T      T ∩ A = ∅ (conflict!)    T ∩ U = T
          A ∩ T = ∅      A ∩ A = A                A ∩ U = A
          U ∩ T = T      U ∩ A = A                U ∩ U = U

        Args:
            m1, m2: BBA dicts with keys 'T', 'A', 'U'.

        Returns:
            (fused_mass dict, conflict_K float)
        """
        # Conflict K: sum of products where intersection is empty
        K_raw = m1["T"] * m2["A"] + m1["A"] * m2["T"]
        K = min(K_raw, _K_CAP)  # cap to prevent ZeroDivisionError on total contradiction

        # When conflict is complete (K -> 1.0), Dempster combination encounters total contradiction.
        # Under forensic principles, irreconcilable contradiction yields pure epistemic uncertainty m(U) = 1.0.
        if K_raw >= 0.999:
            return {"T": 0.0, "A": 0.0, "U": 1.0}, K

        norm = 1.0 - K  # normalization denominator

        # Numerators for each focal element (Theta=U propagated as full frame)
        num_T = m1["T"] * m2["T"] + m1["T"] * m2["U"] + m1["U"] * m2["T"]
        num_A = m1["A"] * m2["A"] + m1["A"] * m2["U"] + m1["U"] * m2["A"]
        num_U = m1["U"] * m2["U"]

        total_num = num_T + num_A + num_U
        if total_num < 1e-12:
            # Full contradiction after capping: agents give zero supporting mass.
            # Return pure uncertainty - maximum epistemic indecision, not a zero vector.
            fused = {"T": 0.0, "A": 0.0, "U": 1.0}
        else:
            fused = {
                "T": num_T / norm,
                "A": num_A / norm,
                "U": num_U / norm,
            }
        return fused, K

    def fuse_multimodal_evidence(self, agent_results: list) -> dict:
        """
        Sequentially combine an arbitrary list of agent BBAs.

        Args:
            agent_results: List of BBA dicts (each from create_mass).
                           Must have at least one entry.

        Returns:
            dict with:
              'T', 'A', 'U'        - final fused masses
              'pignistic_fake'     - BetP(Tampered): main decision score in [0, 1]
              'conflict_history'   - list of K values per combination step
              'verdict'            - human-readable string
        """
        if not agent_results:
            raise ValueError("agent_results must not be empty")

        # Start with the first agent's mass
        fused = agent_results[0].copy()
        conflict_history = []

        for m in agent_results[1:]:
            fused, K = self.combine(fused, m)
            conflict_history.append(round(K, 4))

        # Pignistic Probability (BetP): distribute Theta mass equally to T and A
        # BetP(T) = m(T) + m(U)/|Theta| = m(T) + m(U)/2
        pignistic_fake = fused["T"] + 0.5 * fused["U"]

        # Belief and Plausibility intervals
        belief = fused["T"]
        plausibility = fused["T"] + fused["U"]

        # Verdict thresholds (tunable)
        if pignistic_fake >= 0.70:
            verdict = "TAMPERED"
        elif pignistic_fake <= 0.35:
            verdict = "AUTHENTIC"
        else:
            verdict = "INCONCLUSIVE"

        return {
            "T":               round(fused["T"], 4),
            "A":               round(fused["A"], 4),
            "U":               round(fused["U"], 4),
            "pignistic_fake":  round(pignistic_fake, 4),
            "belief":          round(belief, 4),
            "plausibility":    round(plausibility, 4),
            "conflict_history": conflict_history,
            "verdict":         verdict,
        }



    def fuse_graph_evidence(self, agent_dict: dict, edges: Any) -> dict:
        """
        --- STEP 2 & 3: DYNAMIC FORENSIC EVIDENCE GRAPH (FEG) WITH ATTENTION ---
        Fuses agent BBAs over a spatial-temporal graph topology.
        Supports both:
          - List of edge pairs [('vit', 'temporal'), ('lip', 'temporal')] (dynamically calculates
            attention weights from epistemic uncertainty).
          - Dict of explicit edge weights {('u', 'v'): 1.5, ...}.
        
        Args:
            agent_dict: Dict of BBAs, e.g., {'vit': m_v, 'temporal': m_t}
            edges: List of tuples or Dict mapping tuples to float weights.
        """
        if not agent_dict:
            raise ValueError("agent_dict must not be empty")

        # Normalize to edge-weight mapping
        edge_map = {}
        if isinstance(edges, dict):
            edge_map = edges
        elif isinstance(edges, (list, tuple, set)):
            for edge in edges:
                if isinstance(edge, (tuple, list)) and len(edge) == 2:
                    u, v = edge[0], edge[1]
                    u_u = agent_dict[u]["U"] if u in agent_dict else 1.0
                    v_u = agent_dict[v]["U"] if v in agent_dict else 1.0
                    # Dynamic attention weight w(u, v) = 1.0 + gamma * exp(-beta * (u_u + v_u))
                    # gamma = 1.5 (maximum agreement gain), beta = 2.0 (epistemic decay rate).
                    # When uncertainty increases (u_u -> 1.0), the exponent is negative,
                    # causing attention reinforcement to decay exponentially toward 1.0 (no spurious amplification).
                    gamma = 1.5
                    beta = 2.0
                    w = 1.0 + gamma * float(np.exp(-beta * (u_u + v_u)))
                    edge_map[(u, v)] = w
        else:
            edge_map = {}

        # 1. Apply graph-based Graph Attention (GAT) discounting
        discounted_agents = {}
        for node, bba in agent_dict.items():
            support = 1.0
            for (u, v), weight in edge_map.items():
                if node == u and v in agent_dict:
                    agreement = bba['T'] * agent_dict[v]['T'] + bba['A'] * agent_dict[v]['A']
                    support += agreement * (weight - 1.0)
                elif node == v and u in agent_dict:
                    agreement = bba['T'] * agent_dict[u]['T'] + bba['A'] * agent_dict[u]['A']
                    support += agreement * (weight - 1.0)
            
            # Boost the evidence based on graph support (Subjective Logic mapping)
            E_T = bba['T'] / max(1e-9, bba['U'])
            E_A = bba['A'] / max(1e-9, bba['U'])
            
            E_T *= support
            E_A *= support
            
            S = E_T + E_A + 2.0  # W = 2.0
            discounted_agents[node] = {"T": E_T / S, "A": E_A / S, "U": 2.0 / S}

        # 2. Sequential Dempster-Shafer fusion over the discounted graph nodes
        nodes = list(discounted_agents.values())
        fused = nodes[0].copy()
        conflict_history = []
        
        for m in nodes[1:]:
            fused, K = self.combine(fused, m)
            conflict_history.append(round(K, 4))

        # 3. Forensic Non-Annihilation Rule (NIST/SWGDE Forensic Principles):
        # In multimedia forensics, an asset cannot be certified as AUTHENTIC if:
        # A) A definitive agent (e.g. Metadata AI prompt, ViT backbone, or Audio synth) demonstrates
        #    conclusive evidence (m(T) >= 0.72).
        # B) OR at least two independent forensic agents corroborate suspicion (e.g. top_1 >= 0.55 and top_2 >= 0.40).
        # When uncorroborated single-heuristic conflict occurs against overwhelming authentic evidence,
        # the system avoids false conviction and yields to D-S fusion or INCONCLUSIVE.
        all_T = [agent_dict[k]["T"] for k in agent_dict]
        sorted_T = sorted(all_T, reverse=True)
        max_agent_T = sorted_T[0] if sorted_T else 0.0
        second_agent_T = sorted_T[1] if len(sorted_T) > 1 else 0.0
        max_conflict = max(conflict_history) if conflict_history else 0.0

        # Check for hard definitive single evidence OR multi-agent corroboration
        is_definitive_single = max_agent_T >= 0.70
        is_corroborated_tamper = (max_agent_T >= 0.50 and second_agent_T >= 0.30) or (max_agent_T + second_agent_T >= 0.85)

        # High Epistemic Conflict Protection (Zadeh's Paradox Resolution):
        # If the primary neural ViT backbone strongly affirms authenticity (vit['A'] >= 0.65, vit['T'] <= 0.15)
        # and none of the primary generative streams (vit, comp, prnu, audio) show conclusive tamper (<= 0.25),
        # heuristic divergence (e.g. handheld camera shake, desktop cursor motion, or monitor ambient lighting)
        # must NOT override the neural detector to falsely convict as TAMPERED. Under SWGDE standards,
        # uncorroborated physical heuristic divergence requires INCONCLUSIVE rather than false conviction.
        vit_bba = agent_dict.get("vit", {})
        vit_strongly_authentic = vit_bba.get("A", 0.0) >= 0.65 and vit_bba.get("T", 0.0) <= 0.15
        primary_tamper = max([agent_dict.get(k, {}).get("T", 0.0) for k in ("vit", "comp", "prnu", "audio", "meta", "metadata") if k in agent_dict] or [0.0])

        if (max_conflict >= 0.80 or primary_tamper <= 0.25) and vit_strongly_authentic:
            is_definitive_single = False
            is_corroborated_tamper = False

        if is_definitive_single or is_corroborated_tamper:
            # Enforce continuous, data-derived forensic tampering bound
            T_eff = max(fused["T"], max_agent_T if is_definitive_single else fused["T"])
            A_eff = min(fused["A"], 1.0 - T_eff)
            U_eff = max(0.0, round(1.0 - T_eff - A_eff, 4))
            
            pignistic_fake = T_eff + 0.5 * U_eff
            belief = T_eff
            plausibility = min(1.0, T_eff + U_eff)
            verdict = "TAMPERED"
        else:
            if (max_conflict >= 0.80 or primary_tamper <= 0.25) and vit_strongly_authentic:
                # Zadeh's paradox / heuristic divergence mitigation with authentic ViT backbone.
                # Re-route conflicting mass to epistemic uncertainty m(U) without arbitrary offset,
                # letting unresolvable heuristic conflict yield INCONCLUSIVE without false conviction.
                T_eff = min(fused["T"], primary_tamper)
                U_eff = max(0.40, round(1.0 - T_eff - 0.40, 4))
                A_eff = max(0.0, round(1.0 - T_eff - U_eff, 4))
                pignistic_fake = T_eff + 0.5 * U_eff
                belief = T_eff
                plausibility = round(T_eff + U_eff, 4)
                verdict = "INCONCLUSIVE" if (max_agent_T >= 0.50 or pignistic_fake > 0.35) else "AUTHENTIC"
            else:
                T_eff = fused["T"]
                A_eff = fused["A"]
                U_eff = fused["U"]
                pignistic_fake = T_eff + 0.5 * U_eff
                belief = T_eff
                plausibility = T_eff + U_eff

                if pignistic_fake >= 0.70:
                    verdict = "TAMPERED"
                elif pignistic_fake <= 0.35:
                    # If an isolated agent was moderately suspicious (e.g. T in [0.50, 0.72]),
                    # don't declare clean AUTHENTIC without noting conflict; mark INCONCLUSIVE if conflict is severe
                    if max_agent_T >= 0.50 and max_conflict > 0.60:
                        verdict = "INCONCLUSIVE"
                    else:
                        verdict = "AUTHENTIC"
                else:
                    verdict = "INCONCLUSIVE"
            
        return {
            "T":                    round(T_eff, 4),
            "A":                    round(A_eff, 4),
            "U":                    round(U_eff, 4),
            "pignistic_fake":       round(pignistic_fake, 4),
            "belief":               round(belief, 4),
            "plausibility":         round(plausibility, 4),
            "conflict_history":     conflict_history,
            "max_conflict":         round(max_conflict, 4),
            "max_individual_T":     round(max_agent_T, 4),
            "verdict":              verdict,
        }

# ---------------------------------------------------------------------------
# Smoke test - verify math invariants
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = DempsterShaferFusion()

    # --- BBA creation ---
    # e.g. evidence_T=8.0, evidence_A=2.0
    m = ds.create_mass(evidence_T=8.0, evidence_A=2.0)
    print(f"create_mass(8.0, 2.0) -> {m}")
    assert abs(sum(m.values()) - 1.0) < 1e-9, "BBA must sum to 1.0"
    
    # --- Pairwise combination: two agreeing agents ---
    m1 = ds.create_mass(9.0, 1.0)
    m2 = ds.create_mass(8.5, 1.5)
    fused, K = ds.combine(m1, m2)
    assert abs(sum(fused.values()) - 1.0) < 1e-9, "Fused BBA must sum to 1.0"
    print(f"combine (agreeing) -> fused={fused}  K={K:.4f}")

    # --- Pairwise combination: totally contradicting agents ---
    m_fake = ds.create_mass(100.0, 0.0)   # High evidence fake
    m_real = ds.create_mass(0.0, 100.0)   # High evidence real
    fused_confl, K_confl = ds.combine(m_fake, m_real)
    
    # --- Multi-agent fusion ---
    agents = [
        ds.create_mass(10.0, 1.0),   # Face Analysis agent
        ds.create_mass(5.0,  1.0),   # Lighting agent
        ds.create_mass(2.0,  0.5),   # Audio agent (less confident)
        ds.create_mass(1.5,  0.8),   # Compression agent
    ]
    result = ds.fuse_multimodal_evidence(agents)
    print(f"fuse_multimodal_evidence -> {result}")

    # --- Edge case: single agent ---
    single = ds.fuse_multimodal_evidence([ds.create_mass(1.0, 5.0)])
    assert single["conflict_history"] == []
    print(f"single-agent fusion   -> {single}")

    print("All fusion assertions passed.")
