"""
src/agents/illumination_agent.py
Physical Forensic Agent: 3D Lighting & Illumination Vector Consistency.

Analyzes 2D spatial luminance gradients (Sobel dx/dy), estimates primary light source
direction vectors for foreground vs. background, and flags contradictory illumination angles.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

try:
    from .base_agent import AgentEvidence, BaseForensicAgent
except ImportError:
    from base_agent import AgentEvidence, BaseForensicAgent


class IlluminationAgent(BaseForensicAgent):
    """
    Forensic agent for verifying lighting environment physics,
    surface gradient orientation, and foreground/background light source consistency.
    """

    def __init__(self, base_reliability: float = 0.82) -> None:
        super().__init__(name="Illumination & 3D Lighting Consistency Agent", base_reliability=base_reliability)

    def _estimate_vector_from_gradients(self, sobel_x: np.ndarray, sobel_y: np.ndarray, gray: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Estimates dominant 2D lighting direction (Lx, Ly) and directional coherence.
        Returns (Lx, Ly, angle_degrees, coherence).
        """
        if sobel_x.size == 0 or sobel_y.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        weights = (gray.astype(np.float32) / 255.0) ** 2
        wx = sobel_x * weights
        wy = sobel_y * weights

        lx = float(np.sum(wx))
        ly = float(np.sum(wy))

        norm = np.sqrt(lx**2 + ly**2)
        total_energy = float(np.sum(np.sqrt(wx**2 + wy**2))) + 1e-7
        coherence = norm / total_energy

        if norm < 1e-4 or coherence < 0.015:
            # Diffuse / symmetric cancellation: no single dominant directional light
            return 0.0, 0.0, 0.0, float(coherence)

        lx_unit = lx / (norm + 1e-7)
        ly_unit = ly / (norm + 1e-7)
        angle_deg = float(np.degrees(np.arctan2(ly_unit, lx_unit))) % 360.0
        return lx_unit, ly_unit, angle_deg, float(coherence)

    def _render_illumination_map(
        self,
        rgb_img: np.ndarray,
        fg_mask: np.ndarray,
        fg_vec: Tuple[float, float],
        bg_vec: Tuple[float, float],
    ) -> np.ndarray:
        """
        Renders a forensic visual showing foreground vs. background light vectors.
        """
        canvas = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_LINEAR).copy()
        h, w = canvas.shape[:2]

        # Draw foreground vector in Green (Center)
        cx, cy = w // 2, h // 2
        fg_end = (int(cx + fg_vec[0] * 40), int(cy + fg_vec[1] * 40))
        cv2.arrowedLine(canvas, (cx, cy), fg_end, (0, 255, 100), 2, tipLength=0.3)
        cv2.putText(canvas, "FG Light", (cx - 30, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 100), 1)

        # Draw background vector in Orange (Top-Left)
        bg_start = (40, 40)
        bg_end = (int(40 + bg_vec[0] * 35), int(40 + bg_vec[1] * 35))
        cv2.arrowedLine(canvas, bg_start, bg_end, (255, 165, 0), 2, tipLength=0.3)
        cv2.putText(canvas, "BG Light", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 165, 0), 1)

        return canvas

    def analyze(self, image_input: Any, **kwargs: Any) -> AgentEvidence:
        """
        Executes 3D lighting vector estimation and angular divergence forensics.
        """
        self._validate_no_leakage_kwargs(kwargs)
        if isinstance(image_input, str):
            raise ValueError(
                "IlluminationAgent.analyze() does not accept file paths or filenames. "
                "Pass in-memory PIL Image or numpy array directly to prevent label leakage."
            )
        findings: List[str] = []

        # Convert input to RGB numpy
        if isinstance(image_input, Image.Image):
            rgb_np = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            rgb_np = image_input if image_input.shape[2] == 3 else cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image input type for IlluminationAgent")

        h, w = rgb_np.shape[:2]
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        # 1. Compute Full-Frame 2D Sobel Gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 2. Segment Center Foreground vs Perimeter Background
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_margin_y = int(h * 0.20)
        fg_margin_x = int(w * 0.20)
        fg_mask[fg_margin_y:h - fg_margin_y, fg_margin_x:w - fg_margin_x] = 255
        
        # Erode foreground mask by 4px to avoid border seam spikes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        fg_mask_eroded = cv2.erode(fg_mask, kernel) > 0
        bg_mask = fg_mask == 0

        # 3. Estimate Dominant Illumination Vectors & Coherence
        fg_lx, fg_ly, fg_angle, fg_coherence = self._estimate_vector_from_gradients(
            sobel_x[fg_mask_eroded], sobel_y[fg_mask_eroded], gray[fg_mask_eroded]
        )
        bg_lx, bg_ly, bg_angle, bg_coherence = self._estimate_vector_from_gradients(
            sobel_x[bg_mask], sobel_y[bg_mask], gray[bg_mask]
        )

        # Check for radial lens vignetting / optical falloff
        y_grid, x_grid = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        rx = (x_grid - cx).astype(np.float32)
        ry = (y_grid - cy).astype(np.float32)
        r_norm = np.sqrt(rx**2 + ry**2) + 1e-6
        radial_proj = (sobel_x * (rx / r_norm) + sobel_y * (ry / r_norm))
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2) + 1e-6
        radial_ratio = float(np.mean(radial_proj[bg_mask] / grad_mag[bg_mask]))
        is_radial_vignette = radial_ratio < -0.09 and bg_coherence < 0.02

        # 4. Anomaly Evaluation
        # If lighting is diffuse, symmetric, low-coherence, or lens-vignetted, directional vector comparison is physically invalid
        min_coherence = min(fg_coherence, bg_coherence)
        if is_radial_vignette or min_coherence < 0.15 or (fg_coherence * bg_coherence < 0.03):
            prob_tampered = 0.50
            reliability = 0.0
            verdict = "INCONCLUSIVE"
            findings.append("Scene exhibits diffuse, omnidirectional, or low-coherence lighting. Directional lighting check uninformative.")
            dot_product = 1.0
            angular_divergence = 0.0
        else:
            # Dot product between unit light vectors
            dot_product = fg_lx * bg_lx + fg_ly * bg_ly
            dot_product = max(-1.0, min(1.0, dot_product))
            angular_divergence = float(np.degrees(np.arccos(dot_product)))

            coherence_scale = min(1.0, min_coherence / 0.30)
            if angular_divergence > 65.0:
                prob_tampered = min(1.0, 0.65 + (angular_divergence - 65.0) / 115.0 * 0.35)
                verdict = "TAMPERED"
                findings.append(f"Contradictory scene illumination detected (angular divergence: {angular_divergence:.1f}°).")
                findings.append("Foreground subject key light direction conflicts with background ambient shadow gradient.")
                reliability = round(0.85 * coherence_scale, 3)
            elif angular_divergence < 35.0:
                prob_tampered = max(0.0, (angular_divergence / 35.0) * 0.30)
                verdict = "AUTHENTIC"
                findings.append(f"Consistent global illumination (light vector divergence: {angular_divergence:.1f}°).")
                findings.append("Subject highlights and background ambient gradients align physically.")
                reliability = round(0.80 * coherence_scale, 3)
            else:
                prob_tampered = 0.50
                verdict = "INCONCLUSIVE"
                findings.append(f"Moderate illumination divergence ({angular_divergence:.1f}°); multi-source or diffuse lighting.")
                reliability = round(0.50 * coherence_scale, 3)

        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)
        visual = self._render_illumination_map(rgb_np, None, (fg_lx, fg_ly), (bg_lx, bg_ly))

        telemetry = {
            "fg_light_angle_deg": round(fg_angle, 1),
            "bg_light_angle_deg": round(bg_angle, 1),
            "fg_coherence": round(fg_coherence, 4),
            "bg_coherence": round(bg_coherence, 4),
            "angular_divergence_deg": round(angular_divergence, 1),
            "light_vector_dot_product": round(dot_product, 3),
        }

        return AgentEvidence(
            agent_name=self.name,
            prob_tampered=prob_tampered,
            reliability=reliability,
            bba_mass=bba,
            verdict=verdict,
            findings=findings,
            telemetry=telemetry,
            visual_artifact=visual,
        )


# ---------------------------------------------------------------------------
# Unit / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = IlluminationAgent()

    print(f"Testing {agent.name}...")
    print("-" * 60)

    # Test 1: Uniform consistent illumination (Top-Left light across entire image)
    h, w = 224, 224
    y, x = np.mgrid[0:h, 0:w]
    consistent_img = (np.clip(255 - (x * 0.5 + y * 0.5), 20, 240)).astype(np.uint8)
    consistent_rgb = cv2.cvtColor(consistent_img, cv2.COLOR_GRAY2RGB)

    ev_consistent = agent.analyze(consistent_rgb)
    print(f"Test 1 (Consistent Lighting) -> Verdict: {ev_consistent.verdict} | Divergence: {ev_consistent.telemetry['angular_divergence_deg']}° | P(Tampered): {ev_consistent.prob_tampered:.2f}")
    assert ev_consistent.telemetry["angular_divergence_deg"] < 35.0
    assert ev_consistent.verdict == "AUTHENTIC"

    # Test 2: Inverted lighting (Foreground lit from right, background lit from left)
    inverted_rgb = consistent_rgb.copy()
    # Invert the center crop
    inverted_rgb[45:180, 45:180] = np.fliplr(inverted_rgb[45:180, 45:180])

    ev_inverted = agent.analyze(inverted_rgb)
    print(f"Test 2 (Inverted Splice Lighting) -> Verdict: {ev_inverted.verdict} | Divergence: {ev_inverted.telemetry['angular_divergence_deg']}° | P(Tampered): {ev_inverted.prob_tampered:.2f}")
    assert ev_inverted.telemetry["angular_divergence_deg"] > 60.0
    assert ev_inverted.prob_tampered > 0.60

    print("-" * 60)
    print("All IlluminationAgent unit assertions passed successfully!")
