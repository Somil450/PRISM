"""
src/agents/compression_watermark_agent.py
Compression Artifact, DCT Block Grid, & AI Watermark Forensic Agent.

Analyzes Error Level Analysis (ELA) residuals, Discrete Cosine Transform (DCT)
8x8 block boundary statistics, and high-frequency spectral generative signatures.
"""

from typing import Any, Dict, List, Optional, Tuple
import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

try:
    from .base_agent import AgentEvidence, BaseForensicAgent
except ImportError:
    from base_agent import AgentEvidence, BaseForensicAgent


class CompressionWatermarkAgent(BaseForensicAgent):
    """
    Forensic agent specializing in frequency-domain compression artifacts,
    JPEG block discontinuities, and synthetic watermark footprints.
    """

    def __init__(self, base_reliability: float = 0.85) -> None:
        super().__init__(name="Compression & Watermark Forensic Agent", base_reliability=base_reliability)

    def _compute_ela(self, pil_image: Image.Image, quality: int = 90) -> Tuple[np.ndarray, float, float, float]:
        """
        Computes Error Level Analysis (ELA).
        Returns:
            enhanced_ela_arr: uint8 visual array
            mean_residual: normalized float [0.0, 1.0]
            std_residual: localized variance float [0.0, 1.0]
            max_residual: normalized float [0.0, 1.0]
        """
        rgb = pil_image.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf)

        diff = ImageChops.difference(rgb, recompressed)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        enhanced = ImageEnhance.Brightness(diff).enhance(scale)

        diff_np = np.asarray(diff, dtype=np.float32)
        mean_res = float(np.mean(diff_np)) / 255.0
        std_res = float(np.std(diff_np)) / 255.0
        max_res = float(max_diff) / 255.0

        return np.array(enhanced), mean_res, std_res, max_res

    def _compute_dct_block_discontinuity(self, gray: np.ndarray) -> float:
        """
        Measures 8x8 JPEG block boundary artifact grid energy in both horizontal and
        vertical dimensions. Authentic JPEGs exhibit periodic gradients at multiples of
        8 pixels. Diffusion images and synthetic deepfakes often disrupt this grid.
        Returns the mean boundary-to-interior ratio across both axes.
        """
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.0

        # --- Horizontal axis: differences across row boundaries at multiples of 8 ---
        h_diff = np.abs(gray[7:h-1:8, :].astype(np.float32) - gray[8:h:8, :].astype(np.float32))
        h_intra = np.abs(gray[3:h-1:8, :].astype(np.float32) - gray[4:h:8, :].astype(np.float32))
        h_ratio = float(np.mean(h_diff)) / (float(np.mean(h_intra)) + 1e-6)

        # --- Vertical axis: differences across column boundaries at multiples of 8 ---
        v_diff = np.abs(gray[:, 7:w-1:8].astype(np.float32) - gray[:, 8:w:8].astype(np.float32))
        v_intra = np.abs(gray[:, 3:w-1:8].astype(np.float32) - gray[:, 4:w:8].astype(np.float32))
        v_ratio = float(np.mean(v_diff)) / (float(np.mean(v_intra)) + 1e-6)

        # Average both axes for a 2D block grid energy measure
        return (h_ratio + v_ratio) / 2.0

    def _detect_high_freq_generative_peaks(self, gray: np.ndarray) -> float:
        """
        Detects unnatural isolated delta peaks / ring artifacts in high-frequency 2D FFT.
        Applies a 2D Hann window to eliminate rectangular boundary spectral leakage (sinc cross),
        and measures peak prominence relative to the local spectral background.
        """
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.0

        # Apply 2D Hann window to remove boundary discontinuity cross
        window_2d = np.outer(np.hanning(h), np.hanning(w))
        windowed = (gray.astype(np.float32) - np.mean(gray)) * window_2d

        fshift = np.fft.fftshift(np.fft.fft2(windowed))
        log_mag = np.log(np.abs(fshift) + 1.0).astype(np.float32)

        # Smooth local background in frequency space
        local_bg = cv2.blur(log_mag, (9, 9))
        residual = log_mag - local_bg

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        high_pass_mask = dist_from_center > (min(h, w) * 0.25)

        high_freq_residual = residual[high_pass_mask]
        if len(high_freq_residual) == 0:
            return 0.0

        # Isolated peak prominence over local background
        max_prominence = float(np.max(high_freq_residual))
        # Natural optical photos have local prominence < 3.2.
        # Artificial GAN/diffusion periodic lattices exhibit isolated spike prominence > 4.5.
        hf_peak_anomaly = min(1.0, max(0.0, (max_prominence - 3.2) / 4.0))
        return hf_peak_anomaly

    def analyze(self, image_input: Any, **kwargs: Any) -> AgentEvidence:
        """
        Executes compression and frequency domain watermark forensic inspection.
        """
        self._validate_no_leakage_kwargs(kwargs)
        if isinstance(image_input, str):
            raise ValueError(
                "CompressionWatermarkAgent.analyze() does not accept file paths or filenames. "
                "Pass in-memory PIL Image or numpy array directly to prevent label leakage."
            )
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            rgb_arr = np.array(pil_img)
            gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                rgb_arr = image_input
                gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_input
                rgb_arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_arr)
        else:
            raise ValueError("Unsupported input format for CompressionWatermarkAgent")

        # 1. ELA Analysis
        ela_visual, mean_ela, std_ela, max_ela = self._compute_ela(pil_img, quality=90)

        # 2. DCT 8x8 Grid Discontinuity
        dct_grid_ratio = self._compute_dct_block_discontinuity(gray)

        # 3. High-Frequency Generative Peak Anomaly
        hf_peak_anomaly = self._detect_high_freq_generative_peaks(gray)

        findings: List[str] = []

        # Scoring Logic
        ela_anomaly = min(1.0, max(0.0, (mean_ela * 5.0 + std_ela * 10.0 + max_ela * 0.35)))
        prob_tampered = min(1.0, max(0.0, (ela_anomaly * 0.50 + hf_peak_anomaly * 0.50)))
        
        # If strong localized ELA discrepancy is observed, elevate tampering probability
        if ela_anomaly > 0.60:
            prob_tampered = max(prob_tampered, min(1.0, ela_anomaly * 0.70 + hf_peak_anomaly * 0.30))
            
        reliability = 0.85

        if prob_tampered > 0.65:
            verdict = "TAMPERED"
            findings.append(f"Elevated Error Level Analysis (ELA) residual detected (mean discrepancy: {mean_ela:.3f}).")
            if hf_peak_anomaly > 0.40:
                findings.append("Unnatural high-frequency Fourier spectral peaks detected (consistent with latent diffusion / GAN footprint).")
        elif prob_tampered < 0.35:
            verdict = "AUTHENTIC"
            findings.append(f"Uniform compression error surface across image plane (mean ELA: {mean_ela:.3f}).")
            findings.append("Absence of synthetic Fourier spectral spikes in high-pass frequency domain.")
        else:
            verdict = "INCONCLUSIVE"
            findings.append(f"Moderate compression gradient (ELA: {mean_ela:.3f}); consistent with standard multi-pass encoding.")

        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)

        telemetry = {
            "mean_ela_residual": round(mean_ela, 4),
            "max_ela_residual": round(max_ela, 4),
            "dct_grid_ratio": round(dct_grid_ratio, 4),
            "hf_spectral_peak_score": round(hf_peak_anomaly, 4),
        }

        return AgentEvidence(
            agent_name=self.name,
            prob_tampered=prob_tampered,
            reliability=reliability,
            bba_mass=bba,
            verdict=verdict,
            findings=findings,
            telemetry=telemetry,
            visual_artifact=ela_visual,
        )


# ---------------------------------------------------------------------------
# Unit / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = CompressionWatermarkAgent()

    print(f"Testing {agent.name}...")
    print("-" * 60)

    # Test 1: Clean uniform image
    clean = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    evidence_clean = agent.analyze(clean)
    print(f"Test 1 (Clean Image) -> Verdict: {evidence_clean.verdict} | P(Tampered): {evidence_clean.prob_tampered:.2f}")
    assert evidence_clean.prob_tampered < 0.50
    assert evidence_clean.visual_artifact is not None

    # Test 2: Spliced image with high frequency noise patch
    spliced_arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    spliced_arr[50:120, 50:120] = np.random.randint(0, 255, (70, 70, 3), dtype=np.uint8)
    spliced = Image.fromarray(spliced_arr)
    evidence_spliced = agent.analyze(spliced)
    print(f"Test 2 (Spliced Patch) -> Verdict: {evidence_spliced.verdict} | P(Tampered): {evidence_spliced.prob_tampered:.2f}")
    assert evidence_spliced.prob_tampered > 0.30

    print("-" * 60)
    print("All CompressionWatermarkAgent unit assertions passed successfully!")
