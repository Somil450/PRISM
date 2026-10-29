import cv2
import numpy as np
from typing import Any, Optional, Tuple
from .base_agent import BaseForensicAgent, AgentEvidence

class RPPGAgent(BaseForensicAgent):
    """
    Biological Liveness Agent using Remote Photoplethysmography (rPPG).
    Extracts the heart rate signal from the green channel of the face ROI
    and analyzes its physiological plausibility. Deepfakes typically lack
    consistent cardiovascular micro-color variations.
    """

    def __init__(self):
        super().__init__(name="rPPG_Liveness", base_reliability=0.85, is_positive_authenticator=True)

    def analyze(
        self,
        raw_frames: list,
        has_face: bool = True,
        fps: float = 30.0,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        **kwargs: Any,
    ) -> AgentEvidence:
        """
        Extracts cardiovascular pulse from green channel using CHROM rPPG.
        If no human face is present, gracefully bypasses with pure epistemic uncertainty.

        Args:
            raw_frames:  List of raw BGR frames (in-memory).
            has_face:    Whether a human face is present in the sequence.
            fps:         Actual video frame rate (used for FFT frequency axis).
            face_bbox:   Optional (x, y, w, h) bounding box from a face detector
                         (e.g. YuNet). When provided, crops the ROI from this
                         region instead of the center-35% heuristic.
        """
        self._validate_no_leakage_kwargs(kwargs)
        if isinstance(raw_frames, str):
            raise ValueError(
                "RPPGAgent.analyze() does not accept file paths or filenames. "
                "Pass in-memory list of video frames directly to prevent label leakage."
            )

        # Case 1: No human face detected
        if not has_face:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=["No biological facial subject detected in frame sequence. Biological pulse (rPPG) check bypassed."],
                telemetry={"rPPG_SNR": 0.0, "status": "bypassed_no_face"},
                visual_artifact=None,
            )

        # Case 2: Insufficient frames - minimum is now 8 (reduced from 15)
        if not raw_frames or len(raw_frames) < 8:
            bba = self.create_bba(prob_tampered=0.5, custom_reliability=0.1)
            return AgentEvidence(
                agent_name=self.name, prob_tampered=0.5, reliability=0.1, bba_mass=bba,
                verdict="INCONCLUSIVE", findings=["Insufficient frames for rPPG biological liveness extraction."],
                telemetry={"rPPG_SNR": 0.0, "status": "insufficient_frames"}, visual_artifact=None
            )

        # Warn when frame count is low enough to yield low-SNR estimates
        low_snr_warning = []
        if len(raw_frames) < 32:
            low_snr_warning.append(
                f"WARNING: Only {len(raw_frames)} frames provided (< 32). "
                "rPPG SNR may be low and frequency resolution limited."
            )

        r_signals = []
        g_signals = []
        b_signals = []
        skin_pixel_ratios = []

        for bgr in raw_frames:
            h, w = bgr.shape[:2]

            # Use YuNet/detector bounding box if provided, else center-35% heuristic
            if face_bbox is not None:
                bx, by, bw, bh = face_bbox
                # Clamp to frame bounds
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = max(1, min(bw, w - bx))
                bh = max(1, min(bh, h - by))
                roi = bgr[by:by + bh, bx:bx + bw]
            else:
                # Crop to center 35% to approximate forehead/cheeks ROI
                cx, cy = w // 2, h // 2
                roi_h, roi_w = int(h * 0.35), int(w * 0.35)
                roi = bgr[cy - roi_h//2 : cy + roi_h//2, cx - roi_w//2 : cx + roi_w//2]

            if roi.size == 0:
                continue

            # Inclusive skin tone mask for Fitzpatrick Scales I-VI
            # (South Asian, African, East Asian, Caucasian)
            # Y in [20, 255], Cr in [125, 185], Cb in [65, 140]
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, np.array([20, 125, 65]), np.array([255, 185, 140]))
            skin_pixels = np.count_nonzero(skin_mask)
            skin_ratio = float(skin_pixels) / float(skin_mask.size)
            skin_pixel_ratios.append(skin_ratio)

            if skin_pixels > 20:
                mean_b = float(np.mean(roi[:, :, 0][skin_mask > 0]))
                mean_g = float(np.mean(roi[:, :, 1][skin_mask > 0]))
                mean_r = float(np.mean(roi[:, :, 2][skin_mask > 0]))
            else:
                mean_b = float(np.mean(roi[:, :, 0]))
                mean_g = float(np.mean(roi[:, :, 1]))
                mean_r = float(np.mean(roi[:, :, 2]))

            b_signals.append(mean_b)
            g_signals.append(mean_g)
            r_signals.append(mean_r)

        # If center region lacks biological skin tones, bypass
        mean_skin_ratio = float(np.mean(skin_pixel_ratios)) if skin_pixel_ratios else 0.0
        if mean_skin_ratio < 0.10:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=["Center region lacks dermal/facial skin tones (skin ratio < 10%). Biological pulse check bypassed."],
                telemetry={"rPPG_SNR": 0.0, "skin_ratio": round(mean_skin_ratio, 3), "status": "no_skin_detected"},
                visual_artifact=None,
            )

        # --- CHROM (Chrominance rPPG) Melanin-Invariant Orthogonal Projection ---
        # Wang et al. / de Haan & Jeanne: Projects R, G, B signals orthogonally to the
        # physiological melanin absorption vector, normalizing across Fitzpatrick phototypes I-VI.
        R_arr = np.array(r_signals, dtype=np.float32)
        G_arr = np.array(g_signals, dtype=np.float32)
        B_arr = np.array(b_signals, dtype=np.float32)

        mean_R = np.mean(R_arr) + 1e-6
        mean_G = np.mean(G_arr) + 1e-6
        mean_B = np.mean(B_arr) + 1e-6

        Rn = R_arr / mean_R
        Gn = G_arr / mean_G
        Bn = B_arr / mean_B

        Xs = 3.0 * Rn - 2.0 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn

        std_Xs = float(np.std(Xs))
        std_Ys = float(np.std(Ys))
        alpha = std_Xs / (std_Ys + 1e-6)
        signal = Xs - alpha * Ys

        raw_std = float(np.std(signal))

        # Detrending (remove DC component and slow illumination drift)
        window_size = min(5, len(signal))
        trend = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        detrended = signal - trend

        # Physiological check: Micro-vascular pulse amplitude bounds
        if raw_std < 0.005 or raw_std > 8.0:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=[f"Luminance variance ({raw_std:.2f}) outside physiological cardiovascular bounds. Liveness check bypassed."],
                telemetry={"rPPG_SNR": 0.0, "signal_std": round(raw_std, 3), "status": "non_physiological_variance"},
                visual_artifact=None,
            )

        # Normalize detrended signal
        if np.std(detrended) > 1e-6:
            detrended = detrended / np.std(detrended)

        # Perform FFT to find peak frequency using the actual (dynamic) fps
        fft_result = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended), d=1.0 / fps)

        # Human heart rate is typically 42-180 BPM (0.7 Hz to 3.0 Hz)
        valid_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))[0]

        # Total spectrum energy (excluding DC index 0)
        total_spectrum_energy = float(np.sum(fft_result[1:])) if len(fft_result) > 1 else 1.0

        # Fallback for exactly 1 bin in the physiological band:
        # still compute a partial SNR but flag reliability as 0.25
        if len(valid_idx) == 0:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.10,
                bba_mass=self.create_bba(0.50, custom_reliability=0.10),
                verdict="INCONCLUSIVE",
                findings=low_snr_warning + [
                    "No frequency bins in 0.7-3.0 Hz physiological band. "
                    "Cannot resolve heart rate peak from current frame count."
                ],
                telemetry={"rPPG_SNR": 0.0, "bins_in_band": 0, "fps_used": fps},
                visual_artifact=None,
            )

        partial_snr_mode = len(valid_idx) == 1
        partial_reliability = 0.25  # single-bin fallback reliability cap

        # Calculate Signal-to-Noise Ratio relative to entire spectral distribution
        valid_fft = fft_result[valid_idx]
        peak_energy = float(np.max(valid_fft))
        in_band_energy = float(np.sum(valid_fft))

        # Guard: SNR must be fraction of overall spectrum energy, not just peak / in-band
        snr = peak_energy / (total_spectrum_energy + 1e-6)
        in_band_fraction = in_band_energy / (total_spectrum_energy + 1e-6)

        # Authentic biological pulse: clean peak in band AND majority of non-DC energy in band
        if snr > 0.45 and in_band_fraction > 0.40:
            prob_tampered = 0.10
            finding = (
                f"Consistent biological pulse detected (rPPG spectral peak: {snr:.2f}). "
                "Indicator of human liveness."
            )
            verdict = "AUTHENTIC"
            reliability = partial_reliability if partial_snr_mode else self.base_reliability
        elif in_band_fraction < 0.20:
            prob_tampered = 0.85
            finding = (
                f"Absence of physiological pulse frequencies (band fraction: {in_band_fraction:.2f}). "
                "Indicator of synthetic/modified sequence."
            )
            verdict = "TAMPERED"
            reliability = partial_reliability if partial_snr_mode else self.base_reliability
        else:
            prob_tampered = 0.50
            finding = (
                f"Diffuse spectral energy across frequency spectrum (SNR: {snr:.2f}). "
                "Inconclusive pulse extraction."
            )
            verdict = "INCONCLUSIVE"
            reliability = partial_reliability if partial_snr_mode else 0.30

        if partial_snr_mode:
            low_snr_warning.append(
                "Only 1 frequency bin in physiological band - partial SNR computed with reduced reliability."
            )

        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)

        return AgentEvidence(
            agent_name=self.name, prob_tampered=prob_tampered, reliability=reliability,
            bba_mass=bba, verdict=verdict,
            findings=low_snr_warning + [finding],
            telemetry={
                "rPPG_SNR": float(snr),
                "in_band_fraction": round(in_band_fraction, 3),
                "bins_in_band": len(valid_idx),
                "fps_used": fps,
                "partial_snr_mode": partial_snr_mode,
            },
            visual_artifact=None
        )
