"""
src/agents/lip_sync_agent.py
Cross-Modal Forensic Agent: Lip-Sync & Audio-Visual (AV) Synchrony.

Analyzes cross-modal correlation between acoustic speech energy (phonemes) and
mouth vertical aperture / lip displacement dynamics (visemes) to detect AI dubbing and Wav2Lip deepfakes.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.signal as signal

try:
    from .base_agent import AgentEvidence, BaseForensicAgent
    from .audio_voice_agent import NoSpeechDetectedError
except ImportError:
    from base_agent import AgentEvidence, BaseForensicAgent
    try:
        from audio_voice_agent import NoSpeechDetectedError
    except ImportError:
        class NoSpeechDetectedError(ValueError):
            pass


class LipSyncAgent(BaseForensicAgent):
    """
    Forensic agent for analyzing cross-modal temporal synchrony between
    audio speech envelopes and visual mouth motion kinematics.
    """

    def __init__(self, base_reliability: float = 0.86) -> None:
        super().__init__(name="Lip-Sync & Audio-Visual Synchrony Agent", base_reliability=base_reliability)

    def _extract_mouth_aperture_series(
        self,
        frames: List[np.ndarray],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Extracts temporal mouth aperture and dynamic viseme opening energy for each frame.
        Applies spatial Gaussian low-pass filtering to decouple static beard/moustache
        follicle texture while isolating low-frequency oral aperture expansion.
        Returns 1D array of shape [num_frames].
        """
        apertures = []
        prev_roi_blur: Optional[np.ndarray] = None

        for f in frames:
            if f.ndim == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                gray = f

            h, w = gray.shape
            if face_bbox is not None:
                bx, by, bw, bh = face_bbox
                # Clamp within frame boundaries
                bx = max(0, min(bx, w - 1))
                by = max(0, min(by, h - 1))
                bw = max(1, min(bw, w - bx))
                bh = max(1, min(bh, h - by))
                # Mouth region is typically in lower 35% of detected face
                m_ymin = by + int(bh * 0.62)
                m_ymax = by + int(bh * 0.98)
                m_xmin = bx + int(bw * 0.20)
                m_xmax = bx + int(bw * 0.80)
                mouth_roi = gray[m_ymin:m_ymax, m_xmin:m_xmax]
            else:
                mouth_roi = gray[int(h * 0.60):int(h * 0.95), int(w * 0.25):int(w * 0.75)]

            if mouth_roi.size == 0:
                apertures.append(0.0)
                continue

            # Low-pass filter to remove high-frequency beard/stubble texture
            roi_blur = cv2.GaussianBlur(mouth_roi, (7, 7), 2.0)

            # Spatial aperture variance of smoothed oral cavity
            spatial_energy = float(np.std(roi_blur))

            # Temporal differential motion (moving lips vs stationary facial hair)
            if prev_roi_blur is not None and prev_roi_blur.shape == roi_blur.shape:
                temporal_motion = float(np.mean(np.abs(roi_blur.astype(np.float32) - prev_roi_blur.astype(np.float32))))
            else:
                temporal_motion = 0.0
            prev_roi_blur = roi_blur

            # Combined aperture signal: spatial cavity volume + dynamic articulation motion
            combined_metric = 0.75 * spatial_energy + 0.25 * (temporal_motion * 5.0)
            apertures.append(combined_metric)

        arr = np.array(apertures, dtype=np.float32)
        if len(arr) > 0 and np.std(arr) > 1e-5:
            arr = (arr - np.mean(arr)) / np.std(arr)
        return arr

    def _extract_audio_envelope_series(self, waveform: np.ndarray, sr: int, num_frames: int, fps: float = 30.0) -> np.ndarray:
        """
        Extracts smoothed RMS speech energy envelope binned to match video frame count.
        """
        if len(waveform) == 0 or num_frames == 0:
            return np.zeros(num_frames, dtype=np.float32)

        # Compute Hilbert envelope or squared RMS energy
        samples_per_frame = int(sr / fps)
        envelope = []

        for i in range(num_frames):
            start = i * samples_per_frame
            end = min(len(waveform), (i + 1) * samples_per_frame)
            if start < len(waveform):
                chunk = waveform[start:end]
                rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
            else:
                rms = 0.0
            envelope.append(rms)

        arr = np.array(envelope, dtype=np.float32)
        if len(arr) > 0 and np.std(arr) > 1e-5:
            arr = (arr - np.mean(arr)) / np.std(arr)
        return arr

    def _render_synchrony_visual(self, mouth_series: np.ndarray, audio_series: np.ndarray) -> np.ndarray:
        """
        Renders a temporal waveform alignment chart (224x224 RGB).
        """
        canvas = np.full((224, 224, 3), 18, dtype=np.uint8)  # Dark background
        n = min(len(mouth_series), len(audio_series))
        if n < 2:
            return canvas

        xs = np.linspace(10, 214, n).astype(int)

        # Plot Audio Envelope in Cyan (top half)
        norm_a = np.clip((audio_series[:n] + 2.5) / 5.0, 0, 1)
        ys_a = (100 - (norm_a * 80)).astype(int)
        for i in range(n - 1):
            cv2.line(canvas, (xs[i], ys_a[i]), (xs[i + 1], ys_a[i + 1]), (255, 255, 0), 2)
        cv2.putText(canvas, "Audio Speech Energy", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # Plot Mouth Aperture in Magenta (bottom half)
        norm_m = np.clip((mouth_series[:n] + 2.5) / 5.0, 0, 1)
        ys_m = (210 - (norm_m * 80)).astype(int)
        for i in range(n - 1):
            cv2.line(canvas, (xs[i], ys_m[i]), (xs[i + 1], ys_m[i + 1]), (200, 50, 255), 2)
        cv2.putText(canvas, "Mouth Opening Aperture", (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 50, 255), 1)

        return canvas

    def analyze(self, media_input: Any, fail_loud: bool = False, **kwargs: Any) -> AgentEvidence:
        """
        Calculates cross-modal speech-lip phoneme-viseme synchrony.
        If fail_loud=True, raises NoSpeechDetectedError when audio waveform is missing or silent.

        Args:
            media_input: Tuple of (frames_list, (audio_waveform, sr)) or dict.
            fail_loud: If True, raises NoSpeechDetectedError when speech is absent.
        """
        self._validate_no_leakage_kwargs(kwargs)
        if isinstance(media_input, str):
            raise ValueError(
                "LipSyncAgent.analyze() does not accept file paths or filenames. "
                "Pass in-memory tuple of (frames, (audio, sr)) directly to prevent label leakage."
            )

        frames: List[np.ndarray] = []
        waveform: Optional[np.ndarray] = None
        sr: int = 16000
        fps: float = float(kwargs.get("fps", 30.0))
        face_bbox: Optional[Tuple[int, int, int, int]] = kwargs.get("face_bbox", None)
        frame_offset_samples: int = int(kwargs.get("frame_offset_samples", 0))

        if isinstance(media_input, tuple) and len(media_input) == 2:
            frames, audio_data = media_input
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                waveform, sr = audio_data
            elif isinstance(audio_data, np.ndarray):
                waveform = audio_data
        elif isinstance(media_input, dict):
            frames = media_input.get("frames", [])
            audio_val = media_input.get("audio", None)
            if isinstance(audio_val, tuple) and len(audio_val) == 2:
                waveform, sr = audio_val
            elif isinstance(audio_val, np.ndarray):
                waveform = audio_val
            else:
                waveform = None
            sr = media_input.get("sr", sr)
            fps = float(media_input.get("fps", fps))
            face_bbox = media_input.get("face_bbox", face_bbox)
            frame_offset_samples = int(media_input.get("frame_offset_samples", frame_offset_samples))

        # Synchronize audio by applying temporal frame offset if audio starts before the video frame window
        if waveform is not None and frame_offset_samples > 0 and len(waveform) > frame_offset_samples:
            waveform = waveform[frame_offset_samples:]

        findings: List[str] = []

        # If audio or video stream is absent or silent -> handle
        rms = float(np.sqrt(np.mean(waveform ** 2))) if (waveform is not None and len(waveform) > 0) else 0.0
        if not frames or waveform is None or len(frames) < 6 or len(waveform) < 1000 or rms < 1e-4:
            if fail_loud:
                raise NoSpeechDetectedError("Silent or missing audio stream: no voiced speech available for lip synchrony.")
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=["Cross-modal audio-visual stream not provided or silent. Lip-sync check bypassed."],
                telemetry={"av_synced": False, "is_silent": True, "has_speech": False, "fps_used": fps},
                visual_artifact=None,
            )

        mouth_series = self._extract_mouth_aperture_series(frames, face_bbox=face_bbox)
        audio_series = self._extract_audio_envelope_series(waveform, sr, len(frames), fps=fps)

        # Cross-correlation to determine temporal offset lag within physiological window (+/- 300ms)
        corr = signal.correlate(audio_series, mouth_series, mode="full")
        lags = signal.correlation_lags(len(audio_series), len(mouth_series))
        
        max_lag_frames = int(fps * 0.35)  # +/- 350ms window
        valid_idx = np.where(np.abs(lags) <= max_lag_frames)[0]

        if len(valid_idx) > 0:
            windowed_corr = corr[valid_idx]
            windowed_lags = lags[valid_idx]
            best_rel_idx = int(np.argmax(windowed_corr))
            optimal_lag_frames = int(windowed_lags[best_rel_idx])
            max_corr = float(windowed_corr[best_rel_idx]) / (len(mouth_series) + 1e-5)
        else:
            optimal_lag_frames = 0
            max_corr = 0.0

        optimal_lag_ms = float((optimal_lag_frames / fps) * 1000.0)

        # Zero-lag Pearson correlation
        if len(audio_series) > 1 and np.std(audio_series) > 1e-4 and np.std(mouth_series) > 1e-4:
            zero_lag_corr = float(np.corrcoef(audio_series, mouth_series)[0, 1])
        else:
            zero_lag_corr = 0.0

        if np.isnan(zero_lag_corr):
            zero_lag_corr = 0.0

        # Anomaly logic calibrated for cross-lingual / Indian regional speech and Wav2Lip deepfakes:
        # Authentic speech: |lag| <= 55ms and max_corr >= 0.28
        # AI Dubbed / Desynced Wav2Lip: |lag| > 90ms or max_corr < 0.15
        lag_penalty = min(1.0, max(0.0, (abs(optimal_lag_ms) - 50.0) / 90.0))
        corr_penalty = min(1.0, max(0.0, (0.40 - max_corr) / 0.35))

        prob_tampered = min(1.0, max(0.0, lag_penalty * 0.70 + corr_penalty * 0.30))
        reliability = 0.85

        if prob_tampered > 0.60:
            verdict = "TAMPERED"
            findings.append(f"Audio-Visual temporal desynchronization detected (temporal offset lag: {optimal_lag_ms:+.1f} ms).")
            findings.append(f"Low phoneme-to-viseme kinetic correlation (synchrony index: {max_corr:.2f}, characteristic of AI voice dubbing/Wav2Lip).")
        elif prob_tampered < 0.35:
            verdict = "AUTHENTIC"
            findings.append(f"Natural phoneme-to-viseme synchrony (temporal lag: {optimal_lag_ms:+.1f} ms within natural human latency bounds).")
            findings.append(f"Speech acoustic envelope strongly aligns with mouth opening kinematics (correlation: {max_corr:.2f}).")
        else:
            verdict = "INCONCLUSIVE"
            findings.append(f"Moderate cross-modal synchrony (lag: {optimal_lag_ms:+.1f} ms, corr: {max_corr:.2f}).")

        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)
        visual = self._render_synchrony_visual(mouth_series, audio_series)

        telemetry = {
            "temporal_lag_ms": round(optimal_lag_ms, 1),
            "peak_correlation": round(max_corr, 3),
            "zero_lag_correlation": round(zero_lag_corr, 3),
            "frames_evaluated": len(frames),
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
    agent = LipSyncAgent()

    print(f"Testing {agent.name}...")
    print("-" * 60)

    # Test 1: Generate synchronized AV sequence (single distinct speech pulse at t=0.4s)
    num_frames = 30
    sr = 16000
    fps = 30.0
    t_audio = np.linspace(0, num_frames / fps, int(sr * (num_frames / fps)))

    # Audio speech burst centered at t=0.4s (Gaussian envelope)
    audio_env = np.exp(-((t_audio - 0.4)**2) / (2 * 0.08**2))
    audio_wave = (np.sin(2 * np.pi * 300 * t_audio) * audio_env).astype(np.float32)

    # Video frames have mouth opening centered at frame index corresponding to t=0.4s (frame 12)
    synced_frames = []
    for i in range(num_frames):
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        open_val = int(255 * np.exp(-(((i / fps) - 0.4)**2) / (2 * 0.08**2)))
        canvas[150:150 + max(4, open_val // 10), 90:134] = open_val
        synced_frames.append(canvas)

    ev_synced = agent.analyze((synced_frames, (audio_wave, sr)))
    print(f"Test 1 (Synchronized AV) -> Verdict: {ev_synced.verdict} | Lag: {ev_synced.telemetry['temporal_lag_ms']} ms | P(Tampered): {ev_synced.prob_tampered:.2f}")
    assert ev_synced.visual_artifact is not None
    assert abs(ev_synced.telemetry["temporal_lag_ms"]) < 40.0
    assert ev_synced.verdict == "AUTHENTIC"

    # Test 2: Shift audio by 200ms (speech burst moves to t=0.6s)
    desynced_env = np.exp(-((t_audio - 0.6)**2) / (2 * 0.08**2))
    desynced_audio = (np.sin(2 * np.pi * 300 * t_audio) * desynced_env).astype(np.float32)
    
    ev_desynced = agent.analyze((synced_frames, (desynced_audio, sr)))
    print(f"Test 2 (Desynced AI Dubbing) -> Verdict: {ev_desynced.verdict} | Lag: {ev_desynced.telemetry['temporal_lag_ms']} ms | P(Tampered): {ev_desynced.prob_tampered:.2f}")
    assert abs(ev_desynced.telemetry["temporal_lag_ms"]) > 100.0
    assert ev_desynced.prob_tampered > 0.60

    print("-" * 60)
    print("All LipSyncAgent unit assertions passed successfully!")
