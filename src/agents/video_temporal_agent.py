"""
src/agents/video_temporal_agent.py
Temporal Forensic Agent: Video Optical Flow & Motion Consistency Analysis.

Analyzes frame-to-frame dense optical flow (Gunnar-Farneback), temporal boundary jitter,
and motion vector field warping residuals to detect video deepfakes and face swaps.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from .base_agent import AgentEvidence, BaseForensicAgent
except ImportError:
    from base_agent import AgentEvidence, BaseForensicAgent


class VideoTemporalAgent(BaseForensicAgent):
    """
    Forensic agent for analyzing temporal continuity, inter-frame optical flow,
    and facial boundary warping jitter across video sequences.
    """

    def __init__(self, base_reliability: float = 0.88) -> None:
        super().__init__(name="Video Temporal Consistency Agent", base_reliability=base_reliability)

    def _compute_dense_optical_flow(self, prev_gray: np.ndarray, next_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes dense Farneback optical flow between two consecutive grayscale frames.
        Returns:
            flow: [H, W, 2] motion vector array
            magnitude: [H, W] motion speed
            angle: [H, W] motion direction in degrees [0, 360]
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        return flow, mag, ang

    def _render_flow_hsv(self, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
        """
        Renders optical flow field as an HSV image (Hue=Direction, Value=Speed).
        Returns [H, W, 3] RGB image.
        """
        h, w = mag.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang / 2.0).astype(np.uint8)  # Hue [0, 180]
        hsv[..., 1] = 255                           # Saturation
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = norm_mag.astype(np.uint8)     # Value (Brightness)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    def analyze(self, video_frames_input: Any, **kwargs: Any) -> AgentEvidence:
        """
        Executes video temporal consistency and optical flow forensic analysis.

        Args:
            video_frames_input: List of uint8 BGR/RGB numpy arrays.
        """
        self._validate_no_leakage_kwargs(kwargs)
        if isinstance(video_frames_input, str):
            raise ValueError(
                "VideoTemporalAgent.analyze() does not accept file paths or filenames. "
                "Pass in-memory list of video frames directly to prevent label leakage."
            )

        frames: List[np.ndarray] = []

        # Parse input
        if isinstance(video_frames_input, list):
            frames = video_frames_input
        elif isinstance(video_frames_input, np.ndarray):
            if video_frames_input.ndim == 4:
                frames = [video_frames_input[i] for i in range(video_frames_input.shape[0])]
            else:
                frames = [video_frames_input]

        findings: List[str] = []

        # Case 1: Still image or insufficient frames (< 5 frames) for kinematic optical flow & jerk analysis
        if len(frames) < 5:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,  # Zero reliability -> Pure uncertainty m_U = 1.0
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=[f"Insufficient frames ({len(frames)} frames < 5 required for kinematic motion analysis). Video temporal flow check bypassed."],
                telemetry={"frames_analyzed": len(frames), "is_video": len(frames) > 1},
                visual_artifact=None,
            )

        # Convert frames to grayscale
        gray_frames = []
        for f in frames:
            if f.ndim == 3:
                gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            else:
                gray_frames.append(f)

        flow_variances = []
        flow_magnitudes = []
        flow_angles_std = []
        hsv_visual = None

        for i in range(len(gray_frames) - 1):
            flow, mag, ang = self._compute_dense_optical_flow(gray_frames[i], gray_frames[i + 1])
            flow_magnitudes.append(float(np.mean(mag)))
            flow_variances.append(float(np.var(mag)))

            # Only compute angular divergence on pixels with meaningful motion (mag > 0.6)
            moving_mask = mag > 0.6
            min_moving_pixels = max(400, int(gray_frames[i].size * 0.005))
            if np.sum(moving_mask) > min_moving_pixels:
                # Use circular statistics to avoid 0/360 wrap-around boundary error
                ang_rad = np.deg2rad(ang[moving_mask])
                c_mean = float(np.mean(np.cos(ang_rad)))
                s_mean = float(np.mean(np.sin(ang_rad)))
                R = float(np.hypot(c_mean, s_mean))
                # Circular standard deviation in degrees: sigma_circ = rad2deg(sqrt(-2 * ln(R)))
                circ_std_deg = float(np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(1e-6, R))))))
                flow_angles_std.append(circ_std_deg)
            else:
                flow_angles_std.append(0.0)

            if i == 0:
                hsv_visual = self._render_flow_hsv(mag, ang)

        # Compute inter-frame luminance differences and temporal flicker
        lum_diffs = []
        for i in range(len(gray_frames) - 1):
            diff = np.abs(gray_frames[i + 1].astype(np.float32) - gray_frames[i].astype(np.float32))
            lum_diffs.append(float(np.mean(diff)))
        mean_lum_diff = float(np.mean(lum_diffs)) if lum_diffs else 0.0
        flicker_ratio = float(np.std(lum_diffs) / (mean_lum_diff + 1e-4)) if lum_diffs else 0.0

        # --- Quantitative Kinematic & Temporal Metrics ---
        mean_mag = float(np.mean(flow_magnitudes))
        mean_angle_std = float(np.mean(flow_angles_std))
        mean_spatial_var = float(np.mean(flow_variances))

        # Kinematic acceleration and second-order jerk analysis:
        # v_t: flow_magnitudes
        # a_t = v_{t+1} - v_t (acceleration)
        # j_t = a_{t+1} - a_t = v_{t+2} - 2*v_{t+1} + v_t (discrete jerk)
        # A smooth camera pan (even with accelerating/decelerating pan) has near-zero jerk and 0-1 sign flips.
        # AI/deepfake jitter has large jerk with frequent alternating sign flips.
        n_transitions = len(flow_magnitudes)
        if n_transitions >= 3:
            accelerations = [flow_magnitudes[t + 1] - flow_magnitudes[t] for t in range(n_transitions - 1)]
            jerks = [abs(accelerations[t + 1] - accelerations[t]) for t in range(len(accelerations) - 1)]
            mean_jerk = float(np.mean(jerks))

            # Count alternating sign flips (sawtooth oscillation indicator)
            sign_flips = 0
            for t in range(len(accelerations) - 1):
                if accelerations[t] * accelerations[t + 1] < -1e-5:
                    sign_flips += 1
            flip_ratio = float(sign_flips / max(1, len(accelerations) - 1))

            norm_jerk = mean_jerk / max(mean_mag, 0.4)
            jitter_score = float(min(1.0, norm_jerk * (0.5 + 1.0 * flip_ratio)))
        elif n_transitions == 2:
            acc = abs(flow_magnitudes[1] - flow_magnitudes[0])
            norm_jerk = acc / max(mean_mag, 0.4)
            jitter_score = float(min(1.0, norm_jerk * 0.5))
        else:
            jitter_score = 0.0

        motion_turbulence = float(min(1.0, mean_spatial_var / (mean_mag * 5.0 + 1e-4)))
        angular_chaos = float(min(1.0, max(0.0, (mean_angle_std - 30.0) / 45.0)))

        # Motion regime classification:
        if mean_mag < 0.20:
            # Stationary / static scene (screen capture, desktop presentation, static webcam)
            # Optical flow carries minimal signal; do not trigger false positive jitter alarms
            prob_tampered = 0.10
            reliability = 0.35
        elif mean_mag < 0.60:
            # Subtle/conversational motion (talking head, subtle gestures, slow pan)
            if jitter_score < 0.40 and angular_chaos < 0.30 and flicker_ratio < 0.50:
                prob_tampered = max(0.05, min(0.22, jitter_score * 0.45))
            else:
                prob_tampered = min(0.85, max(0.15, angular_chaos * 0.35 + jitter_score * 0.35 + min(1.0, flicker_ratio) * 0.20))
            reliability = 0.75
        else:
            # Moderate to high motion scene (e.g. camera panning, walking, gestures)
            # Authentic camera motion exhibits high spatial directional coherence (low angular chaos, low jerk)
            # Deepfake face replacements exhibit incoherent localized flow turbulence and inter-frame flicker
            if angular_chaos < 0.25 and jitter_score < 0.30:
                prob_tampered = max(0.05, min(0.28, jitter_score * 0.40 + angular_chaos * 0.30))
            else:
                prob_tampered = min(1.0, max(0.15, angular_chaos * 0.35 + jitter_score * 0.40 + motion_turbulence * 0.15 + min(1.0, flicker_ratio) * 0.10))
            reliability = 0.85

        if prob_tampered > 0.60:
            verdict = "TAMPERED"
            findings.append(f"High-frequency motion vector jerk detected across frame sequence (jitter index: {jitter_score:.2f}).")
            findings.append(f"Inter-frame optical flow discontinuities detected (angular deviation: {mean_angle_std:.1f}°).")
            if flicker_ratio > 0.50:
                findings.append(f"Elevated temporal luminance flicker detected (flicker index: {flicker_ratio:.2f}; characteristic of AI generative synthesis or particle effects).")
        elif prob_tampered < 0.35:
            verdict = "AUTHENTIC"
            findings.append(f"Smooth, continuous optical flow fields across all {len(frames)} analyzed video frames.")
            findings.append("Motion vector gradients consistent with natural camera/subject kinematics.")
        else:
            verdict = "INCONCLUSIVE"
            findings.append(f"Moderate motion vector variance ({jitter_score:.2f}); consistent with complex scene dynamics.")

        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)

        telemetry = {
            "frames_analyzed": len(frames),
            "mean_motion_magnitude": round(mean_mag, 4),
            "motion_jitter_index": round(jitter_score, 4),
            "angular_divergence_deg": round(mean_angle_std, 2),
            "temporal_flicker_ratio": round(flicker_ratio, 4),
            "flow_magnitudes_series": [round(x, 3) for x in flow_magnitudes],
            "lum_diffs_series": [round(x, 3) for x in lum_diffs],
        }

        return AgentEvidence(
            agent_name=self.name,
            prob_tampered=prob_tampered,
            reliability=reliability,
            bba_mass=bba,
            verdict=verdict,
            findings=findings,
            telemetry=telemetry,
            visual_artifact=hsv_visual,
        )


# ---------------------------------------------------------------------------
# Unit / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = VideoTemporalAgent()

    print(f"Testing {agent.name}...")
    print("-" * 60)

    # Test 1: Single frame -> should gracefully return inconclusive with U=1.0
    single = [np.zeros((224, 224, 3), dtype=np.uint8)]
    ev_single = agent.analyze(single)
    print(f"Test 1 (Single Frame) -> Verdict: {ev_single.verdict} | Uncertainty m(U): {ev_single.bba_mass['U']:.2f}")
    assert ev_single.verdict == "INCONCLUSIVE"
    assert ev_single.bba_mass["U"] == 1.0

    # Test 2: Smooth translational video (authentic camera pan)
    pan_frames = []
    base_canvas = np.random.randint(50, 200, (260, 260, 3), dtype=np.uint8)
    for step in range(8):
        crop = base_canvas[step * 2:step * 2 + 224, step * 2:step * 2 + 224]
        pan_frames.append(crop)

    ev_pan = agent.analyze(pan_frames)
    print(f"Test 2 (Smooth Pan Video) -> Verdict: {ev_pan.verdict} | P(Tampered): {ev_pan.prob_tampered:.2f}")
    assert ev_pan.visual_artifact is not None
    assert ev_pan.telemetry["frames_analyzed"] == 8

    # Test 3: Jittery / deepfake flickering video sequence
    flicker_frames = []
    for step in range(8):
        noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) if step % 2 == 1 else pan_frames[step]
        flicker_frames.append(noise)

    ev_flicker = agent.analyze(flicker_frames)
    print(f"Test 3 (Flickering Deepfake) -> Verdict: {ev_flicker.verdict} | P(Tampered): {ev_flicker.prob_tampered:.2f}")
    assert ev_flicker.prob_tampered > 0.50

    print("-" * 60)
    print("All VideoTemporalAgent unit assertions passed successfully!")
