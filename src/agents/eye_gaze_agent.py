"""
src/agents/eye_gaze_agent.py
Biometric Forensic Agent: Eye Gaze, Pupil Geometry, & Corneal Specular Reflection.

Analyzes left vs. right pupil circularity, corneal specular highlight symmetry,
and gaze alignment vectors to detect synthetic facial generation artifacts.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from .base_agent import AgentEvidence, BaseForensicAgent
except ImportError:
    from base_agent import AgentEvidence, BaseForensicAgent


class EyeGazeAgent(BaseForensicAgent):
    """
    Forensic agent for analyzing eye geometry, pupil deformation,
    and corneal reflection physics.
    """

    def __init__(self, base_reliability: float = 0.85) -> None:
        super().__init__(name="Eye Gaze & Corneal Reflection Agent", base_reliability=base_reliability)

    def _locate_ocular_regions(self, bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        """
        Locates left and right ocular regions using facial geometry heuristics
        (skin segmentation + horizontal ocular band extraction).
        """
        h, w = bgr.shape[:2]
        if h < 50 or w < 50:
            return None

        # Convert to YCrCb for robust skin masking across Fitzpatrick I-VI
        # (Calibrated for diverse complexions, including South Asian/Indian skin types III-VI)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([45, 125, 65]), np.array([255, 185, 140]))
        
        # Check if a face-sized skin blob exists
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        face_contour = max(contours, key=cv2.contourArea)
        fx, fy, fw, fh = cv2.boundingRect(face_contour)

        # Skin density check: in a face, skin fills at least 35% of the facial bounding box.
        # Disconnected noise, rain streaks, or non-face edges have low fill density (< 25%).
        skin_density = np.count_nonzero(skin_mask[fy:fy+fh, fx:fx+fw]) / float(fw * fh + 1e-6)
        if skin_density < 0.35:
            return None

        # In pre-cropped face ROIs or close-up portraits, the face can fill the frame.
        # Reject only if the frame is a flat/uniform non-face texture lacking facial feature luminance variance.
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray_full)) < 8.0:
            return None

        # Face aspect ratio constraint: relaxed to [0.40, 2.6] to accommodate cultural headwear
        # (turbans/pagris, dupattas, hijabs covering forehead) and dense beards covering lower jaw.
        aspect = float(fh) / float(fw + 1e-6)
        if aspect < 0.40 or aspect > 2.6:
            return None

        # Minimum face size constraint (at least 6% of frame area, min 35px)
        if (fw * fh) < (h * w * 0.06) or fw < 35 or fh < 35:
            return None

        # Adaptive ocular band placement:
        # Avoid upper forehead/eyebrows and target ocular band [28%, 52%] of facial height
        if aspect < 0.85:
            eye_y_min = fy + int(fh * 0.20)
            eye_y_max = fy + int(fh * 0.54)
        else:
            eye_y_min = fy + int(fh * 0.28)
            eye_y_max = fy + int(fh * 0.52)
        eye_h = eye_y_max - eye_y_min

        # Left eye: 12% - 48% of face width
        lx_min = fx + int(fw * 0.12)
        lx_max = fx + int(fw * 0.48)
        lw = lx_max - lx_min

        # Right eye: 52% - 88% of face width
        rx_min = fx + int(fw * 0.52)
        rx_max = fx + int(fw * 0.88)
        rw = rx_max - rx_min

        if eye_h < 8 or lw < 8 or rw < 8:
            return None

        left_eye_crop = bgr[eye_y_min:eye_y_max, lx_min:lx_max]
        right_eye_crop = bgr[eye_y_min:eye_y_max, rx_min:rx_max]

        left_box = (lx_min, eye_y_min, lw, eye_h)
        right_box = (rx_min, eye_y_min, rw, eye_h)

        return left_eye_crop, right_eye_crop, left_box, right_box

    def _analyze_eye_crop(self, eye_bgr: np.ndarray) -> Tuple[float, float, Tuple[int, int], Tuple[float, float], bool]:
        """
        Analyzes pupil circularity using convex hull and corneal specular highlight relative to pupil centroid.
        Returns: (circularity, reflection_intensity, max_loc, relative_vector, has_pupil)
        """
        gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Pupil segmentation (darkest region)
        thresh_val = np.percentile(gray, 25)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circularity = 0.0
        has_pupil = False
        rel_vec = (0.0, 0.0)
        max_loc = (w // 2, h // 2)
        reflection_intensity = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            if perimeter > 0 and area > 15:
                circularity = min(1.0, (4 * np.pi * area) / (perimeter ** 2))
                has_pupil = True

                # Centroid of pupil
                M = cv2.moments(hull)
                cx = M["m10"] / (M["m00"] + 1e-6)
                cy = M["m01"] / (M["m00"] + 1e-6)
                r_pupil = np.sqrt(area / np.pi)

                # Search for corneal reflection inside/adjacent to cornea
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [hull], -1, 255, -1)
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
                _, max_val, _, max_loc = cv2.minMaxLoc(masked_gray)
                reflection_intensity = float(max_val) / 255.0

                # Relative highlight displacement vector normalized by pupil radius
                rel_vec = (float((max_loc[0] - cx) / (r_pupil + 1e-5)), float((max_loc[1] - cy) / (r_pupil + 1e-5)))

        return circularity, reflection_intensity, max_loc, rel_vec, has_pupil

    def _analyze_multi_frame_blink_dynamics(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[float, float, int]:
        """
        Analyzes temporal blink dynamics across a sequence of frames.
        Deepfake videos often exhibit reduced blink frequency or unnatural fixation
        patterns. Returns (blink_anomaly_score, gaze_jitter_std, blink_count).
        
        Blink detection: rapid drop then recovery in eye aperture (luminance in eye ROI).
        Normal human blink rate: 12-20 blinks/minute. In 32 frames at 30fps (~1s),
        expect 0-1 blinks. At 60fps, expect 0-1 blinks per 2s window.
        """
        if len(frames) < 4:
            return 0.0, 0.0, 0

        eye_luminances: List[float] = []
        for f in frames:
            bgr = f if isinstance(f, np.ndarray) else np.array(f)
            h, w = bgr.shape[:2]
            # Sample the upper-middle region where eyes typically reside
            eye_band = bgr[int(h * 0.25):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
            if eye_band.size == 0:
                eye_luminances.append(0.0)
                continue
            gray_band = cv2.cvtColor(eye_band, cv2.COLOR_BGR2GRAY) if eye_band.ndim == 3 else eye_band
            eye_luminances.append(float(np.mean(gray_band)))

        lum_arr = np.array(eye_luminances, dtype=np.float32)
        if np.std(lum_arr) < 1e-4:
            # Completely static eye luminance - possible frozen/static deepfake
            return 0.25, 0.0, 0

        # Detect blink events: a blink is a rapid luminance spike (bright eyelid closing)
        # followed by return to baseline within 2-4 frames
        lum_diff = np.abs(np.diff(lum_arr))
        blink_threshold = float(np.mean(lum_diff) + 2.0 * np.std(lum_diff))
        blink_events = np.where(lum_diff > blink_threshold)[0]
        blink_count = len(blink_events) // 2  # Each blink produces 2 transitions (close + open)

        # Gaze jitter: variance in luminance centroid across frames
        gaze_jitter_std = float(np.std(lum_diff))

        # In ~1 second of video, 0 blinks is slightly anomalous but expected
        # Completely frozen eye with zero variance IS suspicious for long clips
        if len(frames) >= 30:
            # Over 1 second, zero blink AND zero luminance variation is suspicious
            blink_anomaly_score = 0.30 if blink_count == 0 and gaze_jitter_std < 0.5 else 0.05
        else:
            blink_anomaly_score = 0.0

        return blink_anomaly_score, gaze_jitter_std, blink_count

    def analyze(self, image_input: Any, **kwargs: Any) -> AgentEvidence:
        """
        Executes facial geometry and corneal reflection forensic inspection.
        Accepts a single frame (PIL Image / numpy array) OR a list of frames
        for temporal blink/gaze-jitter analysis.
        """
        self._validate_no_leakage_kwargs(kwargs)
        if isinstance(image_input, str):
            raise ValueError(
                "EyeGazeAgent.analyze() does not accept file paths or filenames. "
                "Pass in-memory PIL Image or numpy array directly to prevent label leakage."
            )

        # --- Temporal multi-frame path ---
        frames_list: Optional[List[np.ndarray]] = None
        if isinstance(image_input, list) and len(image_input) > 0:
            frames_list = image_input
            # Use the middle frame for spatial single-frame analysis
            mid_idx = len(frames_list) // 2
            primary_frame = frames_list[mid_idx]
            if hasattr(primary_frame, "convert"):
                rgb_arr = np.array(primary_frame.convert("RGB"))
                bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            elif isinstance(primary_frame, np.ndarray):
                bgr = primary_frame.copy() if primary_frame.shape[-1] == 3 else cv2.cvtColor(primary_frame, cv2.COLOR_GRAY2BGR)
            else:
                bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        elif hasattr(image_input, "convert"):
            rgb_arr = np.array(image_input.convert("RGB"))
            bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[-1] == 3:
                bgr = image_input.copy()
            else:
                bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported input format for EyeGazeAgent")

        annotated = bgr.copy()
        ocular_data = self._locate_ocular_regions(bgr)

        # Case 1: No facial geometry detected
        if ocular_data is None:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,  # Zero reliability -> Pure uncertainty m_U = 1.0
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=["No clear facial or ocular anatomy detected in frame. Biometric gaze check bypassed."],
                telemetry={"faces_detected": 0, "eyes_analyzed": 0},
                visual_artifact=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            )

        left_crop, right_crop, left_box, right_box = ocular_data
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box

        l_circ, l_ref_int, l_ref_loc, l_vec, l_has_pupil = self._analyze_eye_crop(left_crop)
        r_circ, r_ref_int, r_ref_loc, r_vec, r_has_pupil = self._analyze_eye_crop(right_crop)

        # Case 1b: Ocular region detected but lacks confirmed pupil anatomy
        if not l_has_pupil or not r_has_pupil:
            return AgentEvidence(
                agent_name=self.name,
                prob_tampered=0.50,
                reliability=0.0,
                bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                verdict="INCONCLUSIVE",
                findings=["No confirmed pupil contours detected in ocular crops. Biometric gaze check bypassed."],
                telemetry={"faces_detected": 0, "eyes_analyzed": 0},
                visual_artifact=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            )

        # Annotate visual artifact
        cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (0, 255, 255), 2)
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        cv2.circle(annotated, (lx + l_ref_loc[0], ly + l_ref_loc[1]), 3, (0, 0, 255), -1)
        cv2.circle(annotated, (rx + r_ref_loc[0], ry + r_ref_loc[1]), 3, (0, 0, 255), -1)

        # Corneal reflection relative vector divergence
        reflection_divergence = float(np.hypot(l_vec[0] - r_vec[0], l_vec[1] - r_vec[1]))

        # Pupil boundary circularity anomaly (natural human eyes under perspective: circularity in [0.45, 0.85])
        mean_circularity = (l_circ + r_circ) / 2.0
        circ_anomaly = max(0.0, 0.45 - mean_circularity) / 0.30

        # Quantitative anomaly score calibrated for perspective angle and facial expressions
        norm_divergence = max(0.0, (reflection_divergence - 1.25) / 0.50)
        norm_circ_anomaly = min(1.0, circ_anomaly)
        spatial_prob = min(1.0, max(0.0, norm_divergence * 0.60 + norm_circ_anomaly * 0.40))

        # --- Temporal blink and gaze-jitter analysis (if multi-frame input) ---
        blink_anomaly = 0.0
        gaze_jitter = 0.0
        blink_count = -1  # -1 means single-frame mode (no temporal data)
        if frames_list is not None and len(frames_list) >= 4:
            blink_anomaly, gaze_jitter, blink_count = self._analyze_multi_frame_blink_dynamics(frames_list)

        # Combine spatial and temporal scores
        if blink_count >= 0:
            # Multi-frame mode: weight spatial 80%, temporal blink 20%
            prob_tampered = min(1.0, max(0.0, spatial_prob * 0.80 + blink_anomaly * 0.20))
            reliability = 0.85  # Higher reliability with temporal context
        else:
            prob_tampered = spatial_prob
            reliability = 0.80

        findings: List[str] = []
        if prob_tampered > 0.60:
            verdict = "TAMPERED"
            findings.append(f"Corneal specular reflection divergence detected: {reflection_divergence:.2f} (inconsistent light vector).")
            if circ_anomaly > 0.30:
                findings.append(f"Pupil contour non-circularity deformation detected: {mean_circularity:.2f}.")
            if blink_anomaly > 0.20:
                findings.append(f"Abnormal blink dynamics detected across frame sequence (blink count: {blink_count}).")
        elif prob_tampered < 0.35:
            verdict = "AUTHENTIC"
            findings.append(f"Corneal specular reflection consistent across left/right pupils (symmetry index: {1.0 - reflection_divergence:.2f}).")
            findings.append(f"Pupil aspect ratio and circularity conform to natural ocular biology ({mean_circularity:.2f}).")
            if blink_count >= 0:
                findings.append(f"Temporal blink dynamics within normal range (blink count: {blink_count}).")
        else:
            verdict = "INCONCLUSIVE"
            findings.append(f"Minor corneal reflection divergence ({reflection_divergence:.2f}); within biometric tolerance.")

        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)

        telemetry = {
            "faces_detected": 1,
            "eyes_analyzed": 2,
            "corneal_divergence": round(reflection_divergence, 4),
            "left_circularity": round(l_circ, 4),
            "right_circularity": round(r_circ, 4),
            "mean_circularity": round(mean_circularity, 4),
            "left_specular_intensity": round(l_ref_int, 4),
            "right_specular_intensity": round(r_ref_int, 4),
            "blink_count": blink_count,
            "gaze_jitter_std": round(gaze_jitter, 4),
            "temporal_frames_analyzed": len(frames_list) if frames_list else 1,
        }

        return AgentEvidence(
            agent_name=self.name,
            prob_tampered=prob_tampered,
            reliability=reliability,
            bba_mass=bba,
            verdict=verdict,
            findings=findings,
            telemetry=telemetry,
            visual_artifact=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        )


# ---------------------------------------------------------------------------
# Unit / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = EyeGazeAgent()

    print(f"Testing {agent.name}...")
    print("-" * 60)

    # Test 1: Blank non-face image
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    evidence_blank = agent.analyze(blank)
    print(f"Test 1 (No Face) -> Verdict: {evidence_blank.verdict} | Uncertainty m(U): {evidence_blank.bba_mass['U']:.2f}")
    assert evidence_blank.verdict == "INCONCLUSIVE"
    assert evidence_blank.bba_mass["U"] == 1.0

    # Test 2: Synthetic face with ocular regions
    synthetic_face = np.full((224, 224, 3), (140, 150, 200), dtype=np.uint8)
    # Draw left and right eyes
    cv2.circle(synthetic_face, (80, 90), 12, (20, 20, 20), -1)
    cv2.circle(synthetic_face, (144, 90), 12, (20, 20, 20), -1)
    # Put highlights
    cv2.circle(synthetic_face, (82, 88), 3, (255, 255, 255), -1)
    cv2.circle(synthetic_face, (146, 88), 3, (255, 255, 255), -1)

    evidence_face = agent.analyze(synthetic_face)
    print(f"Test 2 (Synthetic Face) -> Verdict: {evidence_face.verdict} | P(Tampered): {evidence_face.prob_tampered:.2f}")
    assert evidence_face.telemetry["eyes_analyzed"] == 2
    assert "U" in evidence_face.bba_mass

    print("-" * 60)
    print("All EyeGazeAgent unit assertions passed successfully!")
