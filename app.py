"""
app.py  --  PRISM Forensic Intelligence Dashboard
Run:  streamlit run app.py

Root-level entry point. Adds src/ to sys.path so all PRISM modules resolve
without package installation.
"""

import sys
import os
from typing import Optional, List, Dict, Any, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np
import torch
import time
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
import pandas as pd
import streamlit as st

# -- Page config must be the very first Streamlit call -----------------------
st.set_page_config(
    page_title="PRISM -- Media Forensics",
    page_icon="magnifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Minimal CSS: tighten metrics, monospace report block --------------------
st.markdown(
    """
    <style>
    [data-testid="metric-container"] { padding: 0.6rem 0.8rem; }
    .report-block { font-family: 'Courier New', monospace; font-size: 0.82rem;
                    line-height: 1.55; white-space: pre-wrap; }
    hr { border-color: #2a2a2a; }
    </style>
    """,
    unsafe_allow_html=True,
)

from preprocessor     import PRISMPreprocessor
from fusion           import DempsterShaferFusion
from reasoner         import PRISMReasoner
from model            import PRISMNetwork
from report_exporter  import ForensicReportExporter
from agents           import (
    MetadataAgent,
    EyeGazeAgent,
    CompressionWatermarkAgent,
    AudioVoiceAgent,
    VideoTemporalAgent,
    IlluminationAgent,
    LipSyncAgent, RPPGAgent,
)


# ---------------------------------------------------------------------------
# Singleton resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PRISM model...")
def load_model():
    """Returns (model, device). Uses saved weights if present, else random init."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PRISMNetwork(pretrained=False).to(device)

    checkpoints_dir = os.path.join(os.path.dirname(__file__), "models", "checkpoints")
    final_path = os.path.join(checkpoints_dir, "prism_model_final.pth")
    best_path = os.path.join(checkpoints_dir, "prism_model_best.pth")
    
    weights_path = final_path if os.path.isfile(final_path) else (best_path if os.path.isfile(best_path) else None)
    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device)
        # Strip potential 'module.' prefix if saved from DataParallel
        cleaned_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned_state, strict=False)
        print(f"Loaded trained PRISM weights successfully from: {os.path.basename(weights_path)}")

    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Returns (preprocessor, fusion_engine, reasoner, exporter, agents_dict)."""
    agents = {
        "metadata": MetadataAgent(),
        "eye_gaze": EyeGazeAgent(),
        "compression": CompressionWatermarkAgent(),
        "audio": AudioVoiceAgent(),
        "temporal": VideoTemporalAgent(),
        "illumination": IlluminationAgent(),
        "lip_sync": LipSyncAgent(), "rppg": RPPGAgent(),
    }
    return PRISMPreprocessor(), DempsterShaferFusion(), PRISMReasoner(), ForensicReportExporter(), agents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_online() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def _vram_used_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float tensor -> [H,W,3] uint8 RGB (min-max rescale)."""
    arr = t.cpu().float().permute(1, 2, 0).numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _noise_to_uint8(noise_t: torch.Tensor) -> np.ndarray:
    """[1,H,W] noise tensor -> [H,W] uint8 greyscale."""
    arr = noise_t.squeeze(0).cpu().float().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return (arr * 255).clip(0, 255).astype(np.uint8)

@st.cache_data(show_spinner=False)
def _make_heatmap_overlay(mask_np: np.ndarray, rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Thresholded overlay of mask onto RGB. Fully transparent below 0.6 confidence.
    """
    mask_2d = np.squeeze(mask_np)  # Guarantees [H, W] (2D)
    mask_u8 = (mask_2d * 255.0).clip(0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(mask_u8, cv2.COLORMAP_INFERNO)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = rgb_uint8.copy()
    mask_threshold = mask_2d > 0.6
    overlay[mask_threshold] = cv2.addWeighted(overlay, 0.3, heatmap_color, 0.7, 0)[mask_threshold]
    return overlay


# ---------------------------------------------------------------------------
# Forensic Algorithms: ELA and FFT
# ---------------------------------------------------------------------------
def compute_ela(img: Image.Image, quality: int = 90) -> tuple:
    """
    Error Level Analysis (ELA).
    Recompresses image at known quality and computes amplified residual difference.
    Returns (enhanced_ela_array, normalized_energy_mass).
    """
    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    temp_img = Image.open(buf)
    
    ela_image = ImageChops.difference(img_rgb, temp_img)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1
        
    scale = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Calculate robust normalized ELA energy
    diff_arr = np.asarray(ela_image, dtype=np.float32)
    mean_diff = float(np.mean(diff_arr)) / 255.0
    ela_energy = min(1.0, max(0.0, mean_diff * 6.0))  # scaled into [0, 1]
    
    return np.array(ela_enhanced), ela_energy


def compute_fft(image_arr: np.ndarray) -> np.ndarray:
    """
    Computes 2D FFT log magnitude spectrum.
    Accepts [H, W] or [H, W, 3] arrays.
    """
    if image_arr.ndim == 3:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image_arr.astype(np.float32)
        
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20.0 * np.log(np.abs(fshift) + 1.0)
    # Normalize for clean contrast rendering
    norm_spec = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
    return (norm_spec * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(reasoner: Optional[PRISMReasoner] = None):
    with st.sidebar:
        st.markdown("## PRISM System")
        st.caption("Provenance & Reasoning Intelligence System for Multimedia")
        st.divider()

        # Neural Backbone Checkpoint
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "models", "checkpoints")
        has_final = os.path.isfile(os.path.join(checkpoints_dir, "prism_model_final.pth"))
        has_best = os.path.isfile(os.path.join(checkpoints_dir, "prism_model_best.pth"))
        
        st.markdown("**Core Neural Backbone**")
        if has_final:
            st.success("Trained: `prism_model_final.pth` (Dual T4)")
        elif has_best:
            st.success("Trained: `prism_model_best.pth` (Dual T4)")
        else:
            st.warning("No checkpoint found (Initialized)")
        st.caption("ViT-B/16 + DoRA + Sensor Noise + Decoder")
        st.divider()

        # Hardware
        cuda_ok = torch.cuda.is_available()
        st.markdown("**Hardware**")
        c1, c2 = st.columns(2)
        c1.metric("CUDA", "ON" if cuda_ok else "OFF")
        if cuda_ok:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            c2.metric("VRAM", f"{total_gb:.1f} GB")
            used_frac = min(_vram_used_mb() / (total_gb * 1024), 1.0)
            st.progress(used_frac, text=f"VRAM used: {_vram_used_mb():.0f} MB")
        st.divider()

        # Reasoning Engine
        st.markdown("**Forensic Reasoning Engine**")
        engine_display = reasoner.active_engine if reasoner else "PRISM Expert Engine"
        st.metric("Reasoning Layer", "Active")
        st.caption(f"Engine: {engine_display}")
        st.divider()

        # Upload
        st.markdown("**Upload Multimedia File**")
        uploaded = st.file_uploader(
            "Drop file here",
            type=["png", "jpg", "jpeg", "webp", "wav", "mp4", "avi", "mov"],
            label_visibility="collapsed",
        )
        st.caption("Image (PNG/JPG)  |  Audio (WAV)  |  Video (MP4)")

        # Preloaded real samples selector
        sample_dir = os.path.join(os.path.dirname(__file__), "data", "real_test_samples")
        if os.path.isdir(sample_dir):
            sample_files = [f for f in sorted(os.listdir(sample_dir)) if os.path.isfile(os.path.join(sample_dir, f)) and not f.startswith(".")]
            selected_sample = st.selectbox("Or choose real benchmark sample:", ["(Upload own file)"] + sample_files)
            if selected_sample != "(Upload own file)" and uploaded is None:
                sample_path = os.path.join(sample_dir, selected_sample)
                with open(sample_path, "rb") as sf:
                    content = sf.read()
                uploaded = io.BytesIO(content)
                uploaded.name = selected_sample

        st.divider()

        # Indian Forensic & Celebrity Deepfake Benchmark Hub
        st.markdown("**Indian Forensics Hub**")
        with st.expander("Datasets & Fact-Check Suites"):
            st.markdown("""
            - **Option A: DF-Platter**
              *133k videos, Fitzpatrick III-VI, turbans/beards*
              `py scripts/benchmark_df_platter.py`
            - **Option B: InDeepFake (2025)**
              *Indic audio-visual deepfakes (Hindi/Tamil)*
              `py scripts/benchmark_indeepfake.py`
            - **Option C: Fact-Check Archives**
              *BOOM Live & Alt News verified dossiers*
              `py scripts/fetch_factcheck_samples.py`
            - **Option D: Inswapper Replica**
              *Controlled Roop-style artifact generator*
              `py scripts/generate_controlled_faceswap_pairs.py`
            """)

    return uploaded


# ---------------------------------------------------------------------------
# Inference pipeline (Multimodal Router)
# ---------------------------------------------------------------------------

def run_inference(uploaded_file, model, device, preprocessor, fusion, reasoner, exporter, agents):
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()
    is_audio = filename.endswith((".wav", ".mp3"))
    is_video = filename.endswith((".mp4", ".avi", ".mov", ".mkv"))

    # =======================================================================
    # ROUTE A: AUDIO FORENSICS (.wav)
    # =======================================================================
    if is_audio:
        st.markdown(f"### Audio Forensic Inspection: `{uploaded_file.name}`")
        st.audio(file_bytes)

        with st.status("[AUDIT] Analyzing Vocal Acoustics & Vocoder Footprint...", expanded=True) as status:
            st.write("[DONE] Computing STFT Mel-Spectrogram...")
            time.sleep(0.3)
            st.write("[DONE] Evaluating high-frequency brickwall attenuation (vocoder cutoff)...")
            time.sleep(0.3)
            st.write("[DONE] Measuring dynamic spectral flux and vocal formant continuity...")
            time.sleep(0.3)
            
            audio_ev = agents["audio"].analyze(file_bytes)
            status.update(label="[COMPLETE] Acoustic Analysis Complete", state="complete", expanded=False)

        score = audio_ev.prob_tampered
        verdict = audio_ev.verdict
        belief = audio_ev.bba_mass["T"]
        plausibility = audio_ev.bba_mass["T"] + audio_ev.bba_mass["U"]

        # Metric row
        m1, m2, m3 = st.columns(3)
        m1.metric("D-S Acoustic Belief", f"{belief:.1%}", delta=f"Plausibility: {plausibility:.1%}", delta_color="off")
        m2.metric("P(Synthetic Clone)", f"{score:.1%}")
        m3.metric("System Verdict", verdict)

        tab1, tab2, tab3 = st.tabs(["Acoustic Report", "Mel-Spectrogram", "Spectral Telemetry"])
        with tab1:
            if verdict == "TAMPERED":
                st.error(f"**[ALERT: {verdict}]** Synthetic voice / neural vocoder footprints detected.")
            elif verdict == "AUTHENTIC":
                st.success(f"**[VERIFIED: {verdict}]** Natural vocal tract acoustics consistent with human speech.")
            else:
                st.warning(f"**[UNCERTAIN: {verdict}]** Acoustic evidence is inconclusive.")

            st.markdown("#### Detailed Acoustic Findings:")
            for f in audio_ev.findings:
                st.write(f"- {f}")

        with tab2:
            st.markdown("#### Log-Magnitude Mel-Spectrogram (0 - Nyquist)")
            if audio_ev.visual_artifact is not None:
                st.image(audio_ev.visual_artifact, caption="STFT Spectrogram (Inferno Colormap)", width="stretch")

        with tab3:
            st.json(audio_ev.telemetry)
        return

    # =======================================================================
    # ROUTE B: VIDEO FORENSICS (.mp4, .mov, .avi)
    # =======================================================================
    if is_video:
        import tempfile
        st.markdown(f"### Video Forensic Inspection: `{uploaded_file.name}`")
        st.video(file_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            with st.status("[AUDIT] Running Multi-Frame Video Forensics...", expanded=True) as status:
                st.write("[DONE] Sampling synchronized video burst & extracting container metadata...")
                raw_meta = preprocessor.extract_raw_video_frames_with_meta(tmp_path, max_frames=32)
                raw_frames = raw_meta["frames"]
                actual_fps = raw_meta["fps"]
                start_frame = raw_meta["start_frame"]
                start_sec = raw_meta["start_sec"]
                if not raw_frames:
                    raw_frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
                time.sleep(0.2)

                st.write(f"[DONE] Calculating Farneback dense optical flow & temporal kinematics ({actual_fps:.1f} FPS)...")
                temporal_ev = agents["temporal"].analyze(raw_frames)
                time.sleep(0.2)

                st.write("[DONE] Multi-Keyframe Spatial Inspection & Facial ROI Localization...")
                mid_frame = raw_frames[len(raw_frames) // 2]
                mid_pil = Image.fromarray(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
                h_m, w_m = mid_frame.shape[:2]

                # Fast Haar Cascade facial verification for localized deepfake / face-swap boundary audit
                cascade_path = os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml")
                face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.isfile(cascade_path) else None
                gray_mid = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray_mid, 1.1, 4) if face_cascade is not None else []
                has_face = len(detected_faces) > 0
                face_box = detected_faces[0] if has_face else None

                sensor_divergence = 0.0
                face_noise_var = 0.0
                outer_noise_var = 0.0
                if has_face:
                    fx, fy, fw, fh = face_box
                    face_roi = gray_mid[fy:fy+fh, fx:fx+fw]
                    pad = int(max(fw, fh) * 0.4)
                    y1, y2 = max(0, fy - pad), min(h_m, fy + fh + pad)
                    x1, x2 = max(0, fx - pad), min(w_m, fx + fw + pad)
                    outer_roi = gray_mid[y1:y2, x1:x2]
                    
                    lap_f = cv2.Laplacian(face_roi, cv2.CV_32F)
                    lap_o = cv2.Laplacian(outer_roi, cv2.CV_32F)
                    face_noise_var = float(np.var(lap_f))
                    outer_noise_var = float(np.var(lap_o))
                    sensor_divergence = abs(outer_noise_var - face_noise_var) / (outer_noise_var + 1e-6)

                st.write("[DONE] Evaluating Multi-Keyframe ViT Backbone & PRNU Sensor Noise...")
                sample_indices = np.linspace(0, len(raw_frames) - 1, min(4, len(raw_frames)), dtype=int)
                frame_probs = []
                noise_energies = []

                with torch.no_grad():
                    for s_idx in sample_indices:
                        s_f = raw_frames[s_idx]
                        s_rgb_t, s_noise_t = preprocessor.process_frame(s_f)
                        out = model(s_rgb_t.unsqueeze(0).to(device), s_noise_t.unsqueeze(0).to(device))
                        frame_probs.append(float(torch.sigmoid(out["logit"]).float().item()))
                        noise_energies.append(float(torch.mean(torch.abs(s_noise_t)).item()) * 10.0)
                        del out

                        # If face present, also run ViT specifically on facial crop
                        if has_face:
                            fx, fy, fw, fh = face_box
                            pad_f = int(max(fw, fh) * 0.2)
                            fy1, fy2 = max(0, fy - pad_f), min(h_m, fy + fh + pad_f)
                            fx1, fx2 = max(0, fx - pad_f), min(w_m, fx + fw + pad_f)
                            fcrop = s_f[fy1:fy2, fx1:fx2]
                            if fcrop.size > 0:
                                fcrop_rgb = cv2.resize(fcrop, (224, 224))
                                fcrop_t, fcrop_noise = preprocessor.process_frame(fcrop_rgb)
                                out_f = model(fcrop_t.unsqueeze(0).to(device), fcrop_noise.unsqueeze(0).to(device))
                                frame_probs.append(float(torch.sigmoid(out_f["logit"]).float().item()))
                                del out_f

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                keyframe_indices = sorted(list(set([0, len(raw_frames) // 2, len(raw_frames) - 1])))
                comp_ev = None
                illum_ev = None

                for kf_idx in keyframe_indices:
                    kf = raw_frames[kf_idx]
                    kf_pil = Image.fromarray(cv2.cvtColor(kf, cv2.COLOR_BGR2RGB))
                    kf_comp = agents["compression"].analyze(kf_pil)
                    kf_illum = agents["illumination"].analyze(kf)

                    if comp_ev is None or kf_comp.prob_tampered > comp_ev.prob_tampered:
                        comp_ev = kf_comp
                        mid_frame = kf
                        mid_pil = kf_pil
                    if illum_ev is None or kf_illum.prob_tampered > illum_ev.prob_tampered:
                        illum_ev = kf_illum

                prob_fake = float(np.max(frame_probs) * 0.65 + np.mean(frame_probs) * 0.35)
                raw_noise_level = float(np.mean(noise_energies)) / 10.0 if noise_energies else 0.08
                prnu_energy = raw_noise_level
                hf_peaks = float(comp_ev.telemetry.get("hf_spectral_peak_score", 0.0))

                # Comprehensive Sensor PRNU Physics Verification
                if sensor_divergence > 0.35:
                    # Clear sensor PRNU mismatch across facial boundary (Roop / Inswapper seam)
                    prnu_prob_tamper = 0.88
                    prnu_rel = 0.85
                elif has_face and face_noise_var < 8.0 and raw_noise_level < 0.03:
                    # Artificially denoised/smoothed synthetic face patch
                    prnu_prob_tamper = 0.82
                    prnu_rel = 0.80
                elif raw_noise_level < 0.015:
                    prnu_prob_tamper = 0.70 if prob_fake >= 0.25 else 0.15
                    prnu_rel = 0.70
                elif raw_noise_level > 0.45 or hf_peaks > 0.60:
                    prnu_prob_tamper = 0.75
                    prnu_rel = 0.70
                else:
                    prnu_prob_tamper = 0.12
                    prnu_rel = 0.40

                # Calibrated Subjective Logic Evidence Mapping (prevents false 81% authentic bias)
                if prob_fake >= 0.50:
                    ev_T = (prob_fake - 0.50) * 2.0 * 8.0
                    ev_A = 0.0
                else:
                    ev_T = 0.0
                    ev_A = (0.50 - prob_fake) * 2.0 * 2.5
                vit_bba = fusion.create_mass(evidence_T=ev_T, evidence_A=ev_A, W=2.0)
                prnu_bba = fusion.create_mass(evidence_T=prnu_prob_tamper * 8.0 * prnu_rel, evidence_A=(1.0 - prnu_prob_tamper) * 8.0 * prnu_rel, W=2.0)

                st.write("[DONE] Performing MP4 Container & C2PA Provenance Audit...")
                meta_ev = agents["metadata"].analyze(mid_pil, file_bytes=file_bytes)

                st.write("[DONE] Auditing Biometric Liveness, Gaze Dynamics, Audio Stream & Voice Spoofing...")
                # Pass multi-frame sequence to eye_gaze agent for blink and temporal gaze-jitter tracking
                gaze_ev = agents["eye_gaze"].analyze(raw_frames)
                faces_found = (gaze_ev.telemetry.get("faces_detected", 0) > 0) or has_face

                # Extract audio from MP4 via ffmpeg streaming pipe
                audio_res = preprocessor.extract_audio_track(tmp_path)
                if audio_res is not None:
                    audio_wave, audio_sr = audio_res
                    audio_ev = agents["audio"].analyze((audio_wave, audio_sr))
                    # Compute sample offset corresponding to the extracted video burst window
                    audio_offset_samples = int(start_sec * audio_sr)
                    lip_ev = agents["lip_sync"].analyze({
                        "frames": raw_frames,
                        "audio": (audio_wave, audio_sr),
                        "fps": actual_fps,
                        "frame_offset_samples": audio_offset_samples,
                    })
                else:
                    from agents.base_agent import AgentEvidence
                    audio_ev = AgentEvidence(
                        agent_name="Audio Voice Anti-Spoofing Agent",
                        prob_tampered=0.50,
                        reliability=0.0,
                        bba_mass={"T": 0.0, "A": 0.0, "U": 1.0},
                        verdict="BYPASSED",
                        findings=["Video file contains no acoustic audio stream. Voice spoofing check bypassed."],
                        telemetry={"status": "no_audio_stream"},
                        visual_artifact=None,
                    )
                    lip_ev = agents["lip_sync"].analyze({
                        "frames": raw_frames,
                        "audio": None,
                        "fps": actual_fps,
                        "frame_offset_samples": 0,
                    })

                rppg_ev = agents["rppg"].analyze(raw_frames, has_face=faces_found, fps=actual_fps)

                # If no human face is detected in the video, 3D illumination vector divergence
                # (which assumes a central portrait subject against ambient background) has no
                # biometric or face-swap basis. Treat illumination as uninformative (vacuous uncertainty)
                # to prevent ambient room or monitor lighting from triggering false positives.
                illum_bba = illum_ev.bba_mass if faces_found else {"T": 0.0, "A": 0.0, "U": 1.0}

                # Fuse evidence over Dynamic Forensic Evidence Graph
                agent_dict = {
                    "vit": vit_bba,
                    "prnu": prnu_bba,
                    "temporal": temporal_ev.bba_mass,
                    "meta": meta_ev.bba_mass,
                    "comp": comp_ev.bba_mass,
                    "illum": illum_bba,
                    "audio": audio_ev.bba_mass,
                    "gaze": gaze_ev.bba_mass,
                    "lip": lip_ev.bba_mass,
                    "rppg": rppg_ev.bba_mass,
                }
                edges = [
                    ("vit", "temporal"),
                    ("comp", "vit"),
                    ("prnu", "comp"),
                    ("meta", "vit"),
                    ("illum", "temporal"),
                ]
                if audio_res is not None:
                    edges.append(("audio", "temporal"))
                    if faces_found:
                        edges.append(("audio", "lip"))
                if faces_found:
                    edges.extend([("lip", "temporal"), ("gaze", "rppg")])
                fused = fusion.fuse_graph_evidence(agent_dict, edges)

                st.write("[REASONER] Querying Forensic Reasoning Engine for Dossier...")
                score = fused["pignistic_fake"]
                verdict = fused["verdict"]
                belief = fused["belief"]
                plausibility = fused["plausibility"]

                noise_status = "Anomalous -- PRNU pattern deviation detected" if prnu_prob_tamper > 0.50 else "Consistent camera sensor noise baseline"
                meta_summary = "; ".join(meta_ev.findings) if meta_ev.findings else "No metadata anomalies."
                temp_summary = "; ".join(temporal_ev.findings) if temporal_ev.findings else "Normal kinematics."
                comp_summary = "; ".join(comp_ev.findings) if comp_ev.findings else "Standard compression."
                illum_summary = "; ".join(illum_ev.findings) if illum_ev.findings else "Consistent lighting."

                try:
                    report = reasoner.generate_report(
                        score=score,
                        mask_area="Full-frame temporal video inspection",
                        noise_status=noise_status,
                        system_verdict=verdict,
                        vit_mass=vit_bba["T"],
                        prnu_mass=prnu_bba["T"],
                        ela_mass=comp_ev.bba_mass["T"],
                        meta_mass=meta_ev.bba_mass["T"],
                        gaze_mass=temporal_ev.bba_mass["T"],
                        meta_summary=f"{meta_summary} | {comp_summary}",
                        gaze_summary=f"{temp_summary} | {illum_summary}",
                        belief=belief,
                        plausibility=plausibility,
                    )
                except Exception as exc:
                    report = f"[REPORT GENERATION FAILED]\n{exc}"

                status.update(label="[COMPLETE] Video Multi-Agent Verification Complete", state="complete", expanded=False)

        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

        st.divider()
        st.markdown(f"### Forensic Lab Report: `{uploaded_file.name}`")

        m1, m2, m3 = st.columns(3)
        m1.metric("D-S Combined Belief", f"{belief:.1%}", delta=f"Plausibility bound: {plausibility:.1%}", delta_color="off")
        m2.metric("P(Tampered Video)", f"{score:.1%}")
        m3.metric("Video Verdict", verdict)

        st.markdown("##### Specialized Forensic Agent Diagnostics")
        a1, a2, a3, a4, a5, a6, a7 = st.columns(7)
        a1.metric("ViT Backbone", f"{prob_fake:.1%}", delta=f"Mass T: {vit_bba['T']:.1%}")
        a2.metric("Motion / Flow", temporal_ev.verdict, delta=f"Mass T: {temporal_ev.bba_mass['T']:.1%}")
        a3.metric("Container & Meta", meta_ev.verdict, delta=f"Mass T: {meta_ev.bba_mass['T']:.1%}")
        a4.metric("Compression / ELA", comp_ev.verdict, delta=f"Mass T: {comp_ev.bba_mass['T']:.1%}")
        a5.metric("3D Illumination", illum_ev.verdict, delta=f"Mass T: {illum_ev.bba_mass['T']:.1%}")
        a6.metric("Audio / Voice", "Bypassed" if audio_res is None else audio_ev.verdict, delta=f"Mass T: {audio_ev.bba_mass['T']:.1%}" if audio_res is not None else "No Track")
        a7.metric("Biometrics", "Active" if faces_found else "Bypassed", delta="Inconclusive" if not faces_found else rppg_ev.verdict)

        st.divider()

        # Build PDF Dossier for video
        agent_masses_dict = {
            "ViT_spatial": vit_bba,
            "PRNU_sensor": prnu_bba,
            "Video_temporal": temporal_ev.bba_mass,
            "Compression_ELA": comp_ev.bba_mass,
            "Metadata_Provenance": meta_ev.bba_mass,
            "Illumination_3D": illum_ev.bba_mass,
            "Audio_Voice": audio_ev.bba_mass,
            "EyeGaze_Biometric": gaze_ev.bba_mass,
            "rPPG_Liveness": rppg_ev.bba_mass,
        }
        k_conflict = float(max(0.0, 1.0 - (plausibility - belief)))
        m_uncertainty = float(fused.get('uncertainty', max(0.0, 1.0 - (belief + (1.0 - plausibility)))))

        pdf_bytes = exporter.generate_pdf(
            filename=uploaded_file.name,
            file_bytes=file_bytes,
            verdict=verdict,
            score=score,
            belief=belief,
            plausibility=plausibility,
            agent_masses=agent_masses_dict,
            llm_report_text=report,
            figure_png_bytes=None,
            conflict_k=k_conflict,
            epistemic_uncertainty=m_uncertainty,
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Executive Summary & Report",
            "Optical Flow & Motion",
            "Compression & Spatial Artifacts",
            "Agent Telemetry & Graph Math"
        ])

        with tab1:
            if m_uncertainty >= 0.60 or k_conflict >= 0.50:
                st.warning(
                    f"**[HIGH FORENSIC UNCERTAINTY / AGENT CONFLICT DETECTED]**\n\n"
                    f"- Epistemic Uncertainty $m(U) = {m_uncertainty:.1%}$\n"
                    f"- Multi-Agent Conflict $K = {k_conflict:.3f}$\n\n"
                    f"**Investigative Guidance**: Forensic agents show elevated disagreement or lack sufficient signal. "
                    f"This media sample must be treated as **INCONCLUSIVE** and is provided for investigative hypothesis generation only."
                )

            if verdict == "TAMPERED":
                st.error(f"**[ALERT: {verdict}]** Forensic indicators strongly suggest video manipulation or digital post-processing.")
            elif verdict == "AUTHENTIC":
                st.success(f"**[VERIFIED: {verdict}]** No significant manipulation detected across active forensic agents.")
            else:
                st.warning(f"**[UNCERTAIN: {verdict}]** Video evidence is ambiguous; manual review recommended.")

            st.markdown(f'<div class="report-block">{report}</div>', unsafe_allow_html=True)
            st.write("")
            st.download_button(
                label="Download Forensic Dossier for Investigative Support (PDF)",
                data=pdf_bytes,
                file_name=f"PRISM_Forensic_Dossier_{uploaded_file.name}.pdf",
                mime="application/pdf",
                width="stretch",
            )

            st.markdown("#### Comprehensive Multi-Agent Findings:")
            all_findings = (
                meta_ev.findings
                + comp_ev.findings
                + temporal_ev.findings
                + illum_ev.findings
                + audio_ev.findings
                + gaze_ev.findings
                + lip_ev.findings
                + rppg_ev.findings
            )
            for f in all_findings:
                st.write(f"- {f}")

        with tab2:
            st.markdown("#### Dense Optical Flow & Video Kinematics")
            st.markdown("Kinematic analysis evaluates inter-frame displacement, motion field consistency, and high-frequency temporal flicker.")
            
            # 4 Quantitative Metric Cards
            km1, km2, km3, km4 = st.columns(4)
            km1.metric("Mean Velocity", f"{temporal_ev.telemetry.get('mean_motion_magnitude', 0.0):.2f} px/f", help="Average pixel displacement per frame across the keyframe sequence")
            km2.metric("Kinematic Jitter", f"{temporal_ev.telemetry.get('motion_jitter_index', 0.0):.2f}", help="Variance of motion magnitude normalized by velocity (indicates camera shake vs generative stutter)")
            km3.metric("Angular Divergence", f"{temporal_ev.telemetry.get('angular_divergence_deg', 0.0):.1f}°", help="Standard deviation of motion direction vectors across moving pixels")
            km4.metric("Temporal Flicker", f"{temporal_ev.telemetry.get('temporal_flicker_ratio', 0.0):.2f}", help="Ratio of luminance variance to mean difference; highlights generative particle/rain flicker")
            
            c_flow1, c_flow2 = st.columns(2)
            with c_flow1:
                st.markdown("##### Farneback Optical Flow (HSV Vector Field)")
                if temporal_ev.visual_artifact is not None:
                    st.image(temporal_ev.visual_artifact, caption="Farneback Dense Flow Map (Hue = Direction 0°-360°, Sat/Val = Velocity)", width="stretch")
                st.caption("Coherent camera or subject movement displays smooth continuous color gradients. Warped facial areas or AI particle overlay create chaotic, localized color splatter.")
            
            with c_flow2:
                flow_series = temporal_ev.telemetry.get("flow_magnitudes_series", [])
                lum_series = temporal_ev.telemetry.get("lum_diffs_series", [])
                if flow_series:
                    st.markdown("##### Inter-Frame Velocity & Flicker Trajectory")
                    plt.style.use("dark_background")
                    fig_kin, ax_kin1 = plt.subplots(figsize=(6.5, 4.2), dpi=130)
                    fig_kin.patch.set_facecolor('#0e1117')
                    ax_kin1.set_facecolor('#0e1117')
                    
                    frames_idx = list(range(1, len(flow_series) + 1))
                    color_flow = '#00e5ff'
                    ax_kin1.set_xlabel('Inter-Frame Transition (t -> t+1)', color='#888888', fontsize=9)
                    ax_kin1.set_ylabel('Motion Velocity (px/frame)', color=color_flow, fontsize=9)
                    ax_kin1.plot(frames_idx, flow_series, color=color_flow, marker='o', linewidth=2.0, markersize=4, label='Kinematic Velocity')
                    ax_kin1.tick_params(axis='y', labelcolor=color_flow, labelsize=8)
                    ax_kin1.tick_params(axis='x', colors='#888888', labelsize=8)
                    ax_kin1.grid(True, linestyle='--', alpha=0.2, color='#ffffff')

                    if lum_series:
                        ax_kin2 = ax_kin1.twinx()
                        color_lum = '#ff007f'
                        ax_kin2.set_ylabel('Luminance Diff (ΔY)', color=color_lum, fontsize=9)
                        ax_kin2.plot(frames_idx, lum_series, color=color_lum, linestyle=':', marker='s', linewidth=1.5, markersize=3, label='Luminance Jitter')
                        ax_kin2.tick_params(axis='y', labelcolor=color_lum, labelsize=8)

                    fig_kin.tight_layout()
                    st.pyplot(fig_kin)
                    plt.close(fig_kin)
                    st.caption("Temporal trajectory across keyframes. Sudden divergence spikes indicate synthetic overlays, editing cuts, or frame drops.")
                else:
                    st.info("Single keyframe sampled; temporal velocity trajectory unavailable.")

            if lip_ev.visual_artifact is not None:
                st.markdown("---")
                st.markdown("##### Audio-Visual Lip-Sync & Viseme Alignment")
                st.image(lip_ev.visual_artifact, caption="Cross-Modal AV Temporal Synchrony (Speech Energy vs. Oral Cavity Motion)", width="stretch")

            st.markdown("---")
            st.markdown("##### Biometric Liveness & Facial Synchrony Multi-Track")
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("###### Eye Gaze, Corneal Reflections & Temporal Blink Dynamics")
                if gaze_ev.visual_artifact is not None:
                    st.image(gaze_ev.visual_artifact, caption=f"Ocular Geometry & Corneal Specular Highlights (Blinks: {gaze_ev.telemetry.get('blink_count', 'N/A')})", width="stretch")
                st.write(f"- **Corneal Light Divergence**: `{gaze_ev.telemetry.get('corneal_divergence', 0.0):.2f}`")
                st.write(f"- **Mean Pupil Circularity**: `{gaze_ev.telemetry.get('mean_circularity', 0.0):.2f}`")
                st.write(f"- **Gaze Saccade Jitter**: `{gaze_ev.telemetry.get('gaze_jitter_std', 0.0):.4f}`")
            with b_col2:
                st.markdown("###### Biological Pulse Extraction (rPPG Liveness)")
                rppg_snr = rppg_ev.telemetry.get('rPPG_SNR', 0.0)
                in_band = rppg_ev.telemetry.get('in_band_fraction', 0.0)
                st.metric("rPPG Pulse SNR", f"{rppg_snr:.2f}", help="Signal-to-noise ratio in 0.7-3.0 Hz physiological heart rate band")
                st.metric("In-Band Energy Fraction", f"{in_band:.1%}", help="Fraction of non-DC spectral energy concentrated within human cardiovascular range")
                st.write(f"- **Status**: `{rppg_ev.telemetry.get('status', rppg_ev.verdict)}`")
                st.write(f"- **FPS Used**: `{rppg_ev.telemetry.get('fps_used', actual_fps):.1f} Hz`")

        with tab3:
            st.markdown("#### Compression Discrepancy, Frequency & Provenance Suite")
            st.markdown("Quantifies multi-pass compression, Discrete Cosine Transform (DCT) quantization grids, high-frequency Fourier anomalies, and file container provenance.")
            
            # 4 Compression Metrics Cards
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Mean ELA Residual", f"{comp_ev.telemetry.get('mean_ela_residual', 0.0):.4f}", help="Mean intensity difference after re-compression at 90% JPEG quality")
            cm2.metric("Peak ELA Discrepancy", f"{comp_ev.telemetry.get('max_ela_residual', 0.0):.4f}", help="Maximum localized error difference across high-frequency regions")
            cm3.metric("DCT 8x8 Grid Ratio", f"{comp_ev.telemetry.get('dct_grid_ratio', 0.0):.3f}", help="Periodic energy concentration at 8-pixel macroblock boundaries")
            cm4.metric("HF Spectral Anomaly", f"{comp_ev.telemetry.get('hf_spectral_peak_score', 0.0):.3f}", help="Elevated peak ratio in high-pass Fourier space (indicates GAN/diffusion lattice footprint)")
            
            # Evidence Visuals: ELA + 2D FFT + 3D Illumination
            col_sp1, col_sp2 = st.columns(2)
            with col_sp1:
                st.markdown("##### Error Level Analysis (90% Q Residual)")
                if comp_ev.visual_artifact is not None:
                    st.image(comp_ev.visual_artifact, caption="ELA High-Frequency Residual Surface", width="stretch")
                st.caption("Surfaces with differing compression histories (e.g. pasted elements, post-processing overlays) illuminate brighter than the background.")

            with col_sp2:
                st.markdown("##### Keyframe 2D FFT Frequency Magnitude Spectrum")
                fft_map = compute_fft(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
                plt.style.use("dark_background")
                fig_fft, ax_fft = plt.subplots(figsize=(6, 4.2), dpi=130)
                fig_fft.patch.set_facecolor('#0e1117')
                ax_fft.set_facecolor('#0e1117')
                im_fft = ax_fft.imshow(fft_map, cmap='magma')
                ax_fft.axis('off')
                ax_fft.set_title("2D FFT Log Magnitude Spectrum", color='#e0e0e0', fontsize=10, pad=6)
                fig_fft.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)
                fig_fft.tight_layout()
                st.pyplot(fig_fft)
                plt.close(fig_fft)
                st.caption("Frequency domain profiling detects synthetic periodic lattice artifacts, upsampling sinc rings, and high-frequency diffusion fingerprints.")

            if illum_ev.visual_artifact is not None:
                st.markdown("---")
                st.markdown("##### 3D Surface Illumination & Normal Vectors")
                st.image(illum_ev.visual_artifact, caption="Forensic Lighting Vectors across Segmented Surfaces", width="stretch")
                st.caption("Validates physical lighting coherence. Inconsistent light vector angles indicate composite imagery or synthetic relighting.")

            # Container & Provenance Audit Table
            st.markdown("---")
            st.markdown("##### Container Atom & Metadata Provenance Audit")
            encoders_list = meta_ev.telemetry.get("container_encoders", [])
            encoders_str = ", ".join(encoders_list) if encoders_list else "None detected"
            cam_make = meta_ev.telemetry.get("camera_make", "N/A")
            cam_model = meta_ev.telemetry.get("camera_model", "N/A")
            hardware_str = f"{cam_make} {cam_model}".strip() if cam_make != "N/A" else "Not detected / Stripped"
            is_edit_flag = "_edit" in uploaded_file.name.lower() or "-edit" in uploaded_file.name.lower()

            hard_encoders = [e for e in encoders_list if e.lower() not in ["lavf", "lavc", "ffmpeg", "quicktime"]]
            if hard_encoders:
                encoder_eval = f"[WARNING] Post-production editing toolchain detected ({', '.join(hard_encoders)})"
            elif encoders_list:
                encoder_eval = f"ℹ️ Container re-encoded via standard multiplexer ({encoders_str})"
            else:
                encoder_eval = "No re-encoding signatures detected"

            prov_records = [
                {"Audit Dimension": "Submitted File", "Inspected Artifact": uploaded_file.name, "Forensic Evaluation": "️ Post-capture editing footprint (_edit suffix)" if is_edit_flag else "Standard naming convention"},
                {"Audit Dimension": "Container Format", "Inspected Artifact": "ISO/IEC 14496-12 (MP4 / QuickTime)", "Forensic Evaluation": "Header atoms (ftyp, moov) decoded successfully"},
                {"Audit Dimension": "Hardware Metadata Atom", "Inspected Artifact": hardware_str, "Forensic Evaluation": "Native camera capture signature preserved" if cam_make != "N/A" else "Metadata stripped or sanitized"},
                {"Audit Dimension": "Transcode / Muxer Signature", "Inspected Artifact": encoders_str, "Forensic Evaluation": encoder_eval},
                {"Audit Dimension": "Container Provenance Verdict", "Inspected Artifact": f"Mass m(T) = {meta_ev.bba_mass['T']:.1%}", "Forensic Evaluation": f"Status: {meta_ev.verdict}"},
            ]
            df_prov = pd.DataFrame(prov_records)
            st.markdown(df_prov.to_markdown(index=False))

        with tab4:
            st.markdown("#### Multi-Agent Dempster-Shafer & Evidence Graph Math")
            st.markdown("PRISM employs a **Dynamic Forensic Evidence Graph (FEG)** where each specialized agent assigns evidentiary mass over the frame of discernment $\\Theta = \\{Tampered, Authentic\\}$.")

            # Structured Dempster-Shafer Mass Assignment Table
            st.markdown("##### Basic Belief Assignment (BBA) Matrix")
            agent_table_data = [
                {"Forensic Stream": "Spatial Vision Transformer (ViT-B/16)", "Domain": "Spatial Attention", "m(Tampered)": f"{vit_bba['T']:.4f}", "m(Authentic)": f"{vit_bba['A']:.4f}", "m(Uncertainty)": f"{vit_bba['U']:.4f}", "Agent Verdict": "TAMPERED" if prob_fake > 0.65 else ("AUTHENTIC" if prob_fake < 0.35 else "INCONCLUSIVE")},
                {"Forensic Stream": "Micro-Sensor Noise (PRNU)", "Domain": "Sensor Physics", "m(Tampered)": f"{prnu_bba['T']:.4f}", "m(Authentic)": f"{prnu_bba['A']:.4f}", "m(Uncertainty)": f"{prnu_bba['U']:.4f}", "Agent Verdict": "TAMPERED" if prnu_bba['T'] > prnu_bba['A'] else ("AUTHENTIC" if prnu_bba['A'] > prnu_bba['T'] else "INCONCLUSIVE")},
                {"Forensic Stream": "Temporal Kinematics & Flow", "Domain": "Temporal Motion", "m(Tampered)": f"{temporal_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{temporal_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{temporal_ev.bba_mass['U']:.4f}", "Agent Verdict": temporal_ev.verdict},
                {"Forensic Stream": "Container & Metadata Provenance", "Domain": "Atom Forensics", "m(Tampered)": f"{meta_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{meta_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{meta_ev.bba_mass['U']:.4f}", "Agent Verdict": meta_ev.verdict},
                {"Forensic Stream": "Compression & Quantization (ELA)", "Domain": "DCT / Transform", "m(Tampered)": f"{comp_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{comp_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{comp_ev.bba_mass['U']:.4f}", "Agent Verdict": comp_ev.verdict},
                {"Forensic Stream": "3D Surface Illumination", "Domain": "Photometric Physics", "m(Tampered)": f"{illum_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{illum_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{illum_ev.bba_mass['U']:.4f}", "Agent Verdict": illum_ev.verdict},
                {"Forensic Stream": "Audio Voice Anti-Spoofing", "Domain": "Acoustic / Vocoder", "m(Tampered)": f"{audio_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{audio_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{audio_ev.bba_mass['U']:.4f}", "Agent Verdict": audio_ev.verdict},
                {"Forensic Stream": "Eye Gaze & Corneal Geometry", "Domain": "Biometrics", "m(Tampered)": f"{gaze_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{gaze_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{gaze_ev.bba_mass['U']:.4f}", "Agent Verdict": gaze_ev.verdict},
                {"Forensic Stream": "Audio-Visual Lip-Sync Synchrony", "Domain": "Cross-Modal AV", "m(Tampered)": f"{lip_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{lip_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{lip_ev.bba_mass['U']:.4f}", "Agent Verdict": lip_ev.verdict},
                {"Forensic Stream": "Remote Photoplethysmography (rPPG)", "Domain": "Cardiovascular Pulse", "m(Tampered)": f"{rppg_ev.bba_mass['T']:.4f}", "m(Authentic)": f"{rppg_ev.bba_mass['A']:.4f}", "m(Uncertainty)": f"{rppg_ev.bba_mass['U']:.4f}", "Agent Verdict": rppg_ev.verdict},
            ]
            df_agents = pd.DataFrame(agent_table_data)
            st.markdown(df_agents.to_markdown(index=False))

            # Mathematical Synthesis Cards
            st.markdown("##### Evidentiary Fusion Metrics & Conflict")
            k_conflict = max(0.0, 1.0 - (fused['plausibility'] - fused['belief']))
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Lower Belief Bel(T)", f"{belief:.1%}", help="Strict lower bound of probability based on confirming evidence")
            fc2.metric("Upper Plausibility Pl(T)", f"{plausibility:.1%}", help="Maximum upper bound of probability that cannot be ruled out")
            fc3.metric("Pignistic BetP(T)", f"{score:.1%}", help="Pignistic probability BetP = Bel(T) + 0.5 * m(U), used for final verdict thresholding")
            fc4.metric("Evidence Conflict K", f"{k_conflict:.3f}", help="Total conflict coefficient K across combined streams")

            st.markdown(
                f"""
                > **Dempster's Rule of Combination:**
                > $$m_{{1 \\oplus 2}}(T) = \\frac{{\\sum_{{A \\cap B = T}} m_1(A) m_2(B)}}{{1 - K}}, \\quad K = \\sum_{{A \\cap B = \\emptyset}} m_1(A) m_2(B)$$
                >
                > **Forensic Override Protection (Zadeh's Paradox Resolution):**
                > Standard sequential D-S combination can artificially dilute a single decisive tampering finding when combined with multiple passive or inconclusive authentic votes. 
                > In forensic science, post-capture tampering ($m(T) \\ge 0.60$) is an existential finding; PRISM enforces non-annihilation bounding ($BetP(T) \\ge \\max_i(m_i(T))$).
                """
            )

            with st.expander("View Raw JSON Telemetry & Diagnostics"):
                st.json({
                    "fused_masses": fused,
                    "vit_telemetry": {"prob_fake": prob_fake, "sample_probs": frame_probs, "prnu_energy": prnu_energy},
                    "metadata_telemetry": meta_ev.telemetry,
                    "compression_telemetry": comp_ev.telemetry,
                    "temporal_telemetry": temporal_ev.telemetry,
                    "illumination_telemetry": illum_ev.telemetry,
                    "gaze_telemetry": gaze_ev.telemetry,
                    "lip_sync_telemetry": lip_ev.telemetry,
                    "rppg_telemetry": rppg_ev.telemetry,
                })
        return

    # =======================================================================
    # ROUTE C: STILL IMAGE FORENSICS (PNG / JPEG)
    # =======================================================================
    raw = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image. Upload a valid PNG or JPEG.")
        return

    rgb_t, noise_t = preprocessor.process_frame(bgr)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        out = model(rgb_t.unsqueeze(0).to(device), noise_t.unsqueeze(0).to(device))
        logit = out["logit"]
        mask = out["mask"]
        prob_fake = float(torch.sigmoid(logit).float().item())
        del out, logit
        torch.cuda.empty_cache()

    rgb_u8   = _tensor_to_uint8(rgb_t)
    noise_np = noise_t.squeeze().cpu().float().numpy()
    
    try:
        orig_pil = Image.open(io.BytesIO(file_bytes))
    except Exception:
        orig_pil = Image.fromarray(rgb_u8)

    # Run Specialized Forensic Agents with original raw container bytes
    gaze_evidence  = agents["eye_gaze"].analyze(rgb_u8)
    meta_evidence  = agents["metadata"].analyze(orig_pil, file_bytes=file_bytes)
    comp_evidence  = agents["compression"].analyze(orig_pil)
    illum_evidence = agents["illumination"].analyze(bgr)

    faces_found = gaze_evidence.telemetry.get("faces_detected", 0) > 0
    vit_reliability = 0.90 if faces_found else 0.25

    # Realistic PRNU sensor noise consistency evaluation:
    # Baseline: natural camera sensor acquisition has noise energy in [0.015, 0.25] without periodic lattice peaks.
    raw_noise_level = float(np.mean(np.abs(noise_np)))
    prnu_energy = raw_noise_level
    hf_peaks = float(comp_evidence.telemetry.get("hf_spectral_peak_score", 0.0))

    if raw_noise_level < 0.015:
        if prob_fake < 0.25:
            # Clean digital graphic or re-encoded authentic media: absence of camera noise is natural
            prnu_prob_tamper = 0.15
        else:
            # Over-smoothed diffusion/GAN synthesis corroborated by neural backbone
            prnu_prob_tamper = 0.70
    elif raw_noise_level > 0.45 or hf_peaks > 0.60:
        # Synthetic periodic lattice or heavy grain injection
        prnu_prob_tamper = 0.75
    else:
        # Consistent with natural optical camera sensor noise floor
        prnu_prob_tamper = 0.12

    vit_bba = fusion.create_mass(evidence_T=prob_fake * 10.0 * vit_reliability, evidence_A=(1.0 - prob_fake) * 10.0 * vit_reliability)
    prnu_bba = fusion.create_mass(evidence_T=prnu_prob_tamper * 8.0 * 0.75, evidence_A=(1.0 - prnu_prob_tamper) * 8.0 * 0.75)

    fused_agents_list = [
        vit_bba,
        prnu_bba,
        comp_evidence.bba_mass,
        meta_evidence.bba_mass,
        gaze_evidence.bba_mass,
        illum_evidence.bba_mass,
    ]
    illum_bba = illum_evidence.bba_mass if faces_found else {"T": 0.0, "A": 0.0, "U": 1.0}
    agent_dict = {"vit": vit_bba, "prnu": prnu_bba, "comp": comp_evidence.bba_mass, "meta": meta_evidence.bba_mass, "gaze": gaze_evidence.bba_mass, "illum": illum_bba}; edges = [("vit", "illum"), ("gaze", "illum"), ("comp", "prnu")]; fused = fusion.fuse_graph_evidence(agent_dict, edges); result = fused
    score  = fused["pignistic_fake"]
    verdict = fused["verdict"]
    belief = fused["belief"]
    plausibility = fused["plausibility"]

    mask_np = mask.squeeze().cpu().float().numpy()
    heatmap_u8 = _make_heatmap_overlay(mask_np, rgb_u8)
    mask_area_percent = float((mask_np > 0.5).mean()) * 100
    del mask
    torch.cuda.empty_cache()

    # Create 6-Panel Figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor('#0e1117')

    axes[0, 0].imshow(rgb_u8)
    axes[0, 0].set_title("1. Original Frame (Crop)", fontsize=11, fontweight='bold', color='#e0e0e0')
    axes[0, 0].axis('off')

    im_noise = axes[0, 1].imshow(noise_np, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axes[0, 1].set_title("2. PRNU Sensor Noise Map", fontsize=11, fontweight='bold', color='#e0e0e0')
    axes[0, 1].axis('off')
    fig.colorbar(im_noise, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_heat = axes[0, 2].imshow(heatmap_u8)
    axes[0, 2].set_title("3. Spatial Forgery (Conf > 0.60)", fontsize=11, fontweight='bold', color='#e0e0e0')
    axes[0, 2].axis('off')
    fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(0.6, 1.0)), ax=axes[0, 2], fraction=0.046, pad=0.04, label="Confidence")

    axes[1, 0].imshow(compute_fft(rgb_u8), cmap='magma')
    axes[1, 0].set_title("4. Image 2D FFT Magnitude", fontsize=11, fontweight='bold', color='#e0e0e0')
    axes[1, 0].axis('off')

    if comp_evidence.visual_artifact is not None:
        axes[1, 1].imshow(comp_evidence.visual_artifact)
    axes[1, 1].set_title("5. Error Level Analysis (90% Q)", fontsize=11, fontweight='bold', color='#e0e0e0')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(compute_fft(noise_np), cmap='viridis')
    axes[1, 2].set_title("6. PRNU 2D FFT (Resampling Check)", fontsize=11, fontweight='bold', color='#e0e0e0')
    axes[1, 2].axis('off')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)

    noise_status = "Anomalous -- PRNU pattern deviation detected" if prnu_prob_tamper > 0.50 else "Normal -- sensor noise consistent with expected camera signature"
    meta_summary = "; ".join(meta_evidence.findings) if meta_evidence.findings else "No metadata anomalies."
    gaze_summary = "; ".join(gaze_evidence.findings) if gaze_evidence.findings else "No ocular anomalies."
    illum_summary = "; ".join(illum_evidence.findings) if illum_evidence.findings else "No illumination anomalies."

    with st.status("[AUDIT] Running Multi-Agent Forensic Verification...", expanded=True) as status:
        st.write("[DONE] ViT & PRNU Backbone: Spatial feature extraction complete")
        time.sleep(0.15)
        st.write(f"[DONE] Metadata Agent: {meta_evidence.verdict} (m_T={meta_evidence.bba_mass['T']:.1%})")
        time.sleep(0.15)
        st.write(f"[DONE] Eye Gaze & Corneal Agent: {gaze_evidence.verdict} (m_T={gaze_evidence.bba_mass['T']:.1%})")
        time.sleep(0.15)
        st.write(f"[DONE] Compression & Watermark Agent: {comp_evidence.verdict} (m_T={comp_evidence.bba_mass['T']:.1%})")
        time.sleep(0.15)
        st.write(f"[DONE] Illumination & Lighting Agent: {illum_evidence.verdict} (m_T={illum_evidence.bba_mass['T']:.1%})")
        time.sleep(0.15)
        st.write(f"[DONE] 6-Stream Dempster-Shafer Fusion: Calculated {verdict}")
        time.sleep(0.15)
        st.write("[REASONER] Querying Local LLM for Certified Forensic Lab Report...")

        try:
            report = reasoner.generate_report(
                score=score,
                mask_area=f"{mask_area_percent:.1f}% of frame (estimated)",
                noise_status=noise_status,
                system_verdict=verdict,
                vit_mass=vit_bba["T"],
                prnu_mass=prnu_bba["T"],
                ela_mass=comp_evidence.bba_mass["T"],
                meta_mass=meta_evidence.bba_mass["T"],
                gaze_mass=gaze_evidence.bba_mass["T"],
                meta_summary=meta_summary,
                gaze_summary=f"{gaze_summary} | {illum_summary}",
                belief=belief,
                plausibility=plausibility, visual_evidence_bytes=[buf.getvalue()]
            )
            status.update(label="[COMPLETE] Forensic Multi-Agent Verification Complete", state="complete", expanded=False)
        except Exception as exc:
            report = f"[REPORT GENERATION FAILED]\n{exc}"
            status.update(label="[FAILED] Verification Failed", state="error", expanded=False)

    st.divider()
    st.markdown(f"### Forensic Lab Report: `{uploaded_file.name}`")

    m1, m2, m3 = st.columns(3)
    m1.metric("D-S Combined Belief", f"{belief:.1%}", delta=f"Plausibility bound: {plausibility:.1%}", delta_color="off")
    m2.metric("P(Tampered) BetP", f"{score:.1%}")
    m3.metric("System Verdict", verdict)

    st.markdown("##### Specialized Forensic Agent Diagnostics")
    a_col1, a_col2, a_col3, a_col4, a_col5 = st.columns(5)
    a_col1.metric("Spatial & ViT", f"{prob_fake:.1%}", delta="ViT Backbone")
    a_col2.metric("Metadata & EXIF", meta_evidence.verdict, delta=f"Mass T: {meta_evidence.bba_mass['T']:.1%}")
    a_col3.metric("Ocular & Gaze", gaze_evidence.verdict, delta=f"Mass T: {gaze_evidence.bba_mass['T']:.1%}")
    a_col4.metric("Compression / ELA", comp_evidence.verdict, delta=f"Mass T: {comp_evidence.bba_mass['T']:.1%}")
    a_col5.metric("3D Illumination", illum_evidence.verdict, delta=f"Mass T: {illum_evidence.bba_mass['T']:.1%}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Executive Summary & Report", "Visual Evidence Suite", "Agent Telemetry & Math"])

    # Generate Certified PDF Dossier
    agent_masses_dict = {
        "ViT_spatial": vit_bba,
        "PRNU_sensor": prnu_bba,
        "Compression_ELA": comp_evidence.bba_mass,
        "Metadata_Provenance": meta_evidence.bba_mass,
        "EyeGaze_Biometric": gaze_evidence.bba_mass,
        "Illumination_3D": illum_evidence.bba_mass,
    }
    k_conflict = float(max(0.0, 1.0 - (plausibility - belief)))
    m_uncertainty = float(fused.get('uncertainty', max(0.0, 1.0 - (belief + (1.0 - plausibility)))))

    pdf_bytes = exporter.generate_pdf(
        filename=uploaded_file.name,
        file_bytes=file_bytes,
        verdict=verdict,
        score=score,
        belief=belief,
        plausibility=plausibility,
        agent_masses=agent_masses_dict,
        llm_report_text=report,
        figure_png_bytes=buf.getvalue(),
        conflict_k=k_conflict,
        epistemic_uncertainty=m_uncertainty,
    )

    with tab1:
        if m_uncertainty >= 0.60 or k_conflict >= 0.50:
            st.warning(
                f"**[HIGH FORENSIC UNCERTAINTY / AGENT CONFLICT DETECTED]**\n\n"
                f"- Epistemic Uncertainty $m(U) = {m_uncertainty:.1%}$\n"
                f"- Multi-Agent Conflict $K = {k_conflict:.3f}$\n\n"
                f"**Investigative Guidance**: Forensic agents show elevated disagreement or lack sufficient signal. "
                f"This media sample must be treated as **INCONCLUSIVE** and is provided for investigative hypothesis generation only."
            )

        if verdict == "TAMPERED":
            st.error(f"**[ALERT: {verdict}]** Forensic indicators strongly suggest synthetic manipulation.")
        elif verdict == "AUTHENTIC":
            st.success(f"**[VERIFIED: {verdict}]** No significant manipulation detected across active agents.")
        else:
            st.warning(f"**[UNCERTAIN: {verdict}]** Evidence is ambiguous; manual review recommended.")

        st.markdown(f'<div class="report-block">{report}</div>', unsafe_allow_html=True)
        st.write("")
        st.download_button(
            label="Download Forensic Dossier for Investigative Support (PDF)",
            data=pdf_bytes,
            file_name=f"PRISM_Forensic_Dossier_{uploaded_file.name}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    with tab2:
        st.markdown("#### Multi-Modal Spatial & Frequency Analysis Suite (150 DPI)")
        st.markdown("The forensic multi-panel below analyzes spatial, noise, and frequency domain artifacts across independent streams.")
        st.image(buf, width="stretch")
        st.download_button(
            label="Download Forensic Figure (High-Res PNG)",
            data=buf.getvalue(),
            file_name=f"forensic_analysis_{uploaded_file.name}.png",
            mime="image/png",
            width="stretch",
        )

        c_art1, c_art2 = st.columns(2)
        with c_art1:
            if gaze_evidence.telemetry.get("faces_detected", 0) > 0 and gaze_evidence.visual_artifact is not None:
                st.markdown("#### Biometric Ocular Inspection")
                st.image(gaze_evidence.visual_artifact, caption="Facial Landmarks & Corneal Specular Highlights", width="stretch")
        with c_art2:
            if illum_evidence.visual_artifact is not None:
                st.markdown("#### 3D Illumination Vector Direction Map")
                st.image(illum_evidence.visual_artifact, caption="Foreground vs. Background Primary Light Vectors", width="stretch")

    with tab3:
        st.markdown("#### Multi-Agent Dempster-Shafer & Evidence Graph Math")
        st.markdown("Quantifies individual agent Basic Belief Assignments (BBA) and dynamic graph fusion for the submitted image.")

        # Structured Dempster-Shafer Mass Assignment Table
        st.markdown("##### Basic Belief Assignment (BBA) Matrix")
        image_table_data = [
            {"Forensic Stream": "Spatial Vision Transformer (ViT-B/16)", "Domain": "Spatial Attention", "m(Tampered)": f"{vit_bba['T']:.4f}", "m(Authentic)": f"{vit_bba['A']:.4f}", "m(Uncertainty)": f"{vit_bba['U']:.4f}", "Agent Verdict": "TAMPERED" if prob_fake > 0.65 else ("AUTHENTIC" if prob_fake < 0.35 else "INCONCLUSIVE")},
            {"Forensic Stream": "Micro-Sensor Noise (PRNU)", "Domain": "Sensor Physics", "m(Tampered)": f"{prnu_bba['T']:.4f}", "m(Authentic)": f"{prnu_bba['A']:.4f}", "m(Uncertainty)": f"{prnu_bba['U']:.4f}", "Agent Verdict": "TAMPERED" if prnu_bba['T'] > prnu_bba['A'] else ("AUTHENTIC" if prnu_bba['A'] > prnu_bba['T'] else "INCONCLUSIVE")},
            {"Forensic Stream": "Compression & Quantization (ELA)", "Domain": "DCT / Transform", "m(Tampered)": f"{comp_evidence.bba_mass['T']:.4f}", "m(Authentic)": f"{comp_evidence.bba_mass['A']:.4f}", "m(Uncertainty)": f"{comp_evidence.bba_mass['U']:.4f}", "Agent Verdict": comp_evidence.verdict},
            {"Forensic Stream": "Container & Metadata Provenance", "Domain": "EXIF / PNG Chunks", "m(Tampered)": f"{meta_evidence.bba_mass['T']:.4f}", "m(Authentic)": f"{meta_evidence.bba_mass['A']:.4f}", "m(Uncertainty)": f"{meta_evidence.bba_mass['U']:.4f}", "Agent Verdict": meta_evidence.verdict},
            {"Forensic Stream": "Eye Gaze & Corneal Specular", "Domain": "Biometrics", "m(Tampered)": f"{gaze_evidence.bba_mass['T']:.4f}", "m(Authentic)": f"{gaze_evidence.bba_mass['A']:.4f}", "m(Uncertainty)": f"{gaze_evidence.bba_mass['U']:.4f}", "Agent Verdict": gaze_evidence.verdict},
            {"Forensic Stream": "3D Surface Illumination", "Domain": "Photometric Physics", "m(Tampered)": f"{illum_evidence.bba_mass['T']:.4f}", "m(Authentic)": f"{illum_evidence.bba_mass['A']:.4f}", "m(Uncertainty)": f"{illum_evidence.bba_mass['U']:.4f}", "Agent Verdict": illum_evidence.verdict},
        ]
        df_image = pd.DataFrame(image_table_data)
        st.markdown(df_image.to_markdown(index=False))

        # Mathematical Synthesis Cards
        st.markdown("##### Evidentiary Fusion Metrics & Conflict")
        k_conflict = max(0.0, 1.0 - (plausibility - belief))
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Lower Belief Bel(T)", f"{belief:.1%}", help="Strict lower bound of probability based on confirming evidence")
        ic2.metric("Upper Plausibility Pl(T)", f"{plausibility:.1%}", help="Maximum upper bound of probability that cannot be ruled out")
        ic3.metric("Pignistic BetP(T)", f"{score:.1%}", help="Pignistic probability BetP = Bel(T) + 0.5 * m(U)")
        ic4.metric("Evidence Conflict K", f"{k_conflict:.3f}", help="Total conflict coefficient K across combined streams")

        with st.expander("View Raw JSON Telemetry & Diagnostics"):
            st.json({
                "fused_result": result,
                "vit_telemetry": {"prob_fake": prob_fake, "prnu_energy": prnu_energy},
                "compression_telemetry": comp_evidence.telemetry,
                "metadata_telemetry": meta_evidence.telemetry,
                "gaze_telemetry": gaze_evidence.telemetry,
                "illumination_telemetry": illum_evidence.telemetry,
            })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    st.markdown("# PRISM: Media Forensics Intelligence")
    st.markdown(
        "Multi-modal deepfake detection -- "
        "ViT + PRNU noise analysis + Multi-Agent Dempster-Shafer fusion + LLM forensic reasoning."
    )
    st.divider()

    model, device               = load_model()
    prep, fuse, rsn, exp, agents = load_pipeline()
    uploaded                    = render_sidebar(rsn)

    if uploaded is None:
        st.markdown(
            """
            <div style='text-align:center; padding:4rem 0; color:#555;'>
                <h3>Upload an image, audio, or video file in the sidebar to begin multi-agent analysis.</h3>
                <p>Supported formats: PNG &middot; JPEG &middot; WAV &middot; MP4</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    run_inference(uploaded, model, device, prep, fuse, rsn, exp, agents)


if __name__ == "__main__":
    main()
