# Individual Project Contribution Report: Somil
## PRISM: Provenance & Reasoning Intelligence System for Multimedia

- **Student Name**: Somil
- **Course**: BCSE306L Mini-Project
- **Role / Domain**: Physical, Temporal & Container Forensic Micro-Agents Engineering
- **Allocated Contribution**: 33.33% (Equal Partner Contribution)

---

### 1. Primary Technical Deliverables & Ownership
1. **Container Metadata & Provenance Agent (`src/agents/metadata_agent.py`)**:
   - Engineered the raw byte container parser extracting EXIF, XMP, IPTC, and C2PA Content Credentials.
   - Built the regex signature matching suite for generative software tags (*Midjourney, DALL-E, Stable Diffusion, ComfyUI, Automatic1111*).
   - Designed WhatsApp / social media sanitized container handling: safely mapping stripped metadata to **Epistemic Ignorance ($m(U) = 1.0$)** rather than asserting false tampering.

2. **Compression & Error Level Analysis Agent (`src/agents/compression_watermark_agent.py`)**:
   - Implemented Error Level Analysis (ELA) recompressing images at 95% JPEG quality to highlight differential compression error rates on manipulated surfaces.
   - Built the 8x8 Discrete Cosine Transform (DCT) block artifact grid (BAG) discontinuity analyzer and high-frequency Fourier spectral peak detector.

3. **Dense Optical Flow & Temporal Kinematics Agent (`src/agents/video_temporal_agent.py`)**:
   - Implemented Farneback dense optical flow velocity field extraction across consecutive video bursts.
   - Designed circular standard deviation statistics on moving motion vectors to prevent $0^\circ / 360^\circ$ wrap-around boundary errors.
   - Engineered the discrete second-order acceleration ($a_t$) and kinematic jerk ($j_t$) sawtooth oscillation analyzer.
   - Added low-velocity motion regime detection ($<0.20$ px/f) to eliminate false positive alarms on stationary desktop screen recordings.

4. **3D Illumination Vector Agent (`src/agents/illumination_agent.py`)**:
   - Designed 3D Lambertian surface normal estimation from luminance spatial gradients.
   - Implemented foreground portrait subject vs background ambient illuminant vector divergence calculation ($\theta$).
   - Added diffuse / omnidirectional lighting coherence thresholds to prevent ambient room lights from causing false tampering alarms.

---

### 2. Key Code Modules Authored
- `src/agents/base_agent.py`  -  `BaseAgent` class, `AgentEvidence` dataclass, anti-leakage audit protocol
- `src/agents/metadata_agent.py`  -  `MetadataAgent`
- `src/agents/compression_watermark_agent.py`  -  `CompressionWatermarkAgent`
- `src/agents/video_temporal_agent.py`  -  `VideoTemporalAgent`
- `src/agents/illumination_agent.py`  -  `IlluminationAgent`
- `data/cache/perturbation_grid/`  -  Generated 105 systematically degraded benchmark samples
- `tests/test_forensic_vision_somil.py`  -  Physical & visual agents test suite

### 3. Verification & Metrics Achieved
- Standalone Compression ELA AUC: **0.7400**
- Standalone Temporal Optical Flow AUC: **0.6900**
- WhatsApp Stripped Metadata False Alarm Rate: **0.0%** (100% mapped to $m(U)=1.0$)
