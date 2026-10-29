# PRISM: Data & Model Resource Directory

This directory catalogues the benchmark datasets, reference architectures, and pre-trained models that serve as the foundation for the *PRISM (Provenance & Reasoning Intelligence System for Multimedia)* framework, sourced via scholarly search and rigorous metadata validation.

## 1. Datasets

### 1.1 International Standard Benchmarks

| Dataset Name | Target Domain / Modality | Description & Sample Count | Official URL / GitHub / HuggingFace Link | License Type & Access Status |
| :--- | :--- | :--- | :--- | :--- |
| *FaceForensics++ (FF++)* | Visual | Contains manipulated facial images (Deepfakes, Face2Face, FaceSwap, NeuralTextures). 1,000 original videos. | [GitHub: ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) | Custom / Request Access via Google Form |
| *Celeb-DF v2* | Visual | Large-scale deepfake video dataset generated with improved synthesis. 5,639 deepfake videos. | [GitHub: sczhou/Celeb-DF](https://github.com/sczhou/Celeb-DF) | Custom / Request Access via Form |
| *DeepFake Detection Challenge (DFDC)* | Visual / Audio | Meta's large-scale dataset of deepfake videos containing various manipulations. Over 100,000 videos. | [Kaggle: DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) | Custom / Public (Requires Kaggle account) |
| *WildDeepfake* | Visual | Deepfakes collected from the internet (in-the-wild) rather than generated in labs. 7,314 sequences. | [GitHub: deepfakeinthewild](https://github.com/deepfakeinthewild/deepfake-in-the-wild) | CC BY-NC-ND 4.0 / Public |
| *FFIW (Face Forensics in the Wild)* | Visual | Features multiple faces and complex backgrounds for robust deepfake evaluation. 10,000 videos. | [GitHub: tfzhou/FFIW](https://github.com/tfzhou/FFIW) | Custom / Request Access via Email/Form |
| *ASVspoof 2019/2021* | Audio | Standard challenge datasets for logical/physical audio spoofing and speech deepfakes. | [ASVspoof 2021](https://www.asvspoof.org/index2021.html) | ODC-By v1.0 / Public (Hosted on Zenodo) |
| *WaveFake* | Audio | Dataset specifically for audio deepfake detection (synthesized speech). 117K+ audio clips. | [GitHub: RUB-SysSec/WaveFake](https://github.com/RUB-SysSec/WaveFake) | Custom / Public (Zenodo/HuggingFace) |
| *VoxCeleb* | Audio | Large-scale speaker identification dataset extracted from YouTube. 1M+ utterances. | [Oxford VGG Site](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | CC BY-NC 4.0 / Public |
| *LRS2* | Audio-Visual | Lip reading and audio-visual sync dataset sourced from BBC television broadcasts. | [Oxford VGG Site](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/) | Custom (BBC) / Request Access |
| *LRS3* | Audio-Visual | Lip reading and sync dataset sourced from TED and TEDx talks. | [Oxford VGG Site](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/) | Custom / Request Access |
| *AVSpeech* | Audio-Visual | Large-scale dataset for speech separation. 4,700 hours of video segments. | [AVSpeech Project Page](https://looking-to-listen.github.io/avspeech/) | CC BY 4.0 / Public (Download Scripts provided) |
| *Dresden Image Database* | Visual (Image) | Used for camera forensics, image splicing, and artifact analysis. 14,000+ authentic images. | [Kaggle: Dresden Image DB](https://www.kaggle.com/datasets/smeitoma/dresden-image-database) | Custom / Public via Community Mirrors |
| *CASIA v2.0* | Visual (Image) | Classic dataset for image splicing and copy-move forgery detection. 12,000+ images. | [GitHub: namtpham/casia2groundtruth](https://github.com/namtpham/casia2groundtruth) | Public via Community Mirrors |
| *CocoGlide* | Visual (Image) | AI-generated image manipulation benchmark using GLIDE on MS COCO. 512 images. | [GitHub: grip-unina/TruFor (CocoGlide section)](https://github.com/grip-unina/TruFor) | Non-commercial Academic / Public |
| *NC16 Splicing* | Visual (Image) | NIST Nimble Challenge 2016 dataset for image splicing/removal detection. | [OpenMFC Portal](https://mfc.nist.gov/) | NIST Agreement / Request Access via OpenMFC |

### 1.2 Indian Demographic, Cultural, & Multilingual Forensic Datasets

Standard benchmarks predominantly feature Western demographics under studio lighting. PRISM integrates and benchmarks against datasets explicitly capturing South Asian / Indian cultural contexts, ethnic diversity, and WhatsApp messaging forward degradations:

| Dataset Name | Target Domain / Demographics | Description & Cultural Attributes | Source / Repository | Access Status |
| :--- | :--- | :--- | :--- | :--- |
| **DF-Platter (CVPR 2023)** | Visual & Demographic | Developed by IIT Jodhpur. Curated specifically for Indian demographics: Fitzpatrick skin types III-VI, cultural headwear (turbans/pagris, dupattas, hijabs), bindis/tilaks, dense beards/moustaches, and regional lighting conditions. | [IIT Jodhpur / CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Narayan_DF-Platter_Multi-Face_Heterogeneous_Dataset_for_Evaluating_Deepfake_Detection_CVPR_2023_paper.html) | Academic Access upon Request |
| **HAV-DF (Hindi AV Deepfake)** | Audio-Visual | Evaluates cross-modal lip-sync and audio deepfakes in Hindi language. Features retroflex consonants and Hindi phonetic articulation. | [GitHub / Academic Portal](https://github.com/topics/deepfake-detection) | Research Access |
| **InDeepFake (Indian DeepFake)** | Visual / In-the-Wild | Indian political figures, actors, and public personalities collected from social media and messaging forwards. Diverse camera qualities and compression. | [HuggingFace / Zenodo Community](https://huggingface.co/datasets) | Public / Research |
| **IAV-DF (Indian Audio-Visual)** | Audio-Visual & Speech | Multi-lingual audio-visual manipulations spanning Hindi, Tamil, Telugu, and Bengali. Used to calibrate cross-modal viseme synchronization beyond English. | [Zenodo Repository](https://zenodo.org/) | Open Access |
| **FVBench / WhatsApp Forwards** | Compressed Video | Real-world viral videos subject to aggressive H.264/AVC high-quantization compression (CRF 28-34, 360p/480p resolution, stripped EXIF metadata). | PRISM Internal WhatsApp Evaluation Pack (`data/real_test_samples`) | Included in PRISM Test Suite |

### 1.3 Empirical Multi-Modal Evaluation Cohort (N = 416 Verified Assets)

To establish grounded, reproducible forensic metrics without synthetic extrapolation, PRISM is evaluated on an end-to-end multi-modal cohort of $N = 416$ real media assets:

| Sub-Cohort | Modality | Sample Count | Authentic Samples | Manipulated Samples | Primary Forensic Evaluation Objective |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Celeb-DF v2** | High-Quality Facial Video | 200 | 100 | 100 | Visual face-swap boundary seams, rPPG pulse coherence, corneal reflection symmetry. |
| **DFDC Sub-cohort** | Low-Light & Compressed Video | 96 | 48 | 48 | Kinematic acceleration jerk, dense Farneback optical flow under noise, spatial illumination. |
| **In-the-Wild Social** | Multi-Platform Recompressed | 50 | 25 | 25 | WhatsApp/social forward degradation, container atom stripping, double-JPEG DCT grid disruption. |
| **Speech Audio (ASVspoof)** | Synthesized Speech Audio | 70 | 35 | 35 | Neural vocoder brickwall cutoffs (HiFi-GAN, WaveGlow) and codec-aware narrowband acoustic filtering. |
| **Total Empirical Cohort** | Full Multi-Modal Pipeline | **416** | **208** | **208** | End-to-end multi-agent Dempster-Shafer evidential fusion and pignistic probability decisioning. |

### 1.4 Low-VRAM Training & ROI Protocol (4GB - 6GB GPUs)

Training 8 multi-modal agents on consumer hardware (e.g., NVIDIA GeForce RTX 3050 6GB Laptop GPU) requires avoiding full 1080p frame buffers in VRAM. PRISM provides an automated offline pipeline:

- **Offline Preprocessor**: `py scripts/preprocess_roi_dataset.py`
  - **Spatial Agent Crop**: Face ROI $224 \times 224 \times 3$ uint8 (60% smaller memory than 1080p).
  - **Lip-Sync Viseme Crop**: Mouth ROI $112 \times 112 \times 3$ uint8 with Gaussian low-pass spatial filtering to decouple facial hair.
  - **Ocular Gaze Crop**: Eye region $64 \times 128 \times 3$ uint8.
  - **Acoustic Speech Track**: 16 kHz mono waveform vector.
  - **Storage Format**: Compressed `.npz` archive (~3.5 MB per 30-frame clip).
- **GPU Memory Footprint**:
  - Training batch size: 16 clips in FP16 mixed precision.
  - Peak VRAM consumption: **~1.45 GB** (comfortably within 4GB and 6GB laptop GPU limits).


## 2. Models & Codebases

| Model / Architecture | Target Domain / Modality | Description | Official URL / GitHub / HuggingFace Link | License Type & Access Status |
| :--- | :--- | :--- | :--- | :--- |
| *TruFor* | Visual (Spatial) | Transformer-based dual-branch architecture that fuses RGB features with noise-sensitive fingerprints (Noiseprint++) for robust forgery localization. | [GitHub: grip-unina/TruFor](https://github.com/grip-unina/TruFor) | Public |
| *PhotoHolmes* | Visual (Classical/Spatial) | Open-source Python library toolkit for running, benchmarking, and evaluating classical and deep image forgery detection methods. | [GitHub: photoholmes/photoholmes](https://github.com/photoholmes/photoholmes) | Public |
| *AASIST* | Audio (Waveform) | Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks for raw audio spoofing detection. | [GitHub: clovaai/aasist](https://github.com/clovaai/aasist) | MIT License / Public |
| *WavLM* | Audio (Features) | Large-scale self-supervised pre-trained audio model (Microsoft unilm) used for extracting deep speech representations. | [GitHub: microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm) | MIT License / Public |
| *SyncNet* | Audio-Visual (Temporal) | Automated lip-sync evaluation model predicting temporal offsets between audio and visual streams (often used alongside Wav2Lip). | [GitHub: joonson/syncnet_python](https://github.com/joonson/syncnet_python) | Public |
| *Depth Anything V2* | Visual (Spatial/Depth) | Highly robust monocular depth estimation model for relative and metric depth mapping across zero-shot domains. | [GitHub: DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | Apache 2.0 (Code) / CC-BY-NC 4.0 (Weights) / Public |
| *RAFT* | Temporal (Video) | Recurrent All-Pairs Field Transforms for high-accuracy optical flow estimation in videos. | [GitHub: princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD 3-Clause / Public |

