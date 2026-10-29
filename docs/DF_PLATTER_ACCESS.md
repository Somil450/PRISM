# DF-Platter Dataset: Official Academic Access & Benchmark Guide

## 1. Overview
**DF-Platter** is the premier academic benchmark for evaluating deepfake detection algorithms across heterogeneous Indian demographics.

- **Developer**: Image Analysis and Biometrics (IAB) Lab, Indian Institute of Technology (IIT) Jodhpur
- **Publication**: Kartik Narayan, Harsh Agarwal, Kartik Thakral, Surbhi Mittal, Mayank Vatsa, Richa Singh. *"DF-Platter: Multi-Face Heterogeneous Dataset for Evaluating Deepfake Detection."* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023.
- **Official Portal**: [http://iab-rubric.org/df-platter-database](http://iab-rubric.org/df-platter-database)
- **Dataset Composition**:
  - **Total Videos**: 133,260 deepfake and authentic videos
  - **Subjects**: Indian demographic cohorts spanning diverse regional backgrounds
  - **Skin Tones**: Fitzpatrick Scale III, IV, V, and VI
  - **Cultural Headwear & Accessories**: Turbans (Pagris/Dastars), Dupattas (Chunnis), Hijabs, and culturally specific jewelry
  - **Facial Hair**: Full beards, moustaches, stubble, and clean-shaven subjects
  - **Manipulation Techniques**: DeepFaceLab (DFL), FaceSwap (FS), and Latent Diffusion Models (LDM)
  - **Compression Levels**: Raw (lossless), c23 (high quality), and c40 (heavy compression typical of social messaging platforms)

---

## 2. Official Access Request Procedure (EULA)

The DF-Platter dataset is distributed exclusively for academic and non-commercial research purposes under an End User License Agreement (EULA).

### Step-by-Step Request Instructions:
1. **Access the Portal**:
   Navigate to [http://iab-rubric.org/df-platter-database](http://iab-rubric.org/df-platter-database).
2. **Download the EULA Agreement**:
   Download the PDF End User License Agreement form from the portal.
3. **Fill and Sign**:
   - Complete all institutional details.
   - The document must be signed by a permanent faculty member, department head, or authorized institutional representative.
   - Individual student requests without faculty endorsement are generally not processed.
4. **Submit via Institutional Email**:
   Send the scanned signed agreement from an official university/institutional email domain (`.edu`, `.ac.in`, `.res.in`, or research organization) to:
   - **Recipient**: Image Analysis and Biometrics (IAB) Lab, IIT Jodhpur
   - **Email**: `iab-contact@iitj.ac.in`
   - **Subject Line**: `DF-Platter Dataset Access Request - [Your Institution Name]`
5. **Download Credentials**:
   Upon verification, the IAB lab will provide an encrypted download link or SFTP credentials.

---

## 3. Directory Layout in PRISM

Once downloaded and extracted, place the files in the PRISM directory structure as follows:

```
data/datasets/DF-Platter/
├── metadata.csv                     # Official annotation CSV
├── raw/                             # Lossless / raw videos
│   ├── authentic/
│   └── manipulated/
│       ├── deepfacelab/
│       ├── faceswap/
│       └── diffusion/
├── c23/                             # High-quality compressed (c23)
│   ├── authentic/
│   └── manipulated/
└── c40/                             # Heavily compressed (c40)
    ├── authentic/
    └── manipulated/
```

### Expected Metadata Format (`metadata.csv`):
```csv
video_id,relative_path,technique,compression,fitzpatrick,headwear,facial_hair,is_deepfake
df_0001,c23/deepfacelab/subj01.mp4,DeepFaceLab,c23,Fitzpatrick IV,Turban,Beard,1
df_0002,raw/authentic/subj02.mp4,Authentic,raw,Fitzpatrick III,None,Clean-shaven,0
```

---

## 4. Running the Evaluator

To run the automated fairness and demographic benchmark:

```bash
# Evaluate on downloaded DF-Platter dataset with metadata CSV:
py scripts/benchmark_df_platter.py --dataset_dir data/datasets/DF-Platter --metadata data/datasets/DF-Platter/metadata.csv

# Run local sample test (prints explicit disclaimer that results are NOT DF-Platter):
py scripts/benchmark_df_platter.py --sample_dir data/real_test_samples
```
