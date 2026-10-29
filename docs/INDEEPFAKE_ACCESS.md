# InDeepFake & SAMANVAYA Benchmark: Academic Access & Evaluation Guide

## 1. Overview
**InDeepFake** is the multimodal audio-visual deepfake benchmark specifically created for Indian populations and Indic language families.

- **Reference Paper**: Arnab Kumar Das, Ruchira Naskar, et al. *"InDeepFake: A Multimodal Multi-Lingual Deepfake Dataset for Indic Populations."* Pattern Recognition Letters, Elsevier, 2025.
- **Associated Framework**: SAMANVAYA (Multimodal Cross-Attention & Speech-Liveness)
- **Target Language Families**:
  - **Hindi** (Indo-Aryan family)
  - **Bengali** (Indo-Aryan family)
  - **Tamil** (Dravidian family)
  - **Telugu** (Dravidian family)
- **Modalities**:
  - **Acoustic**: Neural Text-to-Speech (TTS) voice clones, diffusion vocoder footprints, acoustic pitch & formant continuity.
  - **Visual**: Facial reenactment, facial boundary replacement, neural mouth articulation.
  - **Cross-Modal**: Phoneme-to-viseme lip synchrony, audio-visual temporal lag offsets.

---

## 2. Institutional Access Request

The dataset is maintained by the Multimedia Security and Forensics Lab (NIT Rourkela / IIT Kharagpur).

### Request Procedure:
1. **Official Publication**:
   Consult the institutional repository or contact the corresponding authors via the Elsevier publication entry:
   - Corresponding Author: Dr. Ruchira Naskar (`ruchira@nitrkl.ac.in`)
2. **Formal EULA**:
   Submit a formal research inquiry specifying:
   - Primary Investigator (faculty name, department, university)
   - Institutional affiliation (`.ac.in` / `.edu`)
   - Explicit agreement not to re-distribute or use for non-consensual synthesis
3. **Directory Structure**:
   Extract dataset archives into:
   ```
   data/datasets/InDeepFake/
   ├── metadata.csv
   ├── hindi/
   │   ├── authentic/
   │   └── manipulated/
   ├── tamil/
   │   ├── authentic/
   │   └── manipulated/
   ├── telugu/
   │   ├── authentic/
   │   └── manipulated/
   └── bengali/
       ├── authentic/
       └── manipulated/
   ```

---

## 3. Running the Benchmark

```bash
# Run on official InDeepFake dataset:
py scripts/benchmark_indeepfake.py --dataset_dir data/datasets/InDeepFake

# Run on local samples (with explicit disclaimer):
py scripts/benchmark_indeepfake.py --sample_dir data/real_test_samples
```
