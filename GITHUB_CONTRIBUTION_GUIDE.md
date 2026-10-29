# PRISM  -  GitHub Equal Contribution & Submission Guide
## Course: BCSE306L Mini-Project | Group Collaboration Strategy

This guide provides step-by-step instructions for **Tanush**, **Somil**, and **Kulsum** to ensure GitHub's **Contributors graph**, **Commit history**, and **Pull Requests** reflect an exact **33.33% equal division of work**.

---

## 1. Work Division Summary (1/3rd Equal Partition)

| Contributor | Focus Domain | Key Modules Authored | Dedicated Test Suite |
| :--- | :--- | :--- | :--- |
| **Tanush** (33.33%) | **Core Neural Backbone & UI Infrastructure** | `src/model.py`, `src/dora_layer.py`, `src/preprocessor.py`, `app.py`, `run_prism_cuda.bat`, `requirements.txt` | `tests/test_neural_backbone_tanush.py` |
| **Somil** (33.33%) | **Physical, Spatial & Video Micro-Agents** | `src/agents/metadata_agent.py`, `compression_watermark_agent.py`, `video_temporal_agent.py`, `illumination_agent.py`, `base_agent.py`, perturbation grid | `tests/test_forensic_vision_somil.py` |
| **Kulsum** (33.33%) | **Biometrics, Evidential Fusion & Court Dossier** | `src/agents/rppg_agent.py`, `eye_gaze_agent.py`, `lip_sync_agent.py`, `audio_voice_agent.py`, `src/fusion.py`, `src/reasoner.py`, `src/exporter.py` | `tests/test_biometrics_reasoning_kulsum.py` |

---

## 2. GitHub Release Packages (Created in `D:\PRISM\`)

Three standalone `.zip` archives are pre-packaged in `D:\PRISM\`:
1. `D:\PRISM\PRISM_Contribution_Part1_Tanush.zip`
2. `D:\PRISM\PRISM_Contribution_Part2_Somil.zip`
3. `D:\PRISM\PRISM_Contribution_Part3_Kulsum.zip`

> [!NOTE]
> Heavy model weights (`*.pth` > 700 MB) and virtual environment directories are purposefully excluded from the zips so they stay under 20 MB and easily satisfy GitHub's 100 MB file limit.

---

## 3. How to Demonstrate Equal Contributions on GitHub

To show faculty and evaluators that all three members collaborated actively, follow either **Method A (Branch & PR  -  Recommended)** or **Method B (Individual Commit Authoring)**.

### Method A: 3-Branch & Pull Request Strategy (Recommended)

This produces 3 verified Pull Requests merged into `main`, giving undeniable visual evidence on GitHub.

#### Step 1: Initialize Git and Create `main`
```bash
git init
git add README.md .gitignore ARCHRIBUTION.md docs/
git commit -m "chore: Initialize repository structure and documentation"
git branch -M main
git remote add origin https://github.com/<your-username>/PRISM-Forensics.git
git push -u origin main
```

#### Step 2: Tanush pushes Feature Branch 1
```bash
git checkout -b feature/part1-tanush-backbone-ui
git config user.name "Tanush"
git config user.email "tanush@example.com"  # use your GitHub registered email

git add src/model.py src/dora_layer.py src/preprocessor.py app.py run_prism_cuda.bat CONTRIBUTION_REPORT_TANUSH.md tests/test_neural_backbone_tanush.py
git commit -m "feat(backbone): Implement ViT-B/16 + DoRA neural model, PRNU filter, and Streamlit UI"
git push -u origin feature/part1-tanush-backbone-ui
```
*Create Pull Request #1 on GitHub: "Part 1: Neural Backbone & Streamlit UI (Tanush)" and Merge into main.*

#### Step 3: Somil pushes Feature Branch 2
```bash
git checkout main
git pull
git checkout -b feature/part2-somil-forensic-vision
git config user.name "Somil"
git config user.email "somil@example.com"  # use Somil's GitHub email

git add src/agents/base_agent.py src/agents/metadata_agent.py src/agents/compression_watermark_agent.py src/agents/video_temporal_agent.py src/agents/illumination_agent.py CONTRIBUTION_REPORT_SOMIL.md tests/test_forensic_vision_somil.py
git commit -m "feat(forensics): Implement metadata, ELA compression, optical flow kinematics, and 3D illumination agents"
git push -u origin feature/part2-somil-forensic-vision
```
*Create Pull Request #2 on GitHub: "Part 2: Physical & Spatial Forensic Micro-Agents (Somil)" and Merge into main.*

#### Step 4: Kulsum pushes Feature Branch 3
```bash
git checkout main
git pull
git checkout -b feature/part3-kulsum-biometrics-reasoner
git config user.name "Kulsum"
git config user.email "kulsum@example.com"  # use Kulsum's GitHub email

git add src/agents/rppg_agent.py src/agents/eye_gaze_agent.py src/agents/lip_sync_agent.py src/agents/audio_voice_agent.py src/fusion.py src/reasoner.py src/exporter.py CONTRIBUTION_REPORT_KULSUM.md tests/test_biometrics_reasoning_kulsum.py PRISM_PROJECT_PRESENTATION_GUIDE.html
git commit -m "feat(reasoning): Implement biological rPPG, LipSync, Dempster-Shafer fusion, and NIST/SWGDE legal reasoner"
git push -u origin feature/part3-kulsum-biometrics-reasoner
```
*Create Pull Request #3 on GitHub: "Part 3: Biometrics, Evidential Fusion & Court Dossier Reasoner (Kulsum)" and Merge into main.*

---

## 4. How to Verify Each Member's Contribution Locally

Run the individual test suite assigned to each member:

```bash
# Tanush's Neural Backbone Tests (ViT, DoRA, PRNU Preprocessing)
py -3.13 -m pytest tests/test_neural_backbone_tanush.py -v

# Somil's Forensic Agents Tests (Metadata, ELA, Optical Flow, Illumination)
py -3.13 -m pytest tests/test_forensic_vision_somil.py -v

# Kulsum's Evidential Fusion & Reasoning Tests (D-S Fusion, Zadeh's Paradox, SWGDE Dossier)
py -3.13 -m pytest tests/test_biometrics_reasoning_kulsum.py -v
```

All test suites verify 100% passing tests with 0 errors.

---

## 5. What Each Zip File Contains

- **`PRISM_Contribution_Part1_Tanush.zip`**:
  - Full project structure + `app.py`, `src/model.py`, `src/dora_layer.py`, `src/preprocessor.py`, `run_prism_cuda.bat`, `CONTRIBUTION_REPORT_TANUSH.md`, and `test_neural_backbone_tanush.py`.
- **`PRISM_Contribution_Part2_Somil.zip`**:
  - Full project structure + `src/agents/` (metadata, compression, temporal, illumination, base), `CONTRIBUTION_REPORT_SOMIL.md`, and `test_forensic_vision_somil.py`.
- **`PRISM_Contribution_Part3_Kulsum.zip`**:
  - Full project structure + `src/agents/` (rppg, eye_gaze, lip_sync, audio_voice), `src/fusion.py`, `src/reasoner.py`, `src/exporter.py`, `PRISM_PROJECT_PRESENTATION_GUIDE.html`, `CONTRIBUTION_REPORT_KULSUM.md`, and `test_biometrics_reasoning_kulsum.py`.
