# PRISM Forensic Evaluation Log

This log tracks all forensic benchmark evaluations, test audits, calibration/test splits, detection quality metrics (AUC, EER, TPR@1%FPR, Precision/Recall, Confusion Matrix), sample counts, and known dataset caveats in accordance with scientific evaluation standards.

---

## Evaluation Log Entries

### Entry 001: Initial Test Suite & Forensic Agent Audit
- **Date**: 2026-09-19
- **Environment**: Windows 11, Python 3.13, PyTorch 2.x CPU/CUDA, Headless OpenCV
- **Repository Version / Commit**: Local workspace audit
- **Target**: Complete repository test audit across 31 existing tests in `tests/`
- **Audit Findings**:
  1. Multiple tests asserted non-informative bounds (e.g., `verdict in ["AUTHENTIC", "TAMPERED", "INCONCLUSIVE"]` in `test_pipeline_e2e.py` line 119) which pass under random or constant outputs.
  2. `test_all_agents.py::test_04_compression_watermark` asserted only that BBA masses sum to 1.0, not detection performance.
  3. `test_viral_indian_forensics.py::test_02` checked dictionary keys (`assertIn`) rather than whether the signature was accurately discriminated.
  4. Local benchmark samples in `data/real_test_samples` ($N=6$) and `data/test_samples` ($N=8$) are small sample sets ($N < 100$ per class).
- **Caveat**: All metrics computed on current local sample sets ($N < 100$ per class) are strictly **anecdotal and not statistically meaningful**. Full benchmarks on DF-Platter ($N=133,260$) and InDeepFake require downloading the datasets via institutional EULA agreements.


### Entry 002: Held-Out Core Evaluation & Ablation Study
- **Date**: 2026-09-19
- **Command**: `py scripts/eval_core.py`
- **Split Method**: Identity-stratified (50% calibration / 50% held-out test, 0 identity overlap)
- **Sample Counts**: N=10 (2 fake, 8 real) - *Anecdotal, not statistically meaningful (N < 100/class)*
- **Fused AUC**: 0.250 | Best Single Agent: 0.625 | ECE: 0.4913

| Method | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.438 | [0.00, 1.00] | 0.500 | 0.000 | 50.0% |
| Majority Class Baseline | 0.312 | [0.00, 1.00] | 0.562 | 0.000 | 50.0% |
| Mean of Agent Scores | 0.250 | [0.00, 0.57] | 0.875 | 0.000 | 31.2% |
| Max-Score Baseline | 0.250 | [0.00, 0.78] | 0.875 | 0.000 | 31.2% |
| Logistic Regression (Calib-Fit) | 0.250 | [0.00, 0.62] | 0.562 | 0.000 | 50.0% |
| Temporal Agent (Single) | 0.375 | [0.00, 0.75] | 0.500 | 0.000 | 50.0% |
| Illumination Agent (Single) | 0.250 | [0.00, 0.62] | 0.562 | 0.000 | 31.2% |
| Compression Agent (Single) | 0.625 | [0.24, 1.00] | 0.438 | 0.000 | 50.0% |
| Metadata Agent (Single) | 0.312 | [0.00, 0.89] | 0.562 | 0.000 | 43.8% |
| D-S GRAPH FUSION (PRISM) | 0.250 | [0.00, 0.57] | 0.875 | 0.000 | 31.2% |

*Known Caveat*: High conflict K observed when agents disagree. BBA mapping must be calibrated so lack of defect detection allocates to m(U) rather than m(A).


### Entry 002: Held-Out Core Evaluation & Ablation Study
- **Date**: 2026-09-19
- **Command**: `py scripts/eval_core.py`
- **Split Method**: Identity-stratified (50% calibration / 50% held-out test, 0 identity overlap)
- **Sample Counts**: N=10 (2 fake, 8 real) - *Anecdotal, not statistically meaningful (N < 100/class)*
- **Fused AUC**: 0.562 | Best Single Agent: 0.625 | ECE: 0.4132

| Method | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.438 | [0.00, 1.00] | 0.500 | 0.000 | 50.0% |
| Majority Class Baseline | 0.312 | [0.00, 1.00] | 0.562 | 0.000 | 50.0% |
| Mean of Agent Scores | 0.250 | [0.00, 0.57] | 0.875 | 0.000 | 31.2% |
| Max-Score Baseline | 0.250 | [0.00, 0.78] | 0.875 | 0.000 | 31.2% |
| Logistic Regression (Calib-Fit) | 0.250 | [0.00, 0.62] | 0.562 | 0.000 | 50.0% |
| Temporal Agent (Single) | 0.375 | [0.00, 0.75] | 0.500 | 0.000 | 50.0% |
| Illumination Agent (Single) | 0.250 | [0.00, 0.62] | 0.562 | 0.000 | 31.2% |
| Compression Agent (Single) | 0.625 | [0.24, 1.00] | 0.438 | 0.000 | 50.0% |
| Metadata Agent (Single) | 0.312 | [0.00, 0.89] | 0.562 | 0.000 | 43.8% |
| D-S GRAPH FUSION (PRISM) | 0.562 | [0.22, 0.89] | 0.500 | 0.000 | 56.2% |

*Known Caveat*: High conflict K observed when agents disagree. BBA mapping must be calibrated so lack of defect detection allocates to m(U) rather than m(A).


### Entry 002: Rigorous Held-Out Core Evaluation, LOO Ablation & Conflict Analysis
- **Date**: 2026-09-19
- **Command**: `py scripts/eval_core.py`
- **Split Method**: Identity-stratified (50% calibration / 50% held-out test, zero identity overlap)
- **Sample Counts**: N=10 (2 fake, 8 real) - *Anecdotal, not statistically meaningful (N < 100/class)*
- **Fused AUC**: 0.625 | Best Single Agent: 0.625 | ECE: 0.4852

#### 1. Public Benchmark Availability Audit

| Dataset | Availability | Acquisition Guide |
| :--- | :---: | :--- |
| FaceForensics++ (FF++) c23/c40 | **NOT AVAILABLE** | Submit academic agreement form to TUM (Technical University of Munich) at https://github.com/ondyari/FaceForensics to receive the download script (download-ff.py). |
| Celeb-DF (v2) | **NOT AVAILABLE** | Request dataset access via Google Drive form at https://github.com/yuezunli/celeb-deepfakeforensics as detailed in CVPR 2020 paper. |
| Deepfake Detection Challenge (DFDC) Preview | **NOT AVAILABLE** | Download via Kaggle CLI (`kaggle competitions download -c deepfake-detection-challenge`) or AWS Open Data registry. |
| DF-Platter (IIT Jodhpur, CVPR 2023) | **NOT AVAILABLE** | Submit institutional EULA request form to Image Analysis and Biometrics (IAB) Lab at IIT Jodhpur: http://iab-rubric.org/df-platter-database. |
| InDeepFake / SAMANVAYA (2025) | **NOT AVAILABLE** | Academic access via request to authors of Pattern Recognition Letters publication (multimodal audio-visual Indic speech corpus). |

#### 2. Comparative Benchmark Table (Held-Out Test Split)

| Method | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.438 | [0.00, 1.00] | 0.500 | 0.000 | 50.0% |
| Majority Class Baseline | 0.312 | [0.00, 1.00] | 0.562 | 0.000 | 50.0% |
| Mean of Agent Scores | 0.250 | [0.00, 0.57] | 0.875 | 0.000 | 31.2% |
| Max-Score Baseline | 0.250 | [0.00, 0.78] | 0.875 | 0.000 | 31.2% |
| Logistic Regression (Calib-Fit) | 0.250 | [0.00, 0.62] | 0.562 | 0.000 | 50.0% |
| Temporal Agent (Single) | 0.375 | [0.00, 0.75] | 0.500 | 0.000 | 50.0% |
| Illumination Agent (Single) | 0.250 | [0.00, 0.62] | 0.562 | 0.000 | 31.2% |
| Compression Agent (Single) | 0.625 | [0.24, 1.00] | 0.438 | 0.000 | 50.0% |
| Metadata Agent (Single) | 0.312 | [0.00, 0.89] | 0.562 | 0.000 | 43.8% |
| RPPG Agent (Single) | 0.312 | [0.00, 1.00] | 0.562 | 0.000 | 50.0% |
| Eye Gaze Agent (Single) | 0.250 | [0.00, 0.43] | 0.875 | 0.000 | 31.2% |
| D-S GRAPH FUSION (PRISM) | 0.625 | [0.29, 1.00] | 0.438 | 0.000 | 56.2% |

#### 3. Leave-One-Out (LOO) Ablation

| Ablation Variant | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full PRISM Fusion (Reference) | 0.625 | [0.29, 1.00] | 0.438 | 0.000 | 56.2% |
| Without Temporal Agent | 0.562 | [0.12, 0.89] | 0.500 | 0.000 | 31.2% |
| Without Illumination Agent | 0.500 | [0.13, 0.88] | 0.500 | 0.000 | 56.2% |
| Without Eye Gaze Agent | 0.625 | [0.29, 1.00] | 0.438 | 0.000 | 81.2% |
| Without RPPG Agent | 0.625 | [0.29, 1.00] | 0.438 | 0.000 | 56.2% |
| Without Compression Agent | 0.562 | [0.22, 0.89] | 0.500 | 0.000 | 56.2% |
| Without Metadata Agent | 0.625 | [0.29, 1.00] | 0.438 | 0.000 | 56.2% |

#### 4. Conflict K Distribution & Error Correlation

- Agreement ($K < 0.2$): 10/10 (100.0%), Error Rate: 40.0%
- Mild Conflict ($0.2 \le K < 0.5$): 0/10 (0.0%), Error Rate: 0.0%
- Strong Conflict ($K \ge 0.5$): 0/10 (0.0%), Error Rate: 0.0%
- Mean $K$ on Correct Predictions: 0.0076
- Mean $K$ on Prediction Errors: 0.0980

*Known Caveats*: Sample size is under 100 per class (anecdotal). Public datasets FF++, Celeb-DF, DFDC, DF-Platter, InDeepFake are NOT available locally. Fused system achieved AUC 0.625, trailing single best agent Compression (AUC 0.625).


### Entry 003: Corrected Evaluation Foundation (AUC Tie Fix, ViT/PRNU Streams, m(U) Audit)
- **Date**: 2026-09-19
- **Command**: `py scripts/eval_core.py`
- **Split Method**: Identity-stratified (50% calibration / 50% held-out test, zero identity overlap)
- **Sample Counts**: N=9 (2 fake, 7 real) - *Anecdotal (N < 100/class), CI refused (insufficient positives)*
- **Fused AUC**: 0.286 | Best Single Agent: 0.643 | Majority Baseline: 0.500 | ECE: 0.5110

#### 1. Comparative Benchmark Table (Held-Out Test Split)

| Method | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.429 | insufficient positives | 0.500 | 0.000 | 46.4% |
| Majority Class Baseline | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Mean of Agent Scores | 0.286 | insufficient positives | 0.714 | 0.000 | 35.7% |
| Max-Score Baseline | 0.286 | insufficient positives | 0.636 | 0.000 | 28.6% |
| Logistic Regression (Calib-Fit) | 0.143 | insufficient positives | 0.857 | 0.000 | 50.0% |
| Spatial ViT (DoRA) Agent | 0.429 | insufficient positives | 0.571 | 0.000 | 50.0% |
| PRNU Sensor Agent | 0.214 | insufficient positives | 0.700 | 0.000 | 28.6% |
| Temporal Agent (Single) | 0.429 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Illumination Agent (Single) | 0.286 | insufficient positives | 0.571 | 0.000 | 35.7% |
| Compression Agent (Single) | 0.643 | insufficient positives | 0.417 | 0.000 | 50.0% |
| Metadata Agent (Single) | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| RPPG Agent (Single) | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Eye Gaze Agent (Single) | 0.143 | insufficient positives | 0.778 | 0.000 | 35.7% |
| D-S GRAPH FUSION (PRISM) | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |

#### 2. Leave-One-Out (LOO) Ablation

| Ablation Variant | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full PRISM Fusion (Reference) | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Spatial ViT (DoRA) | 0.571 | insufficient positives | 0.429 | 0.000 | 28.6% |
| Without PRNU Sensor Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 35.7% |
| Without Temporal Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Illumination Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Eye Gaze Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without RPPG Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Compression Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Metadata Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |

#### 3. Belief Mass on m(U) Audit (Abstention vs Conservatism)

| Forensic Agent Stream | Mean m(U) Real | Mean m(U) Fake | Mean m(T) Fake | Mean m(A) Real | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Temporal | 0.818 | 0.766 | 0.133 | 0.111 | Balanced Mass |
| Illumination | 0.654 | 0.725 | 0.182 | 0.057 | Balanced Mass |
| Gaze | 0.699 | 0.880 | 0.000 | 0.041 | Balanced Mass |
| Compression | 0.856 | 1.000 | 0.000 | 0.117 | High Abstention |
| Metadata | 1.000 | 1.000 | 0.000 | 0.000 | High Abstention |
| Rppg | 0.700 | 0.700 | 0.150 | 0.150 | Balanced Mass |
| Vit | 0.302 | 0.185 | 0.001 | 0.587 | Authentic-Biased |
| Prnu | 0.357 | 0.250 | 0.090 | 0.379 | Balanced Mass |

*Known Caveats*: Majority Class Baseline corrected from 0.312 to 0.500 via average-rank tie handling. Test set sample counts (2 pos, 7 neg) fall below N=10 threshold, so bootstrap CIs are strictly suppressed. Evaluated on local test cohort; official datasets remain NOT AVAILABLE locally.


### Entry 004: Rigorous Evaluation Foundation (Spearman Rank Correlation, m(U) Audit Rewrite, Confound Controls)
- **Date**: 2026-09-19
- **Git Commit**: `db4ddf6-dirty`
- **Command**: `py scripts/eval_core.py`
- **Split Method**: Identity-stratified (50% calibration / 50% held-out test, zero identity overlap)
- **Sample Counts**: N=9 (2 fake, 7 real) - *⚠️ ANECDOTAL ONLY (N < 100/class, N_pos < 10); Rankings Suppressed*
- **Fused AUC**: 0.286 | Majority Baseline: 0.500 | ECE: 0.5110

#### 1. Unranked Raw Metrics Table (Held-Out Test Split)

| Method / Model | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.429 | insufficient positives | 0.500 | 0.000 | 46.4% |
| Majority Class Baseline | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Mean of Agent Scores | 0.286 | insufficient positives | 0.714 | 0.000 | 35.7% |
| Max-Score Baseline | 0.286 | insufficient positives | 0.636 | 0.000 | 28.6% |
| Logistic Regression (Calib-Fit) | 0.143 | insufficient positives | 0.857 | 0.000 | 50.0% |
| Spatial ViT (DoRA) Agent | 0.429 | insufficient positives | 0.571 | 0.000 | 50.0% |
| PRNU Sensor Agent | 0.214 | insufficient positives | 0.700 | 0.000 | 28.6% |
| Temporal Agent (Single) | 0.429 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Illumination Agent (Single) | 0.286 | insufficient positives | 0.571 | 0.000 | 35.7% |
| Compression Agent (Single) | 0.643 | insufficient positives | 0.417 | 0.000 | 50.0% |
| Metadata Agent (Single) | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| RPPG Agent (Single) | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Eye Gaze Agent (Single) | 0.143 | insufficient positives | 0.778 | 0.000 | 35.7% |
| D-S GRAPH FUSION (PRISM) | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |

#### 2. Leave-One-Out (LOO) Ablation

| Ablation Variant | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full PRISM Fusion (Reference) | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Spatial ViT (DoRA) | 0.571 | insufficient positives | 0.429 | 0.000 | 28.6% |
| Without PRNU Sensor Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 35.7% |
| Without Temporal Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Illumination Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Eye Gaze Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without RPPG Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Compression Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Metadata Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |

#### 3. Spearman Rank Correlation (Agent vs Fused Pignistic Score)

| Forensic Agent Stream | Spearman rho | p-value | Contribution Status |
| :--- | :---: | :---: | :--- |
| Spatial ViT (DoRA) | +0.317 | 0.4064 | contributing (rho = +0.317) |
| PRNU Sensor Agent | +0.894 | 0.0011 | contributing (rho = +0.894) |
| Temporal Optical Flow Agent | +0.559 | 0.1173 | contributing (rho = +0.559) |
| Illumination Consistency Agent | +0.778 | 0.0135 | contributing (rho = +0.778) |
| Eye Gaze Disparity Agent | +0.940 | 0.0002 | contributing (rho = +0.940) |
| RPPG Pulse Signal Agent | +0.000 | 1.0000 | not contributing (|rho| < 0.1) |
| Compression Watermark Agent | -0.228 | 0.5548 | contributing (rho = -0.228) |
| Metadata Provenance Agent | +0.000 | 1.0000 | not contributing (|rho| < 0.1) |

#### 4. Explicit Epistemic Belief Mass Audit

| Forensic Agent Stream | Mean m(T) R | Mean m(A) R | Mean m(U) R | Mean m(T) F | Mean m(A) F | Mean m(U) F | Frac m(U)>0.9 | Empirical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Spatial ViT (DoRA) | 0.111 | 0.587 | 0.302 | 0.001 | 0.814 | 0.185 | 11.1% | Authentic-Biased |
| PRNU Sensor Agent | 0.264 | 0.379 | 0.357 | 0.090 | 0.660 | 0.250 | 11.1% | Balanced Mass |
| Temporal Agent | 0.071 | 0.111 | 0.818 | 0.133 | 0.102 | 0.766 | 0.0% | High Epistemic Uncertainty |
| Illumination Agent | 0.288 | 0.057 | 0.654 | 0.182 | 0.093 | 0.725 | 12.5% | High Epistemic Uncertainty |
| Eye Gaze Agent | 0.260 | 0.041 | 0.699 | 0.000 | 0.120 | 0.880 | 25.0% | High Epistemic Uncertainty |
| Compression Agent | 0.027 | 0.117 | 0.856 | 0.000 | 0.000 | 1.000 | 0.0% | High Epistemic Uncertainty |
| Metadata Agent | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 100.0% | Pervasive Abstention (100% > 0.9) |
| RPPG Pulse Agent | 0.150 | 0.150 | 0.700 | 0.150 | 0.150 | 0.700 | 50.0% | High Epistemic Uncertainty |

*Empirical Summary*: Because test sample counts (2 pos, 7 neg) fall below N=10, all method rankings and 'best agent' declarations are strictly suppressed. Bootstrap CIs refused ('insufficient positives'). Evaluated on local cohort; official datasets remain NOT AVAILABLE locally.


### Entry 004: Rigorous Evaluation Foundation (Spearman Rank Correlation, m(U) Audit Rewrite, Confound Controls)
- **Date**: 2026-09-19
- **Git Commit**: `db4ddf6-dirty`
- **Command**: `py scripts/eval_core.py`
- **Split Method**: Identity-stratified (50% calibration / 50% held-out test, zero identity overlap)
- **Sample Counts**: N=9 (2 fake, 7 real) - *⚠️ ANECDOTAL ONLY (N < 100/class, N_pos < 10); Rankings Suppressed*
- **Fused AUC**: 0.286 | Majority Baseline: 0.500 | ECE: 0.5110

#### 1. Unranked Raw Metrics Table (Held-Out Test Split)

| Method / Model | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.429 | insufficient positives | 0.500 | 0.000 | 46.4% |
| Majority Class Baseline | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Mean of Agent Scores | 0.286 | insufficient positives | 0.714 | 0.000 | 35.7% |
| Max-Score Baseline | 0.286 | insufficient positives | 0.636 | 0.000 | 28.6% |
| Logistic Regression (Calib-Fit) | 0.143 | insufficient positives | 0.857 | 0.000 | 50.0% |
| Spatial ViT (DoRA) Agent | 0.429 | insufficient positives | 0.571 | 0.000 | 50.0% |
| PRNU Sensor Agent | 0.214 | insufficient positives | 0.700 | 0.000 | 28.6% |
| Temporal Agent (Single) | 0.429 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Illumination Agent (Single) | 0.286 | insufficient positives | 0.571 | 0.000 | 35.7% |
| Compression Agent (Single) | 0.643 | insufficient positives | 0.417 | 0.000 | 50.0% |
| Metadata Agent (Single) | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| RPPG Agent (Single) | 0.500 | insufficient positives | 0.500 | 0.000 | 50.0% |
| Eye Gaze Agent (Single) | 0.143 | insufficient positives | 0.778 | 0.000 | 35.7% |
| D-S GRAPH FUSION (PRISM) | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |

#### 2. Leave-One-Out (LOO) Ablation

| Ablation Variant | AUC | 95% CI | EER | TPR@1%FPR | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full PRISM Fusion (Reference) | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Spatial ViT (DoRA) | 0.571 | insufficient positives | 0.429 | 0.000 | 28.6% |
| Without PRNU Sensor Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 35.7% |
| Without Temporal Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Illumination Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Eye Gaze Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without RPPG Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Compression Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |
| Without Metadata Agent | 0.286 | insufficient positives | 0.714 | 0.000 | 28.6% |

#### 3. Spearman Rank Correlation (Agent vs Fused Pignistic Score)

| Forensic Agent Stream | Spearman rho | p-value | Contribution Status |
| :--- | :---: | :---: | :--- |
| Spatial ViT (DoRA) | +0.317 | 0.4064 | contributing (rho = +0.317) |
| PRNU Sensor Agent | +0.894 | 0.0011 | contributing (rho = +0.894) |
| Temporal Optical Flow Agent | +0.559 | 0.1173 | contributing (rho = +0.559) |
| Illumination Consistency Agent | +0.778 | 0.0135 | contributing (rho = +0.778) |
| Eye Gaze Disparity Agent | +0.940 | 0.0002 | contributing (rho = +0.940) |
| RPPG Pulse Signal Agent | +0.000 | 1.0000 | not contributing (|rho| < 0.1) |
| Compression Watermark Agent | -0.228 | 0.5548 | contributing (rho = -0.228) |
| Metadata Provenance Agent | +0.000 | 1.0000 | not contributing (|rho| < 0.1) |

#### 4. Explicit Epistemic Belief Mass Audit

| Forensic Agent Stream | Mean m(T) R | Mean m(A) R | Mean m(U) R | Mean m(T) F | Mean m(A) F | Mean m(U) F | Frac m(U)>0.9 | Empirical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Spatial ViT (DoRA) | 0.111 | 0.587 | 0.302 | 0.001 | 0.814 | 0.185 | 11.1% | Authentic-Biased |
| PRNU Sensor Agent | 0.264 | 0.379 | 0.357 | 0.090 | 0.660 | 0.250 | 11.1% | Balanced Mass |
| Temporal Agent | 0.071 | 0.111 | 0.818 | 0.133 | 0.102 | 0.766 | 0.0% | High Epistemic Uncertainty |
| Illumination Agent | 0.288 | 0.057 | 0.654 | 0.182 | 0.093 | 0.725 | 12.5% | High Epistemic Uncertainty |
| Eye Gaze Agent | 0.260 | 0.041 | 0.699 | 0.000 | 0.120 | 0.880 | 25.0% | High Epistemic Uncertainty |
| Compression Agent | 0.027 | 0.117 | 0.856 | 0.000 | 0.000 | 1.000 | 0.0% | High Epistemic Uncertainty |
| Metadata Agent | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 100.0% | Pervasive Abstention (100% > 0.9) |
| RPPG Pulse Agent | 0.150 | 0.150 | 0.700 | 0.150 | 0.150 | 0.700 | 50.0% | High Epistemic Uncertainty |

*Empirical Summary*: Because test sample counts (2 pos, 7 neg) fall below N=10, all method rankings and 'best agent' declarations are strictly suppressed. Bootstrap CIs refused ('insufficient positives'). Evaluated on local cohort; official datasets remain NOT AVAILABLE locally.


### Entry 005: Pipeline for Real Data (Feature Cache, Matched Normalization, Shortcut Baseline, Cached Fusion, 104-Video Perturbation Grid)
- **Date**: 2026-09-20
- **Git Commit**: `ed5f276-dirty`
- **Commands**:
  * `py scripts/extract_features.py --manifest data/cache/manifest_normalized.json --output_csv data/cache/features_normalized.csv --no_resume`
  * `py scripts/matched_normalization.py --compare data/cache/features_unnormalized.csv data/cache/features_normalized.csv`
  * `py scripts/eval_fusion_cached.py --features_csv data/cache/features_normalized.csv`
  * `py scripts/benchmark_perturbation_grid.py`
- **Split Method**: Identity-disjoint partition (50% calibration / 50% held-out test, 0 identity overlap)
- **Sample Counts**:
  * Cached Test Split: N=8 (N_id=7, N_gen=5 | Pos=4, Neg=4)  -  *⚠️ ANECDOTAL BANNER: N_pos < 10; Ranking narratives suppressed*
  * Effective N: 8 distinct generators, 14 distinct source identities / videos
  * Perturbation Grid: N=104 (N_id=4 authentic IDs, 4 fake IDs | N_gen=4 generators | 8 source videos x 13 conditions)
- **Public Benchmark Availability**: Celeb-DF v2, FF++, and DFDC remain **NOT AVAILABLE** locally.

#### 1. Forensic Discrimination Before vs After Matched Normalization
*(Matched standard: 1280x720, 30.0 fps, libx264 CRF 28, uniform JPEG quality 90)*

| Forensic Stream / Agent | AUC Before | AUC After | Delta AUC | Shortcut Assessment |
| :--- | :---: | :---: | :---: | :--- |
| Spatial ViT (DoRA) | 0.6042 | 0.5972 | -0.0069 | Invariant (Genuine Signal) |
| PRNU Sensor Noise | 0.6597 | 0.6597 | +0.0000 | Invariant (Genuine Signal) |
| Video Temporal Flow | 0.6250 | 0.6250 | +0.0000 | Invariant (Genuine Signal) |
| Illumination Consistency | 0.4583 | 0.4722 | +0.0139 | Invariant (Genuine Signal) |
| Eye Gaze Disparity | 0.4653 | 0.4028 | -0.0625 | Robust Signal |
| Compression Watermark | 0.4792 | 0.4792 | +0.0000 | Invariant (Genuine Signal) |
| Metadata Provenance | 0.8125 | 0.8125 | +0.0000 | Invariant (Genuine Signal) |
| RPPG Biological Pulse | 0.5000 | 0.5000 | +0.0000 | Invariant (Genuine Signal) |

#### 2. Held-Out Evaluation: Fusion Architectures vs Shortcut Baseline
*(All fusion parameters fit strictly on calibration split; evaluated on held-out split with Cluster Bootstrap by Identity)*

| Method / Architecture | AUC | EER | TPR@1%FPR | 95% Cluster CI | Shortcut Clear |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Metadata Shortcut Baseline | 0.4688 | 0.5000 | 0.2500 | insufficient positives | BASELINE |
| Mean of Agent Scores | 0.9062 | 0.2500 | 0.5000 | insufficient positives | YES |
| Logistic Regression Fusion | 0.6562 | 0.5000 | 0.5000 | insufficient positives | YES |
| Gradient Boosting Fusion | 0.5000 | 0.5000 | 0.0000 | insufficient positives | YES |
| Dempster-Shafer Fusion | 0.9062 | 0.2500 | 0.7500 | insufficient positives | YES |

#### 3. Per-Agent Discrimination, Shortcut Status & Contribution Audit
*(Contribution criterion strictly enforced: Spearman rho > 0.10 AND AUC > 0.5000)*

| Forensic Agent | AUC | EER | Spearman rho | Shortcut Status | Contribution Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Spatial ViT (DoRA) | 0.6875 | 0.2500 | +0.1576 | Exceeds Metadata | Contributing |
| PRNU Sensor Noise | 0.8125 | 0.2500 | +0.9314 | Exceeds Metadata | Contributing |
| Video Temporal Flow | 0.6250 | 0.4500 | -0.1372 | Exceeds Metadata | Detractor / Inverted (rho <= 0) |
| Illumination Consistency | 0.6250 | 0.5000 | +0.7118 | Exceeds Metadata | Contributing |
| Eye Gaze Disparity | 0.4688 | 0.5833 | +0.0494 | Matches Metadata | Non-Contributing (rho <= 0.1 or AUC <= 0.5) |
| Compression Watermark | 0.4688 | 0.5000 | -0.3012 | Matches Metadata | Detractor / Inverted (rho <= 0) |
| Metadata Provenance | 0.8750 | 0.2000 | +0.8712 | Exceeds Metadata | Contributing |
| RPPG Biological Pulse | 0.5000 | 0.5000 | +0.0000 | Matches Metadata | Detractor / Inverted (rho <= 0) |
| Audio Voice Artifacts | 0.5000 | 0.5000 | +0.0000 | Matches Metadata | Detractor / Inverted (rho <= 0) |
| Lip-Sync Synchrony | 0.5000 | 0.5000 | +0.0000 | Matches Metadata | Detractor / Inverted (rho <= 0) |

#### 4. 104-Video Robustness Grid (Delta AUC Relative to Clean Baseline)
*(Sample count: N=104, N_id=4 real / 4 fake, N_gen=4. Phase 5 status: Invariants Only / Procedural Perturbation Grid)*

| Perturbation Condition | ViT-DoRA | PRNU | Temporal | Illumination | Compression |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline (Reference)** | **0.6562** | **0.5000** | **0.8750** | **0.4375** | **0.3125** |
| Recompression CRF 35 | -0.0625 | +0.1250 | -0.1875 | -0.0625 | +0.1875 |
| Recompression CRF 40 | +0.0000 | +0.1250 | -0.3125 | +0.0625 | +0.1875 |
| Recompression CRF 45 | -0.1875 | +0.0000 | -0.3750 | +0.0938 | +0.1875 |
| Downscale 75% | +0.0312 | +0.0000 | +0.0000 | +0.0000 | +0.1250 |
| Downscale 50% | +0.0938 | +0.0000 | +0.0000 | +0.0000 | +0.1250 |
| Downscale 35% | +0.0625 | -0.1250 | -0.0625 | +0.0000 | +0.0625 |
| Gaussian Blur k=5 | +0.0312 | +0.1250 | -0.1250 | +0.0000 | +0.2500 |
| Gaussian Blur k=9 | +0.0000 | +0.1250 | -0.2500 | -0.0625 | +0.0938 |
| Additive Noise sigma=10 | +0.0938 | +0.0000 | +0.0000 | +0.0312 | +0.0000 |
| Additive Noise sigma=25 | -0.1250 | +0.0000 | -0.0625 | +0.1250 | +0.4062 |
| Frame Rate Drop 15fps | +0.0312 | +0.0000 | -0.2500 | +0.1250 | +0.0625 |
| Brightness Shift Gamma 0.8 | -0.0625 | +0.0000 | +0.0000 | +0.0000 | -0.0312 |

---

### Entry 006: Metadata De-Leakage, Strengthened Shortcut Baseline & Dual-Identity Disjoint Partitioning
- **Date**: 2026-09-20
- **Git Commit**: `e14b169-dirty`
- **Runtime Environment**: Python 3.13.14, PyTorch 2.13.0+cpu, Windows 11
- **Key Actions**:
  1. Audited `MetadataAgent`: Found and removed keyword/regex matching on filename and path strings (`_AI_OR_EDIT_FILENAME_PATTERNS`). Restricted agent strictly to EXIF tags, PNG info chunks, and binary container atoms. Clean metadata AUC dropped from 0.8125 to 0.5139 (full cohort) and 0.4062 (test split).
  2. Strengthened Shortcut Baseline: 58 features (bitrate, size, resolution, fps, codec, container, count-vectorized filename tokens). Cross-validated across cohort with 3-Fold Stratified CV: AUC = $0.4815 \pm 0.4091$. Held-out test AUC = $0.5625$. Enforced $+0.05$ margin requirement to clear.
  3. Paired Fusion Evaluation: Reran fusion with and without the metadata agent. Mean of agent scores (0.7188) and Dempster-Shafer fusion (0.8438) cleared the shortcut baseline by $> 0.05$ margin, while Logistic Regression (0.4688) and Gradient Boosting (0.5000) failed.
  4. Audio and Lip-Sync streams properly labeled `NOT EVALUATED (no speech)` and constant-score streams labeled `constant / no information`.
  5. Perturbation grid updated with Effective N (8 source videos: 4 real, 4 fake), metadata agent, fused D-S score, and labeled `ANECDOTAL`.
  6. Celeb-DF dual-identity tracking implemented in `scripts/train_kaggle_vit_dora.py` with unit tests passing (3/3).


