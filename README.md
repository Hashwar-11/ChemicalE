# Historian-Based Fault Detection & Root Cause Analysis (TK-101 Hold Tank)

**Machine Learning Pipeline Project — Department of Chemical Engineering (PIEAS)**  
**Process System:** Steam-Heated Agitated Hydrolysis Hold Tank (`TK-101`) with Master-Slave Cascade Temperature (`TIC-101` → `FIC-201`) and Inventory Level Control (`LIC-101`)

---

## 1. Project Overview & Dataset Split (70% / 15% / 15%)

The project investigates 90 days of 1-minute industrial historian data (`historian_60s (1).csv`, **129,600 rows × 91 columns**: `timestamp` + 90 numeric sensor/controller tags from `2024-03-01 06:00:00 UTC` to `2024-05-30 05:59:00 UTC`). Because no ground-truth fault labels are supplied, the pipeline uses **physics-informed feature engineering** combined with **unsupervised anomaly detection** and a **chronological 70/15/15 split** to prevent future-to-past data leakage.

| Split | Ratio | Rows | Duration | Start Timestamp (UTC) | End Timestamp (UTC) | Missing Cells | Rows w/ Missing | File Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training** | 70% | 90,720 | 63.0 days | `2024-03-01 06:00:00+00:00` | `2024-05-03 05:59:00+00:00` | 10,760 | 715 | `training/splits/train_70.csv` |
| **Validation** | 15% | 19,440 | 13.5 days | `2024-05-03 06:00:00+00:00` | `2024-05-16 17:59:00+00:00` | 3,495 | 255 | `training/splits/val_15.csv` |
| **Testing** | 15% | 19,440 | 13.5 days | `2024-05-16 18:00:00+00:00` | `2024-05-30 05:59:00+00:00` | 2,460 | 123 | `training/splits/test_15.csv` |
| **Total** | **100%** | **129,600** | **90.0 days** | `2024-03-01 06:00:00+00:00` | `2024-05-30 05:59:00+00:00` | **16,715** | **1,093** | `historian_60s (1).csv` |

### Physics-Based Feature Engineering & Selection (Milestones 5 & 6)
- **90 Raw Historian Tags** → **+18 First-Principles Physics Features** (**108 total variables**):
  - Loop tracking errors: `ERR_TIC-101`, `ERR_LIC-101`, `ERR_FIC-201` ($\text{PV} - \text{SP}$)
  - Energy balance residual: `RES_DUTY = CALC-DUTY - CALC-DUTY-PROC`
  - Coil `E-101` thermal fouling indicator: `UA_ROLLING_DIFF` (`CALC-UA` minus its 1440-min rolling mean)
  - Actuator stiction discrepancies: `VALVE_DISC_TV101` (`TV-101.POS - FIC-201.OP`), `VALVE_DISC_LV101` (`LV-101.POS - LIC-101.OP`)
  - Steam trap subcooling & tank thermal differentials: `DIFF_TI_201_202`, `DIFF_TANK_IN_OUT`, `DIFF_CORE_EXIT_TEMP`
  - 15-minute rolling sensor volatilities (`*_ROLL_STD`) for 8 key process and controller tags
- **Feature Selection:** Removed discrete `.MODE` status integers, VIF-collinear raw tags (`CALC-DUTY-PROC`, `TI-100`), and uncoupled neighboring utility skids (`CW-*`, `N2-*`, `SD-*`, `AIR-*`), yielding **40 core TK-101 process & physics features**.
- **Leakage-Free Preprocessing & Alarm Filtering:** Median imputation, `StandardScaler`, and **99th-percentile Upper Control Limits ($\text{Baseline FAR} = 1.00\%$)** are fitted strictly on the **70% Training split**. Raw alarm spikes are cleaned via a **2-minute spike filter** and **15-minute recovery gap merger**.

---

## 2. Overall Model Comparison Leaderboard

All models were trained on the **70% Training set (`90,720` rows)**, validated on the **15% Validation set (`19,440` rows)**, and evaluated on the unseen **15% Testing set (`19,440` rows)**. Each model stores its own self-contained checkpoint (`.npz`), evaluation script, and score files inside its dedicated directory under `training/models/`.

| Rank | Model Name | Architecture Family | Best Epoch / Stage | Train FAR | Val ROC-AUC | Val F1 | Test ROC-AUC | Test Precision | Test Recall | **Test F1** | **Test Accuracy** | Test SNR | Test Infer Time | **Composite Score** |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Winner)** | **PCA Multivariate Monitor ($T^2 + Q$-Stat)** | Subspace Projection & Residual SPC | `Epoch 13/19 (k=23, w=0.95)` | **1.00%** | 0.9733 | 0.8771 | 0.9670 | **0.9854** | 0.8675 | 0.9227 | 91.67% | 9.57× | **43.57 ms** | **94.95 / 100** |
| **2** | **LSTM Autoencoder (`SeqLen=10, Hidden=20`)** | Deep Temporal Sequence Reconstruction | `Saved Epoch 09 (MSE=0.07925)` | **1.00%** | 0.9562 | 0.8929 | **0.9743** | 0.9710 | **0.8874** | **0.9273** | **92.03%** | **54.98×** | 543.68 ms | **94.32 / 100** |
| **3** | **Local Outlier Factor (`LOF, k=20 Novelty`)** | Local Density $k$-NN Anomaly Detection | `Epoch 05/10 (k=20 NN)` | **1.00%** | **0.9823** | **0.9106** | 0.9608 | 0.9675 | 0.8415 | 0.9001 | 89.30% | 2.80× | 2553.39 ms | **92.18 / 100** |
| **4** | **Isolation Forest (`150 Trees`)** | Tree-Based Random Partitioning | `Epoch 10/10 (150 Trees)` | **1.00%** | 0.9262 | 0.0095 | 0.8599 | **0.9973** | 0.0981 | 0.1787 | 48.31% | 1.18× | 180.68 ms | **67.11 / 100** |

---

## 3. Detailed Results by Model

### 3.1 PCA Multivariate Monitor (Hotelling's $T^2$ + $Q$-Statistic / SPE) — **Rank #1 (`94.95 / 100`)**
- **Directory:** `training/models/pca_t2_q/`
- **Saved Checkpoint:** `training/models/pca_t2_q/pca_t2_q_checkpoint.npz`
- **Test & Evaluate Script:** `training/models/pca_t2_q/test_and_evaluate_pca_t2_q.py`
- **Saved Score Files:** `pca_t2_q_scores.json`, `pca_t2_q_test_scores.csv`, `pca_t2_q_candidate_events.csv`
- **Best Configuration:** `k = 23` Principal Components retaining **97.43% cumulative process variance** (`t2_weight = 0.95`, 99% threshold = `1.0757`, baseline FAR = `1.00%`)
- **Validation (15%) Performance:**
  - `ROC-AUC = 0.9733` | `F1-Score = 0.8771` | `Precision = 0.9998` | `Recall = 0.7813` | `Accuracy = 93.40%` (`6` candidate fault events)
- **Testing (15%) Performance:**
  - `ROC-AUC = 0.9670` | `F1-Score = 0.9227` | `Precision = 0.9854` | `Recall = 0.8675` | `Accuracy = 91.67%` | `SNR = 9.57×` (`11` consolidated fault events, `43.57 ms` inference)
- **Engineering Justification:** Combines **Hotelling's $T^2$** (in-model operating shifts such as steam header swings and gradual coil fouling) with the **$Q$-statistic / SPE** (residual conservation-law breaks such as valve stiction and steam trap blow-through), while offering **instant analytical tag-level root-cause contributions** (`PI-201`, `TI-201`, `CALC-APPROACH`, `FI-101`, `DIFF_TANK_IN_OUT`, `CALC-MBAL`, `ERR_FIC-201`, `UA_ROLLING_DIFF`) and the fastest inference speed (**12× faster than LSTM** and **58× faster than LOF**).

---

### 3.2 LSTM Autoencoder (`SeqLen=10`, `Hidden=20`) — **Rank #2 (`94.32 / 100`, Highest Test F1 & Accuracy)**
- **Directory:** `training/models/lstm_autoencoder/`
- **Saved Checkpoint:** `training/models/lstm_autoencoder/lstm_autoencoder_checkpoint.npz`
- **Test & Evaluate Script:** `training/models/lstm_autoencoder/test_and_evaluate_lstm_autoencoder.py`
- **Saved Score Files:** `lstm_autoencoder_scores.json`, `lstm_autoencoder_test_scores.csv`, `lstm_autoencoder_candidate_events.csv`
- **Best Configuration:** 10-minute recurrent sliding window (`seq_len = 10`, `hidden_dim = 20`) trained via mini-batch Adam optimizer (`MSE = 0.07925` at Best Epoch 09, 99% threshold = `0.2146`, baseline FAR = `1.00%`)
- **Validation (15%) Performance:**
  - `ROC-AUC = 0.9562` | `F1-Score = 0.8929` | `Precision = 0.9975` | `Recall = 0.8082` | `Accuracy = 94.15%` (`6` candidate fault events)
- **Testing (15%) Performance:**
  - `ROC-AUC = 0.9743` | `F1-Score = 0.9273` | `Precision = 0.9710` | `Recall = 0.8874` | `Accuracy = 92.03%` | `SNR = 54.98×` (`12` consolidated fault events, `543.68 ms` inference)
- **Engineering Justification:** Achieves the **highest raw Test F1-Score (`0.9273`)**, **highest Test Accuracy (`92.03%`)**, **highest Test Recall (`0.8874`)**, **highest Test ROC-AUC (`0.9743`)**, and an exceptional **54.98× Signal-to-Noise Ratio** because its recurrent memory explicitly models multi-minute thermal and transport lags in `TK-101`.

---

### 3.3 Local Outlier Factor (`LOF, k=20 Novelty Mode`) — **Rank #3 (`92.18 / 100`, Highest Validation F1)**
- **Directory:** `training/models/local_outlier_factor/`
- **Saved Checkpoint:** `training/models/local_outlier_factor/lof_checkpoint.npz`
- **Test & Evaluate Script:** `training/models/local_outlier_factor/test_and_evaluate_lof.py`
- **Saved Score Files:** `local_outlier_factor_scores.json`, `local_outlier_factor_test_scores.csv`, `local_outlier_factor_candidate_events.csv`
- **Best Configuration:** `k = 20` nearest neighbors over a 3,024-state chronological reference manifold (99% threshold = `1.5281`, baseline FAR = `1.00%`)
- **Validation (15%) Performance:**
  - `ROC-AUC = 0.9823` | `F1-Score = 0.9106` | `Precision = 0.9976` | `Recall = 0.8375` | `Accuracy = 95.04%` (`9` candidate fault events)
- **Testing (15%) Performance:**
  - `ROC-AUC = 0.9608` | `F1-Score = 0.9001` | `Precision = 0.9675` | `Recall = 0.8415` | `Accuracy = 89.30%` | `SNR = 2.80×` (`17` consolidated fault events, `2553.39 ms` inference)
- **Engineering Justification:** Achieves the **highest Validation F1-Score (`0.9106`)** and **Validation ROC-AUC (`0.9823`)** by comparing local reachability density against neighboring normal states, though pairwise $k$-NN distance evaluation makes it slower at inference (`2.55 s`).

---

### 3.4 Isolation Forest (`150 Trees`) — **Rank #4 (`67.11 / 100`, High-Precision Point-Spike Detector)**
- **Directory:** `training/models/isolation_forest/`
- **Saved Checkpoint:** `training/models/isolation_forest/isolation_forest_checkpoint.npz`
- **Test & Evaluate Script:** `training/models/isolation_forest/test_and_evaluate_isolation_forest.py`
- **Saved Score Files:** `isolation_forest_scores.json`, `isolation_forest_test_scores.csv`, `isolation_forest_candidate_events.csv`
- **Best Configuration:** `n_estimators = 150` random isolation trees (`max_samples = 256`, 99% threshold = `0.5469`, baseline FAR = `1.00%`)
- **Validation (15%) Performance:**
  - `ROC-AUC = 0.9262` | `F1-Score = 0.0095` | `Precision = 1.0000` | `Recall = 0.0048` | `Accuracy = 69.97%` (`4` candidate fault events)
- **Testing (15%) Performance:**
  - `ROC-AUC = 0.8599` | `F1-Score = 0.1787` | `Precision = 0.9973` | `Recall = 0.0981` | `Accuracy = 48.31%` | `SNR = 1.18×` (`26` consolidated fault events, `180.68 ms` inference)
- **Engineering Justification:** Exhibits very high precision (`0.9973` on Test, `1.0000` on Validation) and good ranking (`Test ROC-AUC = 0.8599`), but when locked to a strict 99th-percentile training threshold (`1.00%` FAR), axis-aligned random cuts only flag extreme point spikes (`Test Recall = 0.0981`) and miss collinear mass/energy balance shifts and gradual coil fouling.

---

## 4. Repository Structure

```text
.
├── historian_60s (1).csv                              # Source 90-day 1-minute historian dataset (129,600 x 91)
├── Machine Learning Project Guidelines.pdf            # 14-Milestone Chemical Engineering ML Guidelines
├── Historian Based FD ML Milestone 1,2,3,4.pdf        # Milestones 1–4 Report
├── Historian based FD ML Milestone 5,6 Report.pdf     # Milestones 5–6 Report
├── README.md                                          # Project documentation & model evaluation leaderboard
└── training/
    ├── split_dataset.py                               # Chronological 70/15/15 dataset splitter
    ├── pipeline_utils.py                              # Shared physics feature engineering & evaluation utilities
    ├── train_and_evaluate.py                          # Full training & benchmark evaluation script
    ├── test_and_evaluate.py                           # Master runner for all model folders & leaderboard
    ├── splits/
    │   ├── train_70.csv                               # 70% Training split (90,720 rows)
    │   ├── val_15.csv                                 # 15% Validation split (19,440 rows)
    │   ├── test_15.csv                                # 15% Testing split (19,440 rows)
    │   └── split_metadata.json                        # Split timestamps & missing-value audit
    ├── models/
    │   ├── checkpoints_manifest.json                  # Manifest ranking all saved model checkpoints
    │   ├── pca_t2_q/
    │   │   ├── pca_t2_q_checkpoint.npz
    │   │   ├── pca_t2_q_scores.json
    │   │   ├── pca_t2_q_test_scores.csv
    │   │   ├── pca_t2_q_candidate_events.csv
    │   │   └── test_and_evaluate_pca_t2_q.py
    │   ├── lstm_autoencoder/
    │   │   ├── lstm_autoencoder_checkpoint.npz
    │   │   ├── lstm_autoencoder_scores.json
    │   │   ├── lstm_autoencoder_test_scores.csv
    │   │   ├── lstm_autoencoder_candidate_events.csv
    │   │   └── test_and_evaluate_lstm_autoencoder.py
    │   ├── local_outlier_factor/
    │   │   ├── lof_checkpoint.npz
    │   │   ├── local_outlier_factor_scores.json
    │   │   ├── local_outlier_factor_test_scores.csv
    │   │   ├── local_outlier_factor_candidate_events.csv
    │   │   └── test_and_evaluate_lof.py
    │   └── isolation_forest/
    │       ├── isolation_forest_checkpoint.npz
    │       ├── isolation_forest_scores.json
    │       ├── isolation_forest_test_scores.csv
    │       ├── isolation_forest_candidate_events.csv
    │       └── test_and_evaluate_isolation_forest.py
    └── results/
        ├── all_models_test_evaluation.json
        ├── evaluation_summary.json
        ├── model_comparison.csv
        ├── model_comparison_chart.png
        └── test_candidate_fault_events.csv
```

---

## 5. How to Run & Resume From Saved Checkpoints

Each model script automatically checks for its saved `.npz` checkpoint first (`✓ LOADED BEST CHECKPOINT`), resumes from the next epoch/stage, prints a formatted epoch-by-epoch table, and only overwrites the checkpoint when a new higher score is achieved:

```bash
# Run individual model training/evaluation (auto-resumes from best saved checkpoint)
python3 training/models/pca_t2_q/test_and_evaluate_pca_t2_q.py
python3 training/models/lstm_autoencoder/test_and_evaluate_lstm_autoencoder.py
python3 training/models/local_outlier_factor/test_and_evaluate_lof.py
python3 training/models/isolation_forest/test_and_evaluate_isolation_forest.py

# Run all models and print the combined leaderboard
python3 training/test_and_evaluate.py
```
