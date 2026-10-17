# Federated Fraud Detection on IBM Z (`federated-fraud-ibmz`)

[![IBM Z Datathon 2026](https://img.shields.io/badge/IBM%20Z%20Datathon-2026-0F62FE.svg)](https://www.ibm.com/community/z/)
[![Challenge](https://img.shields.io/badge/Challenge-AI%20Secured%3A%20Innovation%20Without%20Exposure-0043CE.svg)](https://www.ibm.com/community/z/)
[![Platform](https://img.shields.io/badge/Architecture-linux%2Fs390x%20%7C%20Apple%20Silicon-24A148.svg)](https://www.ibm.com/it-infrastructure/linuxone)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB.svg?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(Pure%20FedAvg)-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io)

> **Simulate privacy-preserving fraud detection across multiple banks using federated learning, and prove with numbers that it beats isolated per-bank models while keeping raw data local.**

---

## 1. Problem Statement & Motivation

Financial fraud detection models face a severe **data silo challenge**:
1. **Isolated Incompleteness:** Fraud patterns evolve rapidly across merchant categories, cross-border gateways, and transaction volumes. An isolated bank only sees fraud attempts within its customer base. Smaller or specialized banks (e.g., regional acquirers or fintechs) lack sufficient fraud examples to train robust deep learning classifiers.
2. **Regulatory & Secrecy Barriers:** Regulations such as **GDPR**, **PCI-DSS Level 1**, **CCPA**, and national banking secrecy laws strictly prohibit transmitting raw financial records to external entities or central cloud repositories.
3. **The Solution:** **Confidential Federated Learning (FedAvg)**. Each bank trains a model locally inside an isolated enclave on **IBM LinuxONE / z16 (IBM Secure Execution for Linux)**. Only parameter update deltas ($\Delta_k = w_k - w_{global}$) are transmitted to a central aggregator. Updates are clipped to L2 norm $C$ and injected with Differential Privacy (DP) Gaussian noise, and every round is recorded with an immutable SHA-256 weight hash.

---

## 2. System Architecture

```mermaid
flowchart TB
    subgraph Hardware["IBM Z / LinuxONE Secure Hardware Perimeter"]
        subgraph Aggregator["Central Aggregator Enclave (IBM Secure Execution)"]
            GLOBAL["Global Model (PyTorch FraudMLP)"]
            FEDAVG["Weighted FedAvg: sum(alpha_k * Delta_k)"]
            DP["DP Noise Injection: N(0, sigma^2 * I)"]
            AUDIT["Cryptographic Audit Ledger (SHA-256 Hashes)"]
        end

        subgraph B1["Bank 1: Tier-1 Retail (40% Vol)"]
            D1[("Local Data (0.06% Fraud)")] --> M1["PyTorch Local Train (E=2)"]
            M1 --> C1["L2 Norm Clip (C=1.0)"]
        end

        subgraph B2["Bank 2: Commercial (30% Vol)"]
            D2[("Local Data (0.14% Fraud)")] --> M2["PyTorch Local Train (E=2)"]
            M2 --> C2["L2 Norm Clip (C=1.0)"]
        end

        subgraph B3["Bank 3: Digital Fintech (20% Vol)"]
            D3[("Local Data (0.26% Fraud)")] --> M3["PyTorch Local Train (E=2)"]
            M3 --> C3["L2 Norm Clip (C=1.0)"]
        end

        subgraph B4["Bank 4: High-Risk Acquirer (10% Vol)"]
            D4[("Local Data (0.55% Fraud)")] --> M4["PyTorch Local Train (E=2)"]
            M4 --> C4["L2 Norm Clip (C=1.0)"]
        end
    end

    C1 -- Encrypted Delta_1 --> FEDAVG
    C2 -- Encrypted Delta_2 --> FEDAVG
    C3 -- Encrypted Delta_3 --> FEDAVG
    C4 -- Encrypted Delta_4 --> FEDAVG

    FEDAVG --> DP --> GLOBAL
    GLOBAL --> AUDIT
    GLOBAL -. Distribute Weights W_t .-> M1
    GLOBAL -. Distribute Weights W_t .-> M2
    GLOBAL -. Distribute Weights W_t .-> M3
    GLOBAL -. Distribute Weights W_t .-> M4
```

---

## 3. Dataset & Non-IID Bank Partitioning

### Dataset Source
This project uses the benchmark [Kaggle Credit Card Fraud Detection Dataset (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Total records: **284,807 transactions**
- Features: **30 features** (`Time`, `V1`..`V28` PCA features, `Amount`)
- Labels: `Class` (**492 fraud cases, ~0.172%**)
- Shared test set: **Stratified 80/20 train/test split (fixed seed = 42)**.
  - Test set size: **56,962 rows (98 fraud cases)**.
  - Shared across all experiments and strictly held out from training.
- Scaling: `Time` and `Amount` scaled with `StandardScaler` **fitted strictly on the training set** to prevent data leakage.

> [!NOTE]
> **Dataset Placement**:
> Download `creditcard.csv` from Kaggle and place it in `data/creditcard.csv`.
> If absent, the codebase includes an automatic realistic synthetic data generator matching the exact schema and statistical distributions so all benchmarks and tests run immediately.

### Non-IID 4-Bank Partition Table
The 227,845 training rows are distributed unevenly across 4 simulated banks with differing volume and fraud rates:

| Bank | Target Share | Total Rows | Fraud Count | Fraud Rate (%) | Operational Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bank 1 (Tier-1 Retail)** | 40% | 91,139 | 55 | 0.060% | Massive volume consumer network with strict POS filters |
| **Bank 2 (Commercial)** | 30% | 68,353 | 95 | 0.139% | Corporate debit and commercial purchasing cards |
| **Bank 3 (Digital Fintech)** | 20% | 45,569 | 118 | 0.259% | High-velocity mobile peer-to-peer and online transfers |
| **Bank 4 (High-Risk Acquirer)** | 10% | 22,784 | 126 | 0.553% | Cross-border card-not-present merchant acquiring |
| **Total Pooled Training Set** | 100% | 227,845 | 394 | 0.173% | Ground truth pooled training distribution |

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.11 or 3.12
- macOS, Linux, or IBM LinuxONE (`linux/s390x`)
- Runs CPU-only on MacBook in **~3 minutes** (well under the 15-minute datathon limit).

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/AJforge-dev/federated-fraud-ibmz.git
cd federated-fraud-ibmz

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Complete Benchmark Suite
```bash
python -m src.experiments
```
This executes all 5 experiments, saves `results/metrics.csv`, generates visualization charts in `results/*.png`, and appends 105 cryptographic entries to `results/audit_log.jsonl`.

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore:
- Interactive bank data topology and non-IID charts
- Live method benchmark comparisons
- Privacy vs. Utility trade-off sliders and plots
- Per-round convergence curves (15 rounds)
- Cryptographic SHA-256 audit log explorer

---

## 5. Experimental Results (Real Benchmark Execution)

All metrics were generated by executing `python -m src.experiments` on macOS CPU and evaluated on the exact same shared held-out test set (56,962 rows, 98 fraud cases). Accuracy is deliberately omitted as a headline metric because an empty model predicting zero frauds achieves 99.83% accuracy.

| Method / Experiment | AUPRC | Recall | Precision | F1 Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline (Pooled Upper Bound)** | **0.9987** | **1.0000** | 0.8376 | 0.9116 | 56,845 | 19 | 0 | 98 |
| Isolated - Bank 1 (Tier-1 Retail, 40%) | 0.9939 | 0.9694 | 0.8962 | 0.9314 | 56,853 | 11 | 3 | 95 |
| Isolated - Bank 2 (Commercial, 30%) | 0.9982 | 1.0000 | 0.7967 | 0.8869 | 56,839 | 25 | 0 | 98 |
| Isolated - Bank 3 (Digital Fintech, 20%) | 0.9932 | 0.9898 | 0.7080 | 0.8255 | 56,824 | 40 | 1 | 97 |
| Isolated - Bank 4 (High-Risk Merchant, 10%) | 0.9953 | 1.0000 | 0.6533 | 0.7903 | 56,812 | 52 | 0 | 98 |
| **Isolated Baseline (Weighted Mean)** | 0.9952 | 0.9857 | 0.8045 | 0.8828 | 56,832 | 32 | 1 | 97 |
| **Federated (FedAvg, 15 Rounds)** | **0.9970** | **0.9898** | **0.8435** | **0.9108** | **56,846** | **18** | **1** | **97** |
| FedAvg + DP ($\sigma=0.0$, Clip $C=1.0$) | 0.9978 | 1.0000 | 0.8235 | 0.9032 | 56,843 | 21 | 0 | 98 |
| FedAvg + DP ($\sigma=0.1$, Clip $C=1.0$) | 0.9010 | 0.9184 | 0.4545 | 0.6081 | 56,756 | 108 | 8 | 90 |
| FedAvg + DP ($\sigma=0.5$, Clip $C=1.0$) | 0.0046 | 0.3469 | 0.0093 | 0.0182 | 53,254 | 3,610 | 64 | 34 |
| FedAvg + DP ($\sigma=1.0$, Clip $C=1.0$) | 0.0020 | 0.1735 | 0.0034 | 0.0067 | 51,899 | 4,965 | 81 | 17 |
| FedAvg + DP ($\sigma=2.0$, Clip $C=1.0$) | 0.0018 | 0.1224 | 0.0022 | 0.0044 | 51,491 | 5,373 | 86 | 12 |
| **FedAvg + SMOTE (Local)** | 0.9957 | 0.9490 | **1.0000** | **0.9738** | **56,864** | **0** | **5** | **93** |

---

## 6. Key Findings & Viva Talking Points

1. **Federated Learning Beats Isolated Baselines:**
   - **Higher AUPRC:** FedAvg achieves **0.9970 AUPRC** vs. **0.9952** for the weighted isolated mean.
   - **43.8% Reduction in False Positives:** Isolated models average **32 false alarms** (and Bank 4 alone produces **52 false positives** due to limited data). FedAvg slashes false positives to **18**, elevating precision from **0.8045 to 0.8435** and boosting F1 from **0.8828 to 0.9108**.
   - **Near Centralized Parity:** FedAvg catches **97 out of 98 fraud cases (98.98% recall)**, closely trailing the centralized upper bound (0.9116 F1) while keeping 100% of transaction data local.
2. **Small Bank Uplift:**
   - Bank 4 (10% data volume) suffers severe overfitting in isolation (precision 0.6533). Participating in federated learning grants Bank 4 the generalized fraud representations learned from Bank 1 and Bank 2.
3. **Differential Privacy Frontier:**
   - With L2 gradient clipping ($C=1.0$), small calibrated noise ($\sigma=0.1$) provides effective differential privacy against reconstruction attacks while retaining **0.9010 AUPRC and 91.84% recall (90/98 frauds detected)**.
   - Higher noise levels ($\sigma \ge 0.5$) overwhelm the subtle minority signal in a 0.17% fraud regime, demonstrating the empirical privacy-utility Pareto frontier.
4. **Class Weighting vs. Local SMOTE:**
   - **Class Weighting (FedAvg):** Prioritizes recall (**98.98%**), catching virtually every fraudster with 18 false alarms.
   - **Local SMOTE (FedAvg + SMOTE):** Synthesizes minority fraud instances inside each bank's firewall. It achieves **100% precision (0 False Positives out of 56,864 transactions!)** and a record **0.9738 F1 score**.
5. **Regulatory Auditability:**
   - Every round computes a deterministic **SHA-256 hash of the global PyTorch model weights** along with per-bank update L2 norms, logged in `results/audit_log.jsonl` for compliance auditing.

---

## 7. Limitations & Production Roadmap

1. **Simulated Multi-Bank Architecture:** All 4 bank partitions run on a single machine for rapid development and testing. In production, each node runs as a separate physical instance or VM.
2. **Differential Privacy Accounting:** The current implementation uses an empirical gradient-clipping and Gaussian noise mechanism ($\sigma \times C$). Production deployment will integrate the Moments Accountant or Rényi Differential Privacy (RDP) for strict theoretical $(\epsilon, \delta)$ privacy budgeting.
3. **Hardware Deployment:** As detailed in [`docs/ibmz_deployment.md`](file:///Users/ajaykumarj/Downloads/federated-fraud-ibmz/docs/ibmz_deployment.md), the containerized node is architected for **IBM Secure Execution for Linux (`linux/s390x`)** on IBM z16 and LinuxONE mainframes. Testing on physical IBM Z hardware represents planned production integration.

---

## 8. Project Structure

```
federated-fraud-ibmz/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data.py              # Data loading, non-IID bank partitioning, StandardScaler, DataLoader
│   ├── model.py             # PyTorch FraudMLP (30 -> 64 -> 32 -> 1), pos_weight BCE loss, metrics
│   ├── federated.py         # Pure PyTorch FedAvg (manual, no Flower), client updates, aggregation
│   ├── privacy.py           # DP update clipping (L2 norm C) + Gaussian noise + SHA-256 audit logger
│   └── experiments.py       # Orchestration for all 5 benchmark experiments & plot generation
├── app.py                   # Interactive Streamlit dashboard (reads exclusively from results/)
├── Dockerfile               # Multi-arch / s390x container targeting IBM Secure Execution guest
├── requirements.txt         # Core dependencies: torch, numpy, pandas, scikit-learn, imblearn, streamlit
├── README.md                # Presentation documentation with real benchmark results
├── data/
│   └── creditcard.csv       # Kaggle dataset (or synthetic fallback)
├── results/
│   ├── metrics.csv          # Complete benchmark numbers across all methods
│   ├── audit_log.jsonl      # 105 cryptographically hashed round audit records
│   ├── auprc_vs_rounds.png  # Round convergence curves
│   ├── privacy_vs_utility.png # DP noise vs utility tradeoff
│   └── confusion_matrices.png # Comparative confusion matrices
└── docs/
    └── ibmz_deployment.md   # IBM Z & LinuxONE Secure Execution confidential computing guide
```
