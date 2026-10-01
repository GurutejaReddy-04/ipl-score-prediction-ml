# IPL Score Prediction (Academic ML Project)

An academic machine learning project demonstrating end-to-end data preprocessing, classical regression baselines, a deep neural network, evaluation methodology, and an interactive prediction interface for Indian Premier League (IPL) cricket match scores.

Originally developed as a B.Tech Minor Project in the Department of Electronics and Communication Engineering at the **National Institute of Technology (NIT) Andhra Pradesh** under the mentorship of **Dr. B. Thulasya Naik**.

---

## Project Attribution & Maintainer
- **Repository Maintainer**: [Guruteja Reddy Nallachi](https://github.com/GurutejaReddy-04)
- **Academic Context**: Developed as a collaborative undergraduate minor project at NIT Andhra Pradesh. The repository is maintained for portfolio and educational reference. Contributions from undergraduate student co-authors are gratefully acknowledged from the original academic submission.

---

## Overview & Scope
This project predicts final innings totals in Twenty20 cricket based on current match states and context.

This project was built to explore regression modeling on historical ball-by-ball IPL match telemetry:
- **Classical Baselines**: Dummy estimators, Ordinary Least Squares Linear Regression, and Ridge Regression.
- **Deep Neural Network**: A 3-layer Sequential MLP (512 → 216 → 1) optimized with Huber loss.
- **Interactive UI**: An IPyWidgets interface within Jupyter Notebook allowing interactive match scenario predictions.
- **Methodological Evaluation**: Comparative analysis between the historical row-wise experiment and leak-free match-grouped and temporal evaluation protocols.

---

## Project Structure
```text
.
├── data/
│   ├── README.md                           # Dataset schema and provenance details
│   └── ipl.csv                             # Historical IPL ball-by-ball dataset (2008–2017)
├── docs/
│   ├── .gitkeep
│   └── architecture.md                     # System architecture & evaluation methodology
├── notebooks/
│   └── IPL Score Prediction Project.ipynb  # Interactive EDA and model exploration notebook
├── src/
│   ├── evaluate_baselines.py               # Auditable leak-free benchmark evaluation script
│   └── ipl_score_prediction_project.py     # Core training script with IPyWidgets UI
├── .gitignore                              # Git exclusion rules
├── LICENSE                                 # MIT open-source license
├── README.md                               # Project documentation
└── requirements.txt                        # Tested environment dependencies
```
*(Note: Internal university assessment documents, presentation slides, and intermediate scratch files are excluded from public version control via `.gitignore`.)*

---

## Experimental Results & Benchmark Evaluation

### Table 1: Historical Academic Result & Audit Reconstruction
> [!NOTE]
> **Historical academic result — not directly comparable to the corrected benchmark**
> The **12.93 MAE** neural network result follows the original academic implementation (Colab notebook using LabelEncoder and an 80/20 random row-wise split) and is preserved as an archival documentation record. The baseline metrics (Dummy, Linear Regression, Ridge) were reconstructed during the audit under the identical historical row-wise split protocol. This protocol reflects within-match delivery overlap and reuse of the test set for validation during training.

| Protocol | Model | Test MAE (Runs) | Notes |
| :--- | :--- | :---: | :--- |
| **Original 80/20 Random Row Split** | Dummy Regressor (Mean baseline) | 22.76 | Audit reconstruction (target mean = 159.9 runs) |
| | Dummy Regressor (Median baseline) | 22.74 | Audit reconstruction (target median = 158 runs) |
| | Multiple Linear Regression | 14.87 | Audit reconstruction (R² = 0.5191) |
| | Ridge Regression (α = 1.0) | 14.87 | Audit reconstruction (R² = 0.5191) |
| | **Deep Neural Network** | **12.93** | Original notebook result (Cell 23) |

---

### Table 2: Methodologically Corrected Benchmark (Strict Holdout Evaluation)
To prevent delivery-level data leakage, preprocessing (`OneHotEncoder(drop='first', handle_unknown='ignore')` + `MinMaxScaler()`) is fitted strictly on the training partition. All benchmarks below evaluate on untouched holdout test matches and are reproducible under the documented, seed-controlled environment by running [`src/evaluate_baselines.py`](./src/evaluate_baselines.py):

| Protocol | Partition Breakdown (Matches / Rows) | Dummy Mean MAE | Linear Regression MAE | Ridge (α = 1.0) MAE | Neural Network MAE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Match-Grouped Split** | Train: 431 / 53,046<br>Val: 61 / 7,507<br>Test: 125 / 15,461 | **22.47** | **15.32** | **15.32** | **23.59** |
| **Temporal Split** | Train (≤2014): 448 / 55,226<br>Val (2015): 55 / 6,714<br>Test (2016–17): 114 / 14,074 | **22.10** | **15.53** | **15.51** | **24.21** |

> [!TIP]
> **Key Methodological Takeaways**:
> 1. **Match-Grouped Split**: Strictly groups by match ID (`mid`), guaranteeing that no delivery from a test match appears in the training partition. Classical linear baselines achieve **15.32 MAE** on completely unseen matches.
> 2. **Temporal Split**: Tests true forecasting into future seasons (train on ≤2014, validate on 2015, test on 2016–2017). Using `drop='first'` prevents dummy-variable collinearity, while `handle_unknown='ignore'` gracefully encodes unseen franchise additions in test seasons without feature dimension shifts, enabling Linear Regression to achieve **15.53 MAE** (matching Ridge's **15.51 MAE**).
> 3. **Neural Network Generalization Insight**: Under leak-free holdout evaluation, the fixed 512 → 216 → 1 MLP achieves **23.59 MAE** on unseen matches and **24.21 MAE** on future seasons, underperforming the linear baselines and indicating poor out-of-sample generalization for this unregularized configuration. The original 12.93 MAE should not be interpreted as a directly comparable estimate of unseen-match performance because its row-wise split allowed within-match overlap and reused the test set for validation.
> 4. For deep architectural and methodology details, see [`docs/architecture.md`](./docs/architecture.md).

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/GurutejaReddy-04/ipl-score-prediction-ml.git
cd ipl-score-prediction-ml
```

### 2. Create and activate a virtual environment
```bash
# On Linux / macOS:
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install verified dependencies
```bash
pip install -r requirements.txt
```

> **Tested Environment**: Python 3.10 / 3.11 on Windows 11 and Ubuntu Linux (CPU execution). The original exploratory notebook was developed in Google Colab.

---

## Usage

### 1. Run the Auditable Benchmark Evaluation
Reproduce the clean leak-free baseline comparisons in Table 2:
```bash
python src/evaluate_baselines.py
```

### 2. Run the Interactive Prediction Script
Execute the main model training script and launch the prediction interface:
```bash
python src/ipl_score_prediction_project.py
```

### 3. Run the Jupyter Notebook
For interactive exploratory data analysis and IPyWidgets sliders:
```bash
jupyter notebook "notebooks/IPL Score Prediction Project.ipynb"
```

---

## Dataset Provenance & Third-Party Terms
- **Dataset**: Historical ball-by-ball match data covering IPL seasons 2008 to 2017 (617 matches, 76,014 records).
- **Source**: Kaggle (*"IPL Dataset Season 2008 to 2017"*).
- **Third-Party Terms**: The dataset in [`data/ipl.csv`](./data/ipl.csv) is third-party historical sports data included for academic demonstration. The software license below applies exclusively to the repository code and documentation, and does not grant license rights over the underlying sports dataset. Refer to [`data/README.md`](./data/README.md) for full schema details.

---

## Limitations & Academic Scope
- **Real-Time Context**: The model uses intermediate scorecard state (runs, wickets, overs, venue, teams) and does not ingest live weather telemetry, pitch degradation, bowler spell limits, or player form.
- **Target Distribution**: Target values represent full 20-over innings totals and do not dynamically adjust for rain interruptions (DLS method) or mid-innings declarations.

---

## License
The code and documentation in this repository are released under the [MIT License](./LICENSE).
