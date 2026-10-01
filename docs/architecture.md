# System Architecture & Evaluation Methodology

## 1. Overview & Problem Definition
The objective of this project is to predict the final total score of an Indian Premier League (IPL) cricket innings based on intermediate match state telemetry. 

While the prediction target is a match-level quantity (`total` runs scored in the innings), the observations in the dataset are recorded at the delivery/over level as the match unfolds.

---

## 2. Feature Representation & Preprocessing Pipeline

### Selected Features
From the 15 raw columns in `data/ipl.csv`, the modeling pipeline extracts 6 predictor features:
- **Categorical (Context)**: `venue`, `bat_team`, `bowl_team`
- **Numerical (State)**: `wickets` (0–9), `overs` (0.0–20.0), `runs` (current cumulative score)
- **Target**: `total` (final innings total score)

### Preprocessing Strategies
1. **Historical Academic Implementation** (`LabelEncoder` + `MinMaxScaler`):
   - Categorical columns are converted to integer ranks via scikit-learn's `LabelEncoder`.
   - Numerical columns are normalized to $[0, 1]$ via `MinMaxScaler`.
   - *Limitation*: Treats nominal categories (venues, team names) as ordinal integers within a continuous metric space.

2. **Corrected Benchmark Implementation** (`OneHotEncoder` + `MinMaxScaler`):
   - Categorical columns are transformed using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
   - Dropping the first category eliminates dummy-variable collinearity, providing full column rank for ordinary least squares linear regression.
   - `handle_unknown='ignore'` maps previously unseen categories in future seasons (e.g., new franchises) to zero vectors without breaking matrix dimensions.
   - Numerical columns are scaled with `MinMaxScaler()`.
   - **Crucially, all transformers are fitted strictly on the training partition.** Validation and test sets are transformed without refitting.

---

## 3. Deep Learning Model Architecture
The primary neural network regression model is implemented using Keras / TensorFlow:

```text
Input Layer (Shape: [n_features])
       │
       ▼
Dense Layer (512 units, ReLU activation)
       │
       ▼
Dense Layer (216 units, ReLU activation)
       │
       ▼
Output Layer (1 unit, Linear activation)
```

### Loss Function & Optimization
- **Loss Function**: Huber Loss ($\delta = 1.0$)
  $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
  Huber loss behaves quadratically for small errors and linearly for large errors, providing robustness against extreme cricket totals (e.g., uncharacteristic collapse or record chase).
- **Optimizer**: Adam ($\alpha = 0.001$)
- **Training Epochs & Batch Size**: 50 epochs, batch size 64.

---

## 4. Evaluation Methodology Critique

### The Ball-Level vs. Match-Level Leakage Problem
In the original collegiate experiment, data was partitioned using a simple row-wise random split:
```python
train_test_split(X, y, test_size=0.2, random_state=42)
```
Because the dataset contains multiple delivery-state rows for each match (up to ~120 balls per innings), a random row split causes delivery records from the **same match** to appear in both training and test partitions.
- *Example*: Ball 10.2 of Match #42 is in the training set; ball 15.4 of Match #42 is in the test set.
- *Consequence*: The model is evaluated on interpolating within an innings whose trajectory it has already partially observed, producing an artificially optimistic test MAE (12.93 runs).

### Validation Set Contamination
In the original script, `validation_data=(X_test_scaled, y_test)` was supplied to `model.fit()`. The test set was repeatedly observed at the end of every epoch, violating the principle of an untouched holdout test partition.

---

## 5. Corrected 3-Way Partitioning Protocols

To measure true generalization performance, two leak-free protocols are defined:

```text
Protocol 1: Match-Grouped Split (Unseen Matches)
Total 617 Matches
  ├── 70% Train Matches (431 matches / 53,046 deliveries)  ── Preprocessing fit & model training
  ├── 10% Validation Matches (62 matches / 7,631 deliveries) ── NN epoch validation & tuning
  └── 20% Untouched Test Matches (124 matches / 15,337 deliveries) ── Final single-pass evaluation

Protocol 2: Temporal Split (Future Seasons Forecasting)
Seasons 2008–2017
  ├── Historical Train: ≤ 2014 Seasons (448 matches / 55,226 deliveries) ── Preprocessing fit & training
  ├── Validation Period: 2015 Season (55 matches / 6,714 deliveries) ── NN epoch validation & tuning
  └── Untouched Future Test: 2016–2017 Seasons (114 matches / 14,074 deliveries) ── Final single-pass evaluation
```

Under these protocols:
1. No match appears in more than one partition.
2. Preprocessors are fitted strictly on the training partition.
3. Linear baselines (Dummy Regressor, Linear Regression, Ridge) establish lower bounds for regression performance.
4. Deep learning models evaluate against untouched test matches after freezing parameters.
