"""
IPL Score Prediction - Benchmark Evaluation Script

This script evaluates baseline machine learning models on the IPL dataset using
methodologically sound, leak-free evaluation protocols:
1. Match-Grouped Split: Strict match isolation preventing within-match data leakage.
2. Temporal Split: Historical training (<=2014), validation (2015), and future holdout testing (2016-2017).

All preprocessing transformers (OneHotEncoder with drop='first' and MinMaxScaler)
are fitted strictly on the training partition to prevent preprocessing leakage.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

warnings.filterwarnings("ignore")

# Resolve project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ipl.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at expected path: {DATA_PATH}")

# -----------------------------------------------------------------------------
# 1. Dataset Loading & Feature Specification
# -----------------------------------------------------------------------------
print("=" * 80)
print("IPL SCORE PREDICTION - AUDITABLE BENCHMARK EVALUATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
features = ['venue', 'bat_team', 'bowl_team', 'wickets', 'overs', 'runs']
cat_cols = ['venue', 'bat_team', 'bowl_team']
num_cols = ['wickets', 'overs', 'runs']
target = 'total'

# Drop rows with null values in core features
df_clean = df[features + [target, 'mid', 'date']].dropna().copy()
df_clean['year'] = pd.to_datetime(df_clean['date'], format='%d-%m-%Y', errors='coerce').dt.year

print(f"Dataset Path        : {DATA_PATH}")
print(f"Total Rows          : {len(df_clean)}")
print(f"Unique Matches      : {df_clean['mid'].nunique()}")
print(f"Predictor Features  : {features}")
print(f"Target Feature      : {target}")
print(f"Categorical Encoding: OneHotEncoder(drop='first', handle_unknown='ignore') [fit on train only]")
print(f"Numerical Scaling   : MinMaxScaler() [fit on train only]")
print(f"Random Seed         : 42")
print("-" * 80)


def build_preprocessor() -> ColumnTransformer:
    """Builds a fresh column transformer to ensure no state leakage across runs."""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', MinMaxScaler(), num_cols)
        ]
    )


def evaluate_baselines_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol_name: str,
    extra_provenance: str = ""
) -> dict:
    """
    Evaluates Dummy, Linear Regression, and Ridge baselines on given 3-way split.
    Transformers are fitted strictly on train_df.
    """
    print(f"\nEvaluating Protocol: {protocol_name}")
    if extra_provenance:
        print(f"Provenance Details : {extra_provenance}")
    print(f"Partition Sizes     : Train = {len(train_df):,} rows ({train_df['mid'].nunique()} matches)")
    print(f"                      Val   = {len(val_df):,} rows ({val_df['mid'].nunique()} matches)")
    print(f"                      Test  = {len(test_df):,} rows ({test_df['mid'].nunique()} matches)")

    preprocessor = build_preprocessor()

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Preprocessing: Fit strictly on train partition!
    X_tr_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_te_proc = preprocessor.transform(X_test)

    # 1. Dummy Regressor (Mean)
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_tr_proc, y_train)
    dummy_pred = dummy.predict(X_te_proc)
    dummy_mae = mean_absolute_error(y_test, dummy_pred)

    # 2. Linear Regression (OLS)
    lr = LinearRegression()
    lr.fit(X_tr_proc, y_train)
    lr_pred = lr.predict(X_te_proc)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    # 3. Ridge Regression (alpha=1.0)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_proc, y_train)
    ridge_pred = ridge.predict(X_te_proc)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_r2 = r2_score(y_test, ridge_pred)

    # Check for optional Neural Network evaluation
    nn_mae_str = "TBD - measured after leak-free retraining"
    try:
        import tensorflow as tf
        import keras
        # Set random seeds for controlled execution
        tf.random.set_seed(42)
        if hasattr(keras, 'utils') and hasattr(keras.utils, 'set_random_seed'):
            keras.utils.set_random_seed(42)
        # Build matching sequential model
        nn_model = keras.Sequential([
            keras.layers.Input(shape=(X_tr_proc.shape[1],)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(216, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        nn_model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=1.0))
        # Train using true validation partition
        nn_model.fit(
            X_tr_proc, y_train,
            validation_data=(X_val_proc, y_val),
            epochs=50,
            batch_size=64,
            verbose=0
        )
        nn_pred = nn_model.predict(X_te_proc, verbose=0)
        nn_mae = mean_absolute_error(y_test, nn_pred)
        nn_mae_str = f"{nn_mae:.2f}"
    except ImportError:
        pass

    results = {
        'dummy_mae': dummy_mae,
        'lr_mae': lr_mae,
        'lr_rmse': lr_rmse,
        'lr_r2': lr_r2,
        'ridge_mae': ridge_mae,
        'ridge_rmse': ridge_rmse,
        'ridge_r2': ridge_r2,
        'nn_mae': nn_mae_str
    }

    print(f"Results on Holdout Test:")
    print(f"  - Dummy (Mean) MAE     : {dummy_mae:.2f}")
    print(f"  - Linear Regression MAE: {lr_mae:.2f} (RMSE: {lr_rmse:.2f}, R^2: {lr_r2:.4f})")
    print(f"  - Ridge (alpha=1.0) MAE: {ridge_mae:.2f} (RMSE: {ridge_rmse:.2f}, R^2: {ridge_r2:.4f})")
    print(f"  - Neural Network MAE   : {nn_mae_str}")

    return results


# -----------------------------------------------------------------------------
# 2. Protocol 1: Match-Grouped Split (70% Train, 10% Val, 20% Test)
# -----------------------------------------------------------------------------
matches = df_clean['mid'].unique()
np.random.seed(42)
np.random.shuffle(matches)
n_matches = len(matches)

n_train = int(0.7 * n_matches)
n_val = int(0.1 * n_matches)

train_m = matches[:n_train]
val_m = matches[n_train:n_train + n_val]
test_m = matches[n_train + n_val:]

train_grp = df_clean[df_clean['mid'].isin(train_m)]
val_grp = df_clean[df_clean['mid'].isin(val_m)]
test_grp = df_clean[df_clean['mid'].isin(test_m)]

res_grp = evaluate_baselines_on_split(
    train_grp, val_grp, test_grp,
    protocol_name="Match-Grouped Split (70/10/20 Match Partitions)",
    extra_provenance="Grouped by match ID (mid); zero within-match delivery leakage."
)

# -----------------------------------------------------------------------------
# 3. Protocol 2: Temporal Split (<=2014 Train, 2015 Val, 2016-2017 Test)
# -----------------------------------------------------------------------------
train_tmp = df_clean[df_clean['year'] <= 2014]
val_tmp = df_clean[df_clean['year'] == 2015]
test_tmp = df_clean[df_clean['year'] >= 2016]

res_tmp = evaluate_baselines_on_split(
    train_tmp, val_tmp, test_tmp,
    protocol_name="Temporal Split (Future Season Forecasting)",
    extra_provenance="Train: seasons <=2014 | Val: season 2015 | Test: seasons 2016-2017"
)

# -----------------------------------------------------------------------------
# 4. Summary Table Output
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY BENCHMARK TABLE (REPRODUCED FOR README TABLE 2)")
print("=" * 80)
print(f"{'Protocol':<25} | {'Dummy Mean':<10} | {'Linear Reg':<10} | {'Ridge':<10} | {'Neural Net':<10}")
print("-" * 80)
print(f"{'Match-Grouped (70/10/20)':<25} | {res_grp['dummy_mae']:<10.2f} | {res_grp['lr_mae']:<10.2f} | {res_grp['ridge_mae']:<10.2f} | {res_grp['nn_mae']:<10}")
print(f"{'Temporal (<=14/15/16-17)':<25} | {res_tmp['dummy_mae']:<10.2f} | {res_tmp['lr_mae']:<10.2f} | {res_tmp['ridge_mae']:<10.2f} | {res_tmp['nn_mae']:<10}")
print("=" * 80)
print("Benchmark run complete. All results auditable and reproducible.")
