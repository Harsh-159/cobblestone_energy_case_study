#!/usr/bin/env python3
"""
AutoGluon Tabular evaluation on the SAME OOS period as all other models.

Trains AutoGluon (high_quality, 8-fold bagging, no stacking) for 5 minutes
on the same 60+ features, then applies AR + hourly-EW bias correction.

Prints hourly / weekly / monthly MAE, RMSE, and directional accuracy.

Usage:
    python scripts/run_autogluon_tabular.py
"""

import sys, time, warnings, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.ingestion import DATA_PROCESSED
from src.models import TARGET_COL, OOS_TEST_START
from scripts.run_forecasting import apply_hourly_exp_expanding_bias_correction

# ── Config ──────────────────────────────────────────────────────────
AG_TIME_LIMIT = 300        # 5 minutes
AG_PATH = PROJECT_ROOT / "models" / "ag_tabular_eval"
NUM_BAG_FOLDS = 4

# ── Load data ───────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PROCESSED / "de_power_dataset.parquet")
fm = build_feature_matrix(df)

oos_start = pd.Timestamp(OOS_TEST_START, tz="UTC")
train = fm.loc[fm.index < oos_start].dropna(subset=[TARGET_COL])
test  = fm.loc[fm.index >= oos_start].dropna(subset=[TARGET_COL])

X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
X_test,  y_test  = test.drop(columns=[TARGET_COL]),  test[TARGET_COL]

print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows  |  Features: {X_train.shape[1]}")
print(f"OOS period: {X_test.index[0].date()} → {X_test.index[-1].date()}")

# ── Train AutoGluon ─────────────────────────────────────────────────
from autogluon.tabular import TabularPredictor

train_df = X_train.fillna(-999).copy()
train_df[TARGET_COL] = y_train

print(f"\n{'='*60}")
print(f"Training AutoGluon for {AG_TIME_LIMIT//60} minutes (high_quality, {NUM_BAG_FOLDS}-fold bagging)...")
print(f"{'='*60}")

t0 = time.time()
predictor = TabularPredictor(
    label=TARGET_COL,
    eval_metric="mean_absolute_error",
    path=str(AG_PATH),
    verbosity=1,
).fit(
    train_data=train_df,
    time_limit=AG_TIME_LIMIT,
    presets="high_quality",
    dynamic_stacking=False,
    num_stack_levels=0,
    num_bag_folds=NUM_BAG_FOLDS,
)
elapsed = time.time() - t0
print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

lb = predictor.leaderboard(silent=True)
print(f"Best model: {predictor.model_best}  |  Total models: {len(lb)}")
print("\nTop 5 models (validation MAE):")
for _, r in lb.head(5).iterrows():
    print(f"  {r['model']:35s}  MAE={-r['score_val']:.2f}")

# ── Predict ─────────────────────────────────────────────────────────
print("\nPredicting on OOS set...")
ag_oos_pred = predictor.predict(X_test.fillna(-999)).values

print("Predicting on train set (for bias correction)...")
ag_train_pred = predictor.predict(X_train.fillna(-999)).values

# ── AR + Hourly-EW bias correction ─────────────────────────────────
print("Applying hourly exponential-weighted bias correction...")
ag_ew_pred, _ = apply_hourly_exp_expanding_bias_correction(
    y_actual_test=y_test.values,
    raw_preds_test=ag_oos_pred,
    hours_test=X_test.index.hour.values,
    y_actual_train=y_train.values,
    raw_preds_train=ag_train_pred,
    hours_train=X_train.index.hour.values,
    ew_halflife=672,
)

# ── Compute metrics ─────────────────────────────────────────────────
idx = X_test.index
actual = y_test.values
lag24  = X_test["price_lag_24h"].values
lag168 = X_test["price_lag_168h"].values


def full_metrics(pred, label):
    """Compute hourly, weekly, monthly MAE/RMSE/DirAcc."""
    m = ~np.isnan(pred) & ~np.isnan(actual)
    p, a = pred[m], actual[m]

    # Hourly
    h_mae  = np.abs(a - p).mean()
    h_rmse = np.sqrt(((a - p)**2).mean())

    # Dir Acc vs lag-168h
    l168 = lag168[m]
    valid = ~np.isnan(l168)
    pd_sign = np.sign(p[valid] - l168[valid])
    ad_sign = np.sign(a[valid] - l168[valid])
    nz = (pd_sign != 0) & (ad_sign != 0)
    h_da = (pd_sign[nz] == ad_sign[nz]).mean() * 100

    # Weekly
    df = pd.DataFrame({"a": a, "p": p}, index=idx[m])
    df["wk"] = df.index.isocalendar().week.astype(int)
    df["yr"] = df.index.year
    wg = df.groupby(["yr", "wk"]).mean()
    w_mae  = np.abs(wg["a"] - wg["p"]).mean()
    w_rmse = np.sqrt(((wg["a"] - wg["p"])**2).mean())

    # Monthly
    df["mo"] = df.index.to_period("M")
    mg = df.groupby("mo").mean()
    m_mae  = np.abs(mg["a"] - mg["p"]).mean()
    m_rmse = np.sqrt(((mg["a"] - mg["p"])**2).mean())

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  {'Granularity':<12} {'MAE':>10} {'RMSE':>10} {'Dir Acc':>10}")
    print(f"  {'─'*42}")
    print(f"  {'Hourly':<12} {h_mae:>10.2f} {h_rmse:>10.2f} {h_da:>9.1f}%")
    print(f"  {'Weekly':<12} {w_mae:>10.2f} {w_rmse:>10.2f} {'—':>10}")
    print(f"  {'Monthly':<12} {m_mae:>10.2f} {m_rmse:>10.2f} {'—':>10}")

    return {"h_mae": h_mae, "h_rmse": h_rmse, "h_da": h_da,
            "w_mae": w_mae, "m_mae": m_mae}


# Also compute naive baselines for comparison
naive24_pred = lag24.copy()
naive168_pred = lag168.copy()

print(f"\n{'='*60}")
print(f"  RESULTS — OOS: {X_test.index[0].date()} → {X_test.index[-1].date()}")
print(f"  ({len(X_test):,} hours)")
print(f"{'='*60}")

full_metrics(naive24_pred,  "Seasonal Naive (24h)")
full_metrics(naive168_pred, "Seasonal Naive (168h)")
full_metrics(ag_oos_pred,   "AutoGluon (5min, raw)")
full_metrics(ag_ew_pred,    "AutoGluon (5min) + AR + hourly-EW bias")

# ── Cleanup ─────────────────────────────────────────────────────────
shutil.rmtree(AG_PATH, ignore_errors=True)
print(f"\nCleaned up {AG_PATH}")
print("Done.")
