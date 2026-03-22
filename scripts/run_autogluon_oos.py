#!/usr/bin/env python3
"""
Run AutoGluon on the SAME OOS split as the main pipeline.
Reports hourly, weekly, and monthly MAE for direct comparison.

Gives AutoGluon 900 seconds (15 min) with best_quality preset.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features import build_feature_matrix
from src.ingestion import DATA_PROCESSED
from src.models import TARGET_COL, OOS_TEST_START, OOS_TEST_END, compute_weekly_monthly_metrics
from src.autogluon_forecaster import AutoGluonForecaster

AUTOGLUON_TIME_LIMIT = 900  # 15 minutes — plenty for ~33k training rows


def main():
    sep = "=" * 60

    # ---- Load data & features ----
    print(f"{sep}\nLoading data and building features...\n{sep}")
    df = pd.read_parquet(DATA_PROCESSED / "de_power_dataset.parquet")
    feature_matrix = build_feature_matrix(df)
    print(f"Feature matrix: {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} cols")

    # ---- Same train/test split as main pipeline ----
    oos_start_ts = pd.Timestamp(OOS_TEST_START, tz="UTC")
    oos_end_ts = pd.Timestamp(f"{OOS_TEST_END} 23:00", tz="UTC")

    train = feature_matrix.loc[feature_matrix.index < oos_start_ts].dropna(subset=[TARGET_COL])
    test = feature_matrix.loc[
        (feature_matrix.index >= oos_start_ts) & (feature_matrix.index <= oos_end_ts)
    ].dropna(subset=[TARGET_COL])

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    print(f"Train: {len(train)} rows ({train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')})")
    print(f"Test:  {len(test)} rows ({test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')})")

    # ---- Fit AutoGluon ----
    print(f"\n{sep}\nFitting AutoGluon (time_limit={AUTOGLUON_TIME_LIMIT}s, presets=best_quality)...\n{sep}")
    ag = AutoGluonForecaster(
        time_limit=AUTOGLUON_TIME_LIMIT,
        presets="best_quality",
        save_path=str(PROJECT_ROOT / "models" / "autogluon_oos"),
    )
    ag.fit(X_train, y_train)

    # ---- Predict ----
    print(f"\n{sep}\nGenerating OOS predictions...\n{sep}")
    preds = ag.predict(X_test)

    # ---- Build OOS DataFrame ----
    oos_df = pd.DataFrame({
        "y_actual": y_test.values,
        "y_pred": preds,             # AutoGluon as main pred
        "y_pred_autogluon": preds,   # Also explicit column
    }, index=y_test.index)

    # Also add naive for comparison
    if "price_lag_168h" in X_test.columns:
        oos_df["y_pred_naive"] = X_test["price_lag_168h"].values

    # ---- Compute metrics at all granularities ----
    metrics = compute_weekly_monthly_metrics(oos_df)

    # ---- Print results ----
    print(f"\n{sep}")
    print(" AUTOGLUON OOS RESULTS (same split as main pipeline)")
    print(f" OOS Period: {OOS_TEST_START} to {OOS_TEST_END}")
    print(f" Time limit: {AUTOGLUON_TIME_LIMIT}s | Preset: best_quality")
    print(f" Best model: {ag.predictor.model_best}")
    print(f" Models trained: {len(ag.leaderboard())}")
    print(f"{sep}\n")

    for granularity in ["hourly", "weekly", "monthly"]:
        if granularity not in metrics:
            continue
        gran_data = metrics[granularity]
        print(f"  {granularity.upper()} ACCURACY:")
        print(f"  {'Model':28s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAPE':>8s}  {'MBE':>8s}  {'N':>5s}")
        print(f"  {'-' * 65}")
        for model_name, m in gran_data.items():
            mae = f"{m.get('MAE', 0):.2f}"
            rmse = f"{m.get('RMSE', 0):.2f}"
            mape = f"{m.get('MAPE', 0):.1f}%" if pd.notna(m.get("MAPE")) else "N/A"
            mbe = f"{m.get('MBE', 0):+.2f}"
            n = f"{m.get('n', 0)}"
            print(f"  {model_name:28s}  {mae:>8s}  {rmse:>8s}  {mape:>8s}  {mbe:>8s}  {n:>5s}")
        print()

    # ---- Print comparison table ----
    print(f"{sep}")
    print(" COMPARISON: AutoGluon vs Our Pipeline vs Naive")
    print(f"{sep}")
    print(f"  {'Metric':20s}  {'AutoGluon':>12s}  {'Our Pipeline':>12s}  {'Naive-168h':>12s}")
    print(f"  {'-' * 60}")

    ag_hourly = metrics.get("hourly", {}).get("autogluon", {}).get("MAE", float("nan"))
    ag_weekly = metrics.get("weekly", {}).get("autogluon", {}).get("MAE", float("nan"))
    ag_monthly = metrics.get("monthly", {}).get("autogluon", {}).get("MAE", float("nan"))

    print(f"  {'Hourly MAE':20s}  {ag_hourly:>12.2f}  {'13.86':>12s}  {'34.84':>12s}")
    print(f"  {'Weekly MAE':20s}  {ag_weekly:>12.2f}  {'5.82':>12s}  {'18.93':>12s}")
    print(f"  {'Monthly MAE':20s}  {ag_monthly:>12.2f}  {'3.98':>12s}  {'5.67':>12s}")
    print(f"{sep}")

    # ---- Print leaderboard ----
    print(f"\nAutoGluon Leaderboard (top 10):")
    lb = ag.leaderboard()
    print(lb.head(10).to_string())

    # Cleanup
    import shutil
    ag_path = Path(PROJECT_ROOT / "models" / "autogluon_oos")
    if ag_path.exists():
        shutil.rmtree(ag_path, ignore_errors=True)
        print(f"\nCleaned up {ag_path}")

    return metrics


if __name__ == "__main__":
    main()
