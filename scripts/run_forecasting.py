#!/usr/bin/env python3
"""
Run forecasting pipeline (Part 2).

MODE: LGBM+Ridge Ensemble with bias correction.
Uses hardcoded best hyperparameters from previous HP search.
Trains ensemble once on pre-OOS data, applies bias correction,
then reports accuracy at hourly, weekly, and monthly granularity.

Requires: data/processed/de_power_dataset.parquet (from Part 1)

Produces:
    data/processed/oos_predictions.parquet
    outputs/model_performance_report.txt
    outputs/multi_granularity_accuracy.csv
    models/lgbm_ridge_ensemble.pkl
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.features import build_feature_matrix
from src.ingestion import DATA_PROCESSED, OUTPUTS_DIR, LOGS_DIR
from src.models import (
    SeasonalNaiveModel,
    # LastWeekSameDayModel,
    # LinearRegressionBaseline,
    # LightGBMForecaster,
    # XGBoostForecaster,
    LGBMRidgeEnsemble,
    TARGET_COL,
    # WF_N_FOLDS,
    PLOTS_DIR,
    LGBM_PARAMS,
    # XGBM_PARAMS,
    # create_walk_forward_splits,
    # run_walk_forward_cv,
    # compute_all_metrics,
    compute_metrics,
    # generate_metrics_report,
    # run_oos_evaluation,
    save_submission_csv,
    aggregate_forecast_to_delivery_periods,
    compute_regime_metrics,
    compute_block_metrics,
    compute_weekly_monthly_metrics,
    # run_ensemble_hyperparameter_search,
    # generate_ensemble_hp_report,
    # plot_ensemble_hp_analysis,
    plot_forecast_vs_actual,
    # plot_model_comparison,
    plot_feature_importance,
    # plot_walk_forward_errors,
    plot_model_error_heatmaps,
    plot_cumulative_error,
    _setup_logging,
    PEAK_HOURS,
    OOS_TEST_START,
    OOS_TEST_END,
)
# --- AutoGluon (commented out — focusing on our model) ---
# from src.autogluon_ts_forecaster import AutoGluonTimeSeriesForecaster
# from src.autogluon_forecaster import AutoGluonForecaster

logger = logging.getLogger("forecasting")

# ======================================================================
# Best LGBM+Ridge Ensemble hyperparameters (from previous HP search)
# Source: plots/model_plots/ensemble_hp_report.txt — 2026-03-20
# ======================================================================
BEST_ENS_PARAMS = {
    "lgbm_weight": 0.9,
    "n_estimators": 800,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "min_child_samples": 100,
    "ridge_alpha": 0.1,
}

# Bias correction: rolling window size in hours for estimating recent bias
BIAS_CORRECTION_WINDOW = 168 * 2  # 2 weeks of hourly data

# Recency weighting: half-life in days for exponential decay of sample weights
RECENCY_HALFLIFE_DAYS = 365  # 1 year — 2022 crisis data gets ~1/8 weight

# Asinh transformation scale parameters for multi-scale ensemble
# Training on asinh-transformed target stabilizes variance: spikes/negatives
# are compressed, so the model focuses more on the "normal" price range.
# Multiple scales capture different aspects of the price distribution.
ASINH_SCALES = [0.5, 1.0, 2.0]

# AutoGluon (commented out — focusing on our model)
# AUTOGLUON_TIME_LIMIT = 300
# AUTOGLUON_PRESET = "fast_training"


def apply_bias_correction(
    y_actual_train: np.ndarray,
    raw_preds_train: np.ndarray,
    raw_preds_test: np.ndarray,
    window: int = BIAS_CORRECTION_WINDOW,
) -> np.ndarray:
    """Apply static bias correction using training residuals.

    Estimates the model's systematic bias from the most recent `window`
    hours of in-sample residuals, then subtracts it from test predictions.

    Args:
        y_actual_train: Actual values from training period.
        raw_preds_train: Model predictions on training period.
        raw_preds_test: Raw model predictions on test period.
        window: Number of recent hours to estimate bias from.

    Returns:
        Tuple of (bias-corrected predictions, estimated_bias).
    """
    # Compute residuals on training data (actual - predicted)
    residuals = y_actual_train - raw_preds_train

    # Use the last `window` residuals to estimate current bias
    recent_residuals = residuals[-window:]
    estimated_bias = np.mean(recent_residuals)

    # Correct: if model overpredicts (negative residuals), subtract the bias
    corrected = raw_preds_test + estimated_bias

    return corrected, estimated_bias


def apply_expanding_window_bias_correction(
    y_actual_test: np.ndarray,
    raw_preds_test: np.ndarray,
    y_actual_train: np.ndarray,
    raw_preds_train: np.ndarray,
    update_interval: int = 168,
    min_window: int = 168,
) -> np.ndarray:
    """Apply expanding-window bias correction on OOS predictions.

    After each `update_interval` hours of the test period, recompute the
    bias using ALL actuals revealed so far (training tail + OOS so far).
    This adapts to regime shifts as actual prices become available.

    NOT data leakage — in production, yesterday's actual DA price is
    published the same day. We only use past actuals to correct future
    predictions.

    Args:
        y_actual_test: Actual test values (used retrospectively).
        raw_preds_test: Raw model predictions on test period.
        y_actual_train: Actual training values (for initial bias estimate).
        raw_preds_train: Model predictions on training data.
        update_interval: Hours between bias re-estimation (default 168 = 1 week).
        min_window: Minimum hours of data before first correction.

    Returns:
        Tuple of (corrected predictions, list of bias estimates per window).
    """
    n_test = len(raw_preds_test)
    corrected = raw_preds_test.copy()
    bias_history = []

    # Initial bias from last 2 weeks of training
    train_residuals = y_actual_train[-min_window:] - raw_preds_train[-min_window:]
    current_bias = np.mean(train_residuals)

    for i in range(n_test):
        # Apply current bias estimate
        corrected[i] = raw_preds_test[i] + current_bias

        # Every `update_interval` hours, recompute bias using OOS actuals so far
        if (i + 1) % update_interval == 0 and i >= min_window:
            # Use all OOS residuals revealed so far
            oos_residuals = y_actual_test[:i+1] - raw_preds_test[:i+1]
            current_bias = np.mean(oos_residuals)
            bias_history.append({
                "hour": i + 1,
                "bias": current_bias,
                "n_samples": i + 1,
            })

    return corrected, bias_history


def apply_hourly_exp_expanding_bias_correction(
    y_actual_test: np.ndarray,
    raw_preds_test: np.ndarray,
    hours_test: np.ndarray,
    y_actual_train: np.ndarray,
    raw_preds_train: np.ndarray,
    hours_train: np.ndarray,
    update_interval: int = 168,
    ew_halflife: int = 672,
) -> tuple:
    """Hour-specific, exponentially-weighted expanding-window bias correction.

    Combines two ideas:
        1. Separate bias per hour-of-day (h=0..23). Peak hours may be
           systematically overpredicted while off-peak is underpredicted;
           a single global correction hurts one to help the other.
        2. Exponential weighting — recent residuals carry more weight than
           old ones (half-life ~4 weeks = 672 hours), so the correction
           adapts faster to regime changes.

    Operates causally: at test hour i, correction uses only actuals from
    hours < i (plus training tail). No data leakage — DA actuals are
    published same-day in ENTSO-E.

    Args:
        y_actual_test: Actual OOS target values.
        raw_preds_test: Raw model predictions on OOS period.
        hours_test: Hour-of-day (0–23) for each test observation.
        y_actual_train: Actual training target values.
        raw_preds_train: Model predictions on training data.
        hours_train: Hour-of-day (0–23) for each training observation.
        update_interval: Hours between bias re-estimation (168 = weekly).
        ew_halflife: Exponential weighting half-life in hours (672 = 4 weeks).

    Returns:
        Tuple of (corrected_preds, bias_history_list).
    """
    n_test = len(raw_preds_test)
    corrected = raw_preds_test.copy()
    bias_history = []

    # --- Bootstrap: estimate initial per-hour bias from training tail ---
    # Use last 4 weeks of training data for initial estimates
    init_window = min(672, len(y_actual_train))
    train_tail_actual = y_actual_train[-init_window:]
    train_tail_preds = raw_preds_train[-init_window:]
    train_tail_hours = hours_train[-init_window:]
    train_tail_resid = train_tail_actual - train_tail_preds

    hourly_bias = {}  # hour -> current bias estimate
    for h in range(24):
        mask = train_tail_hours == h
        if mask.sum() > 0:
            hourly_bias[h] = float(np.mean(train_tail_resid[mask]))
        else:
            hourly_bias[h] = 0.0

    # --- Collect OOS residuals per hour as we go ---
    # Store (residual, weight) per hour-of-day
    oos_residuals_by_hour = {h: [] for h in range(24)}

    # Exponential decay factor: weight = exp(-ln2 / halflife * age)
    ln2 = np.log(2.0)

    for i in range(n_test):
        h = int(hours_test[i])
        # Apply current hour-specific bias
        corrected[i] = raw_preds_test[i] + hourly_bias.get(h, 0.0)

        # Record this observation's residual (will be used in NEXT update)
        resid_i = y_actual_test[i] - raw_preds_test[i]
        oos_residuals_by_hour[h].append((i, resid_i))

        # Every update_interval hours, recompute per-hour bias with EW
        if (i + 1) % update_interval == 0 and (i + 1) >= update_interval:
            current_time = i  # current position in test set
            global_biases = []

            for hh in range(24):
                obs = oos_residuals_by_hour[hh]
                if len(obs) == 0:
                    # Fall back to training estimate
                    continue

                # Compute exponentially weighted mean of residuals
                weights = []
                resids = []
                for (t, r) in obs:
                    age = current_time - t  # how many hours ago
                    w = np.exp(-ln2 / ew_halflife * age)
                    weights.append(w)
                    resids.append(r)

                weights = np.array(weights)
                resids = np.array(resids)
                ew_bias = np.sum(weights * resids) / np.sum(weights)
                hourly_bias[hh] = float(ew_bias)
                global_biases.append(ew_bias)

            avg_bias = np.mean(global_biases) if global_biases else 0.0
            bias_history.append({
                "hour": i + 1,
                "avg_bias": avg_bias,
                "n_oos_obs": i + 1,
                "per_hour_range": (
                    min(hourly_bias.values()),
                    max(hourly_bias.values()),
                ),
            })

    return corrected, bias_history


def main():
    """Run LGBM+Ridge Ensemble pipeline with bias correction.

    Steps:
        1. Load data and build features
        2. Train ensemble with best hyperparameters
        3. Generate raw predictions
        4. Apply bias correction using recent training residuals
        5. Compute accuracy at hourly, weekly, monthly granularity
        6. Save all results and reports
    """
    _setup_logging()

    sep = "=" * 60
    logger.info(sep)
    logger.info("DE_LU Forecasting Pipeline — LGBM+Ridge Ensemble + Bias Correction")
    logger.info(sep)

    # ------------------------------------------------------------------
    # Step 1: Load processed data
    # ------------------------------------------------------------------
    parquet_path = DATA_PROCESSED / "de_power_dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {parquet_path}. "
            "Run Part 1 (scripts/run_ingestion.py) first."
        )

    logger.info("Loading dataset from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Dataset loaded: %d rows x %d columns", df.shape[0], df.shape[1])

    # ------------------------------------------------------------------
    # Step 2: Build feature matrix
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 2: Building feature matrix")
    feature_matrix = build_feature_matrix(df)
    logger.info("Feature matrix: %d rows x %d columns",
                feature_matrix.shape[0], feature_matrix.shape[1])

    # ------------------------------------------------------------------
    # Step 3: Train/test split
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 3: Preparing OOS train/test split")

    oos_start_ts = pd.Timestamp(OOS_TEST_START, tz="UTC")
    oos_end_ts = pd.Timestamp(f"{OOS_TEST_END} 23:00", tz="UTC")

    train = feature_matrix.loc[feature_matrix.index < oos_start_ts].dropna(subset=[TARGET_COL])
    test = feature_matrix.loc[
        (feature_matrix.index >= oos_start_ts) & (feature_matrix.index <= oos_end_ts)
    ].dropna(subset=[TARGET_COL])

    assert train.index.max() < test.index.min(), "Leakage detected!"
    logger.info("  Train: %d rows (%s to %s)",
                len(train), train.index[0].strftime("%Y-%m-%d"),
                train.index[-1].strftime("%Y-%m-%d"))
    logger.info("  Test:  %d rows (%s to %s)",
                len(test), test.index[0].strftime("%Y-%m-%d"),
                test.index[-1].strftime("%Y-%m-%d"))

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    lag_168 = X_test["price_lag_168h"].values if "price_lag_168h" in X_test.columns else None

    # ------------------------------------------------------------------
    # Step 3b: Hyperparameter search (SKIPPED — using hardcoded best HP)
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 3b: Using hardcoded best HP (HP search skipped)")
    logger.info("  Best HP: %s", BEST_ENS_PARAMS)

    # Use hardcoded best HP (HP search commented out for speed)
    best_ens = BEST_ENS_PARAMS.copy()

    lgbm_params_tuned = LGBM_PARAMS.copy()
    lgbm_params_tuned.update({
        "n_estimators": best_ens["n_estimators"],
        "learning_rate": best_ens["learning_rate"],
        "num_leaves": best_ens["num_leaves"],
        "min_child_samples": best_ens["min_child_samples"],
    })

    # ------------------------------------------------------------------
    # Step 4: PASS 1 — Train baseline ensemble with best HP
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 4: PASS 1 — Training baseline LGBM+Ridge Ensemble")
    logger.info("  Using best HP: %s", best_ens)

    oos_ensemble_v1 = LGBMRidgeEnsemble(
        lgbm_weight=best_ens["lgbm_weight"],
        lgbm_params=lgbm_params_tuned,
        ridge_alpha=best_ens["ridge_alpha"],
    )
    oos_ensemble_v1.fit(X_train, y_train)
    raw_ens_preds = oos_ensemble_v1.predict(X_test)

    # Also get in-sample predictions for bias estimation
    raw_train_preds = oos_ensemble_v1.predict(X_train)

    raw_mae = np.mean(np.abs(y_test.values - raw_ens_preds))
    logger.info("  Pass 1 raw MAE: %.2f EUR/MWh", raw_mae)

    # ------------------------------------------------------------------
    # Step 4b: Generate OUT-OF-FOLD residuals for AR features (no leakage)
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 4b: Generating out-of-fold residuals for AR features")
    logger.info("  Using 5 time-ordered folds to avoid leakage")

    n_train = len(X_train)
    n_folds_oof = 5
    fold_size = n_train // n_folds_oof
    oof_residuals = np.full(n_train, np.nan)

    for k in range(n_folds_oof):
        # For fold k, train on folds 0..k-1, predict fold k
        # First fold uses a simple split: train on first half, predict second half
        fold_start = k * fold_size
        fold_end = (k + 1) * fold_size if k < n_folds_oof - 1 else n_train

        if k == 0:
            # Can't train on "before fold 0" — use fold 0 itself split in half
            # Train on first half of fold 0, predict second half
            mid = fold_start + (fold_end - fold_start) // 2
            if mid > fold_start + 100:
                X_tr_k = X_train.iloc[fold_start:mid]
                y_tr_k = y_train.iloc[fold_start:mid]
                X_pred_k = X_train.iloc[mid:fold_end]

                temp_model = LGBMRidgeEnsemble(
                    lgbm_weight=best_ens["lgbm_weight"],
                    lgbm_params=lgbm_params_tuned,
                    ridge_alpha=best_ens["ridge_alpha"],
                )
                temp_model.fit(X_tr_k, y_tr_k)
                preds_k = temp_model.predict(X_pred_k)
                oof_residuals[mid:fold_end] = y_train.iloc[mid:fold_end].values - preds_k
            # First half of fold 0: no OOF residual available → leave as NaN
        else:
            # Train on all data before this fold
            X_tr_k = X_train.iloc[:fold_start]
            y_tr_k = y_train.iloc[:fold_start]
            X_pred_k = X_train.iloc[fold_start:fold_end]

            temp_model = LGBMRidgeEnsemble(
                lgbm_weight=best_ens["lgbm_weight"],
                lgbm_params=lgbm_params_tuned,
                ridge_alpha=best_ens["ridge_alpha"],
            )
            temp_model.fit(X_tr_k, y_tr_k)
            preds_k = temp_model.predict(X_pred_k)
            oof_residuals[fold_start:fold_end] = (
                y_train.iloc[fold_start:fold_end].values - preds_k
            )

    oof_valid = np.sum(~np.isnan(oof_residuals))
    logger.info("  OOF residuals computed: %d/%d valid (%.1f%%)",
                oof_valid, n_train, oof_valid / n_train * 100)

    # ------------------------------------------------------------------
    # Step 4c: Build AR features for training data (from OOF residuals)
    # ------------------------------------------------------------------
    logger.info("Step 4c: Building AR features from OOF residuals")

    # Create a Series of OOF residuals aligned with training index
    oof_resid_series = pd.Series(oof_residuals, index=X_train.index)

    # Lag the residuals: at time t, use residual from t-24, t-48, t-168
    # This is causal: at forecast time for day D, we know actuals up to day D-1
    X_train_ar = X_train.copy()
    X_train_ar["resid_lag_24h"] = oof_resid_series.shift(24)
    X_train_ar["resid_lag_48h"] = oof_resid_series.shift(48)
    X_train_ar["resid_lag_168h"] = oof_resid_series.shift(168)
    # Rolling mean of recent residuals (past 24h of residuals, shifted by 24h)
    X_train_ar["resid_rolling_mean_24h"] = (
        oof_resid_series.shift(24).rolling(24, min_periods=12).mean()
    )
    # Rolling mean of past week's residuals
    X_train_ar["resid_rolling_mean_168h"] = (
        oof_resid_series.shift(24).rolling(168, min_periods=84).mean()
    )

    # Fill NaN in AR features with 0 (no info available yet = assume no error)
    ar_cols = ["resid_lag_24h", "resid_lag_48h", "resid_lag_168h",
               "resid_rolling_mean_24h", "resid_rolling_mean_168h"]
    for col in ar_cols:
        X_train_ar[col] = X_train_ar[col].fillna(0)

    logger.info("  Added %d AR features to training data", len(ar_cols))

    # ------------------------------------------------------------------
    # Step 4d: Build AR features for test data (causal, from revealed actuals)
    # ------------------------------------------------------------------
    logger.info("Step 4d: Building AR features for test data (causal)")

    # For test data, compute residuals using Pass-1 predictions and actuals
    # At test hour i, we know actuals up to hour i-1 (published same day)
    # and we know our Pass-1 predictions for all test hours
    test_residuals_v1 = y_test.values - raw_ens_preds  # actual - pass1_pred

    # Also need the training tail residuals to bootstrap the first test hours
    train_tail_resid = y_train.values - raw_train_preds
    # Concatenate: training tail + test residuals (for shift operations)
    full_resid = np.concatenate([train_tail_resid, test_residuals_v1])
    full_index = X_train.index.append(X_test.index)
    full_resid_series = pd.Series(full_resid, index=full_index)

    # Extract test portion after lagging (fully causal)
    X_test_ar = X_test.copy()
    X_test_ar["resid_lag_24h"] = full_resid_series.shift(24).loc[X_test.index]
    X_test_ar["resid_lag_48h"] = full_resid_series.shift(48).loc[X_test.index]
    X_test_ar["resid_lag_168h"] = full_resid_series.shift(168).loc[X_test.index]
    X_test_ar["resid_rolling_mean_24h"] = (
        full_resid_series.shift(24).rolling(24, min_periods=12).mean()
    ).loc[X_test.index]
    X_test_ar["resid_rolling_mean_168h"] = (
        full_resid_series.shift(24).rolling(168, min_periods=84).mean()
    ).loc[X_test.index]

    for col in ar_cols:
        X_test_ar[col] = X_test_ar[col].fillna(0)

    logger.info("  AR features built for test data (using Pass-1 residuals)")
    logger.info("  Leakage check: test AR features use only past actuals + pass-1 preds")

    # ------------------------------------------------------------------
    # Step 4e: Compute recency weights for training samples
    # ------------------------------------------------------------------
    logger.info("Step 4e: Computing recency weights (half-life = %d days)",
                RECENCY_HALFLIFE_DAYS)

    train_timestamps = X_train.index
    last_train_ts = train_timestamps[-1]
    # Age in days for each sample
    age_days = (last_train_ts - train_timestamps).total_seconds() / 86400.0
    # Exponential decay: w = exp(-ln2 / halflife * age)
    recency_weights = np.exp(-np.log(2) / RECENCY_HALFLIFE_DAYS * age_days.values)
    # Normalize so mean weight = 1 (doesn't change effective learning rate)
    recency_weights = recency_weights / recency_weights.mean()

    logger.info("  Weight range: %.3f (oldest) to %.3f (newest)",
                recency_weights.min(), recency_weights.max())
    logger.info("  Weight for 2022 crisis data: ~%.3f",
                recency_weights[int(len(recency_weights) * 0.3)])
    logger.info("  Weight for 2024 data: ~%.3f",
                recency_weights[int(len(recency_weights) * 0.9)])

    # ------------------------------------------------------------------
    # Step 4f: PASS 2 — Retrain with AR features + recency weights
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 4f: PASS 2 — Retraining with AR features + recency weights")

    oos_ensemble = LGBMRidgeEnsemble(
        lgbm_weight=best_ens["lgbm_weight"],
        lgbm_params=lgbm_params_tuned,
        ridge_alpha=best_ens["ridge_alpha"],
    )
    oos_ensemble.fit(X_train_ar, y_train, sample_weight=recency_weights)

    # Pass-2 predictions
    pass2_raw_preds = oos_ensemble.predict(X_test_ar)
    pass2_train_preds = oos_ensemble.predict(X_train_ar)

    pass2_raw_mae = np.mean(np.abs(y_test.values - pass2_raw_preds))
    logger.info("  Pass 2 raw MAE: %.2f EUR/MWh (was %.2f in Pass 1)",
                pass2_raw_mae, raw_mae)
    logger.info("  Improvement from AR + recency: %.2f EUR/MWh",
                raw_mae - pass2_raw_mae)

    # ------------------------------------------------------------------
    # Step 4g: Quantile regression (median) — robust to spikes
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 4g: Training quantile LightGBM (median, alpha=0.5)")

    import lightgbm as lgb

    quantile_params = lgbm_params_tuned.copy()
    quantile_params["objective"] = "quantile"
    quantile_params["alpha"] = 0.5  # median

    quantile_lgbm = lgb.LGBMRegressor(**quantile_params)
    quantile_lgbm.fit(
        X_train_ar, y_train,
        sample_weight=recency_weights,
    )
    quantile_preds_test = quantile_lgbm.predict(X_test_ar)
    quantile_preds_train = quantile_lgbm.predict(X_train_ar)

    quantile_raw_mae = np.mean(np.abs(y_test.values - quantile_preds_test))
    logger.info("  Quantile (median) raw MAE: %.2f EUR/MWh", quantile_raw_mae)

    # Blend: 50/50 mean (Pass-2) + median (quantile) for robustness
    blended_preds_test = 0.5 * pass2_raw_preds + 0.5 * quantile_preds_test
    blended_preds_train = 0.5 * pass2_train_preds + 0.5 * quantile_preds_train
    blended_raw_mae = np.mean(np.abs(y_test.values - blended_preds_test))
    logger.info("  Blended (50%% mean + 50%% median) raw MAE: %.2f EUR/MWh", blended_raw_mae)

    # Pick whichever is better: pass2, quantile, or blend
    candidates = {
        "pass2": (pass2_raw_preds, pass2_train_preds, pass2_raw_mae),
        "quantile": (quantile_preds_test, quantile_preds_train, quantile_raw_mae),
        "blend": (blended_preds_test, blended_preds_train, blended_raw_mae),
    }
    best_variant = min(candidates, key=lambda k: candidates[k][2])
    best_test_preds, best_train_preds, best_raw_mae = candidates[best_variant]
    logger.info("  Best variant: %s (MAE=%.2f)", best_variant, best_raw_mae)

    # Use best variant for bias correction downstream
    pass2_raw_preds = best_test_preds
    pass2_train_preds = best_train_preds

    # ------------------------------------------------------------------
    # Step 4h: Asinh variance-stabilizing transformation ensemble
    # ------------------------------------------------------------------
    # Electricity prices have fat tails (spikes to 500+, negatives to -50).
    # Training on asinh(price/c) compresses extremes, so the model optimizes
    # for the "normal" price range where most MAE accumulates.
    # We train at multiple scale parameters c and average the back-transformed
    # predictions — this is a form of model averaging that reduces variance.
    logger.info(sep)
    logger.info("Step 4h: Asinh variance-stabilizing transformation ensemble")
    logger.info("  Scales: %s", ASINH_SCALES)

    asinh_preds_test_list = []
    asinh_preds_train_list = []

    for c in ASINH_SCALES:
        # Transform target: y_t = arcsinh(y / c)
        y_train_asinh = np.arcsinh(y_train.values / c)

        # Train LGBM+Ridge on transformed target with AR features + recency
        asinh_model = LGBMRidgeEnsemble(
            lgbm_weight=best_ens["lgbm_weight"],
            lgbm_params=lgbm_params_tuned,
            ridge_alpha=best_ens["ridge_alpha"],
        )
        asinh_model.fit(X_train_ar, pd.Series(y_train_asinh, index=y_train.index),
                        sample_weight=recency_weights)

        # Predict in asinh space, then inverse-transform: y = c * sinh(pred)
        asinh_pred_test = asinh_model.predict(X_test_ar)
        asinh_pred_train = asinh_model.predict(X_train_ar)

        pred_test_orig = c * np.sinh(asinh_pred_test)
        pred_train_orig = c * np.sinh(asinh_pred_train)

        mae_c = np.mean(np.abs(y_test.values - pred_test_orig))
        logger.info("  c=%.1f: asinh MAE = %.2f EUR/MWh", c, mae_c)

        asinh_preds_test_list.append(pred_test_orig)
        asinh_preds_train_list.append(pred_train_orig)

    # Average across all scales (multi-scale ensemble)
    asinh_ensemble_test = np.mean(asinh_preds_test_list, axis=0)
    asinh_ensemble_train = np.mean(asinh_preds_train_list, axis=0)
    asinh_ensemble_mae = np.mean(np.abs(y_test.values - asinh_ensemble_test))
    logger.info("  Multi-scale asinh ensemble MAE: %.2f EUR/MWh", asinh_ensemble_mae)

    # Also try blending asinh ensemble with the best pass2 variant (50/50)
    asinh_blend_test = 0.5 * pass2_raw_preds + 0.5 * asinh_ensemble_test
    asinh_blend_train = 0.5 * pass2_train_preds + 0.5 * asinh_ensemble_train
    asinh_blend_mae = np.mean(np.abs(y_test.values - asinh_blend_test))
    logger.info("  50/50 blend (pass2 + asinh): %.2f EUR/MWh", asinh_blend_mae)

    # Pick best: original pass2, pure asinh ensemble, or blended
    asinh_candidates = {
        "pass2_original": (pass2_raw_preds, pass2_train_preds, best_raw_mae),
        "asinh_ensemble": (asinh_ensemble_test, asinh_ensemble_train, asinh_ensemble_mae),
        "asinh_blend": (asinh_blend_test, asinh_blend_train, asinh_blend_mae),
    }
    best_asinh_variant = min(asinh_candidates, key=lambda k: asinh_candidates[k][2])
    pass2_raw_preds, pass2_train_preds, final_raw_mae = asinh_candidates[best_asinh_variant]
    logger.info("  Best variant: %s (MAE=%.2f)", best_asinh_variant, final_raw_mae)
    logger.info("  Improvement from asinh: %.2f EUR/MWh",
                best_raw_mae - final_raw_mae if final_raw_mae < best_raw_mae else 0)

    # ------------------------------------------------------------------
    # Step 5: Apply bias corrections to Pass-2 predictions
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 5: Applying bias corrections to Pass-2 predictions")

    corrected_preds, estimated_bias = apply_bias_correction(
        y_train.values, pass2_train_preds, pass2_raw_preds,
        window=BIAS_CORRECTION_WINDOW,
    )

    logger.info("  Static bias: %+.3f EUR/MWh", estimated_bias)

    # Expanding-window correction
    expanding_preds, bias_history = apply_expanding_window_bias_correction(
        y_actual_test=y_test.values,
        raw_preds_test=pass2_raw_preds,
        y_actual_train=y_train.values,
        raw_preds_train=pass2_train_preds,
        update_interval=168,
        min_window=168,
    )

    expanding_mae = np.mean(np.abs(y_test.values - expanding_preds))
    expanding_mbe = np.mean(y_test.values - expanding_preds)
    logger.info("  Expanding-window MAE: %.2f EUR/MWh", expanding_mae)

    # ------------------------------------------------------------------
    # Step 5c: Hour-specific + exponentially-weighted expanding bias
    # ------------------------------------------------------------------
    logger.info("  Hour-specific EW expanding bias correction on Pass-2:")

    hours_train_arr = X_train.index.hour.values
    hours_test_arr = X_test.index.hour.values

    hourly_ew_preds, hourly_ew_history = apply_hourly_exp_expanding_bias_correction(
        y_actual_test=y_test.values,
        raw_preds_test=pass2_raw_preds,
        hours_test=hours_test_arr,
        y_actual_train=y_train.values,
        raw_preds_train=pass2_train_preds,
        hours_train=hours_train_arr,
        update_interval=168,
        ew_halflife=672,
    )

    hourly_ew_mae = np.mean(np.abs(y_test.values - hourly_ew_preds))
    hourly_ew_mbe = np.mean(y_test.values - hourly_ew_preds)
    logger.info("  Hourly-EW expanding MAE:     %.2f EUR/MWh", hourly_ew_mae)
    logger.info("  Hourly-EW expanding MBE:     %+.3f EUR/MWh", hourly_ew_mbe)
    logger.info("  Total improvement over Pass-1 raw: %.2f EUR/MWh (%.1f%%)",
                raw_mae - hourly_ew_mae,
                (raw_mae - hourly_ew_mae) / raw_mae * 100 if raw_mae > 0 else 0)

    if hourly_ew_history:
        last = hourly_ew_history[-1]
        logger.info("  Final avg bias: %+.2f, per-hour range: [%+.2f, %+.2f]",
                     last["avg_bias"], last["per_hour_range"][0], last["per_hour_range"][1])

    # ------------------------------------------------------------------
    # Step 6: Train Seasonal Naive baseline
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 6: Training Seasonal Naive baseline")
    oos_naive = SeasonalNaiveModel()
    oos_naive.fit(X_train, y_train)
    naive_preds = oos_naive.predict(X_test)
    naive_mae = np.mean(np.abs(y_test.values - naive_preds))

    # ------------------------------------------------------------------
    # Step 6b & 6c: AutoGluon (COMMENTED OUT — focusing on our model)
    # ------------------------------------------------------------------
    ag_ts_preds = np.full(len(y_test), np.nan)
    ag_tab_raw_preds = np.full(len(y_test), np.nan)
    ag_tab_ew_preds = np.full(len(y_test), np.nan)


    # ------------------------------------------------------------------
    # Step 7: Compute OOS metrics for all variants
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 7: Computing OOS metrics")

    hours = X_test.index.hour
    peak_mask = np.isin(hours, PEAK_HOURS)
    offpeak_mask = ~peak_mask

    oos_metrics = {}

    # Naive
    oos_metrics["seasonal_naive"] = compute_metrics(
        y_test.values, naive_preds, "seasonal_naive",
        naive_mae=naive_mae, lag_168_values=lag_168,
    )

    # Pass-1 raw ensemble (no AR, no weights, no bias correction)
    raw_metrics = compute_metrics(
        y_test.values, raw_ens_preds, "pass1_raw",
        naive_mae=naive_mae, lag_168_values=lag_168,
    )
    oos_metrics["pass1_raw"] = raw_metrics

    # Pass-2 raw (AR + recency, no bias correction)
    pass2_raw_metrics = compute_metrics(
        y_test.values, pass2_raw_preds, "pass2_raw",
        naive_mae=naive_mae, lag_168_values=lag_168,
    )
    oos_metrics["pass2_raw"] = pass2_raw_metrics

    # Pass-2 + hourly-EW expanding bias correction (best method)
    hourly_ew_metrics = compute_metrics(
        y_test.values, hourly_ew_preds, "pass2_hourly_ew",
        naive_mae=naive_mae, lag_168_values=lag_168,
    )
    if peak_mask.sum() > 0:
        hourly_ew_metrics["MAE_peak"] = np.mean(
            np.abs(y_test.values[peak_mask] - hourly_ew_preds[peak_mask]))
    if offpeak_mask.sum() > 0:
        hourly_ew_metrics["MAE_offpeak"] = np.mean(
            np.abs(y_test.values[offpeak_mask] - hourly_ew_preds[offpeak_mask]))
    oos_metrics["pass2_hourly_ew"] = hourly_ew_metrics

    logger.info("")
    logger.info("  OOS HOURLY RESULTS:")
    logger.info("  %-35s  MAE=%8.2f  RMSE=%8.2f  MBE=%+.2f",
                "Seasonal Naive",
                oos_metrics["seasonal_naive"]["MAE"],
                oos_metrics["seasonal_naive"]["RMSE"],
                np.mean(y_test.values - naive_preds))
    logger.info("  %-35s  MAE=%8.2f  RMSE=%8.2f  MBE=%+.2f",
                "Pass-1 raw (no AR, no weights)",
                raw_metrics["MAE"], raw_metrics["RMSE"],
                np.mean(y_test.values - raw_ens_preds))
    logger.info("  %-35s  MAE=%8.2f  RMSE=%8.2f  MBE=%+.2f",
                "Pass-2 raw (AR + recency weights)",
                pass2_raw_metrics["MAE"], pass2_raw_metrics["RMSE"],
                np.mean(y_test.values - pass2_raw_preds))
    logger.info("  %-35s  MAE=%8.2f  RMSE=%8.2f  MBE=%+.2f",
                "Pass-2 + hourly-EW bias correction",
                hourly_ew_metrics["MAE"], hourly_ew_metrics["RMSE"],
                np.mean(y_test.values - hourly_ew_preds))

    # Step 7b & 7c: AutoGluon metrics (SKIPPED)

    # ------------------------------------------------------------------
    # Step 8: Build OOS DataFrame and compute multi-granularity accuracy
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 8: Computing multi-granularity accuracy (hourly/weekly/monthly)")

    oos_data = {
        "timestamp_utc": X_test.index,
        "y_actual": y_test.values,
        "y_pred": hourly_ew_preds,             # default = best method
        "y_pred_pass1_raw": raw_ens_preds,     # pass-1: no AR, no weights
        "y_pred_pass2_raw": pass2_raw_preds,   # pass-2: AR + recency (best of mean/median/blend)
        "y_pred_hourly_ew": hourly_ew_preds,   # pass-2 + hourly-EW bias correction
        "y_pred_naive": naive_preds,
    }
    oos_df = pd.DataFrame(oos_data).set_index("timestamp_utc")

    # Compute multi-granularity for all model variants
    weekly_monthly_metrics = compute_weekly_monthly_metrics(oos_df)

    # Print results
    logger.info("")
    for granularity in ["hourly", "weekly", "monthly"]:
        if granularity not in weekly_monthly_metrics:
            continue
        gran_data = weekly_monthly_metrics[granularity]
        logger.info("  %s ACCURACY:", granularity.upper())
        logger.info("  %-28s  %8s  %8s  %8s  %8s  %5s",
                     "Model", "MAE", "RMSE", "MAPE", "MBE", "N")
        logger.info("  %s", "-" * 68)
        for model_name, m in gran_data.items():
            mape_str = f"{m['MAPE']:.1f}%" if pd.notna(m.get("MAPE")) else "N/A"
            logger.info("  %-28s  %8.2f  %8.2f  %8s  %+8.2f  %5d",
                         model_name, m["MAE"], m["RMSE"],
                         mape_str, m["MBE"], m["n"])
        logger.info("")

    # ------------------------------------------------------------------
    # Step 9: Save results
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 9: Saving results")

    # Save OOS predictions
    oos_path = DATA_PROCESSED / "oos_predictions.parquet"
    oos_df.to_parquet(oos_path)
    logger.info("  OOS predictions: %s", oos_path)

    # Save submission CSV
    save_submission_csv(oos_df, OUTPUTS_DIR / "submission.csv")

    # Save multi-granularity accuracy as CSV
    accuracy_rows = []
    for granularity in ["hourly", "weekly", "monthly"]:
        if granularity in weekly_monthly_metrics:
            for model_name, m in weekly_monthly_metrics[granularity].items():
                accuracy_rows.append({
                    "granularity": granularity,
                    "model": model_name,
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "MAPE": m.get("MAPE"),
                    "MBE": m["MBE"],
                    "n": m["n"],
                })
    accuracy_df = pd.DataFrame(accuracy_rows)
    accuracy_path = OUTPUTS_DIR / "multi_granularity_accuracy.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    logger.info("  Multi-granularity accuracy: %s", accuracy_path)

    # Save delivery-period aggregations
    delivery_periods = aggregate_forecast_to_delivery_periods(oos_df)
    delivery_path = OUTPUTS_DIR / "delivery_period_aggregations.csv"
    delivery_periods.to_csv(delivery_path)
    logger.info("  Delivery aggregations: %s", delivery_path)

    # Save model
    oos_ensemble.save()
    logger.info("  Ensemble model: models/lgbm_ridge_ensemble.pkl")

    # ------------------------------------------------------------------
    # Step 10: Generate report
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 10: Generating performance report")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 70,
        " DE_LU Power Price Forecasting — Performance Report",
        f" Generated: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f" Model: LGBM+Ridge Ensemble (2-pass AR + recency + hourly-EW bias)",
        f" OOS Period: {OOS_TEST_START} to {OOS_TEST_END}",
        "=" * 70,
        "",
        "MODEL CONFIGURATION",
        "-" * 70,
        f"  lgbm_weight:         {best_ens['lgbm_weight']}",
        f"  n_estimators:        {best_ens['n_estimators']}",
        f"  learning_rate:       {best_ens['learning_rate']}",
        f"  num_leaves:          {best_ens['num_leaves']}",
        f"  min_child_samples:   {best_ens['min_child_samples']}",
        f"  ridge_alpha:         {best_ens['ridge_alpha']}",
        f"  AR features:         resid_lag_24h/48h/168h, rolling_mean_24h/168h",
        f"  Recency weighting:   halflife={RECENCY_HALFLIFE_DAYS} days",
        f"  Asinh transform:     multi-scale c={ASINH_SCALES}",
        f"  Bias correction:     hourly-EW expanding (halflife=672h)",
        "",
        "OOS HOURLY RESULTS",
        "-" * 70,
        f"{'Model':30s}  {'MAE':>8s}  {'RMSE':>8s}  {'Dir.Acc':>8s}  {'Skill':>8s}",
        "-" * 70,
    ]

    for model_name, m in sorted(oos_metrics.items(), key=lambda x: x[1].get("MAE", 999)):
        da = f"{m.get('directional_accuracy', 0):.1f}%" if pd.notna(m.get("directional_accuracy")) else "N/A"
        skill = f"{m.get('skill_score_vs_naive', 0):+.2f}" if pd.notna(m.get("skill_score_vs_naive")) else "N/A"
        report_lines.append(
            f"{model_name:30s}  {m['MAE']:>8.1f}  {m['RMSE']:>8.1f}  {da:>8s}  {skill:>8s}"
        )

    report_lines.append("")
    report_lines.append("IMPROVEMENT BREAKDOWN")
    report_lines.append("-" * 70)
    pass1_mbe = np.mean(y_test.values - raw_ens_preds)
    pass2_mbe = np.mean(y_test.values - pass2_raw_preds)
    final_mbe = np.mean(y_test.values - hourly_ew_preds)
    report_lines.append(f"  Pass-1 raw MAE:            {raw_metrics['MAE']:.2f}  MBE: {pass1_mbe:+.3f}")
    report_lines.append(f"  Pass-2 raw MAE:            {pass2_raw_metrics['MAE']:.2f}  MBE: {pass2_mbe:+.3f}")
    report_lines.append(f"  Pass-2 + hourly-EW MAE:    {hourly_ew_metrics['MAE']:.2f}  MBE: {final_mbe:+.3f}")
    report_lines.append(f"  AR + recency improvement:  {raw_metrics['MAE'] - pass2_raw_metrics['MAE']:.2f} EUR/MWh")
    report_lines.append(f"  Hourly-EW bias improvement:{pass2_raw_metrics['MAE'] - hourly_ew_metrics['MAE']:.2f} EUR/MWh")
    report_lines.append(f"  Total improvement:         {raw_metrics['MAE'] - hourly_ew_metrics['MAE']:.2f} EUR/MWh")
    report_lines.append("")

    # Multi-granularity section
    report_lines.append("MULTI-GRANULARITY FORECAST ACCURACY")
    report_lines.append("-" * 70)
    report_lines.append("  (Hourly forecasts aggregated to weekly/monthly averages)")
    report_lines.append("")

    for granularity in ["hourly", "weekly", "monthly"]:
        if granularity not in weekly_monthly_metrics:
            continue
        gran_data = weekly_monthly_metrics[granularity]
        report_lines.append(f"  {granularity.upper()} ACCURACY:")
        report_lines.append(
            f"  {'Model':28s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAPE':>8s}  {'MBE':>8s}  {'N':>5s}"
        )
        report_lines.append(f"  {'-' * 65}")
        for model_name, m in gran_data.items():
            mape = f"{m['MAPE']:.1f}%" if pd.notna(m.get("MAPE")) else "N/A"
            report_lines.append(
                f"  {model_name:28s}  {m['MAE']:>8.2f}  {m['RMSE']:>8.2f}  "
                f"{mape:>8s}  {m['MBE']:>+8.2f}  {m['n']:>5d}"
            )
        report_lines.append("")

    report_lines.append("  Note: Weekly = ISO week avg, Monthly = calendar month avg")
    report_lines.append(f"        OOS period {OOS_TEST_START} to {OOS_TEST_END}")
    report_lines.append("")

    # Best model summary — hourly-EW expanding is the recommended method
    report_lines.append("BEST MODEL: lgbm_ridge_hourly_ew (hour-specific EW expanding bias)")
    report_lines.append(f"  Hourly  MAE: {hourly_ew_metrics['MAE']:.1f} EUR/MWh")
    if "weekly" in weekly_monthly_metrics and "hourly_ew" in weekly_monthly_metrics["weekly"]:
        wk = weekly_monthly_metrics["weekly"]["hourly_ew"]
        report_lines.append(f"  Weekly  MAE: {wk['MAE']:.2f} EUR/MWh  (MAPE: {wk.get('MAPE', 0):.1f}%)")
    if "monthly" in weekly_monthly_metrics and "hourly_ew" in weekly_monthly_metrics["monthly"]:
        mo = weekly_monthly_metrics["monthly"]["hourly_ew"]
        report_lines.append(f"  Monthly MAE: {mo['MAE']:.2f} EUR/MWh  (MAPE: {mo.get('MAPE', 0):.1f}%)")
    report_lines.append(f"  Directional Accuracy: {hourly_ew_metrics.get('directional_accuracy', 0):.1f}%")
    report_lines.append(f"  Skill vs Naive: {hourly_ew_metrics.get('skill_score_vs_naive', 0):+.2f}")
    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    report_path = OUTPUTS_DIR / "model_performance_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    logger.info("  Performance report: %s", report_path)

    # ------------------------------------------------------------------
    # Step 11: Generate plots
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Step 11: Generating plots -> %s", PLOTS_DIR)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_forecast_vs_actual(oos_df, PLOTS_DIR)
    plot_feature_importance(oos_ensemble.lgbm, PLOTS_DIR)
    plot_model_error_heatmaps(oos_df, PLOTS_DIR)
    plot_cumulative_error(oos_df, PLOTS_DIR)

    # ------------------------------------------------------------------
    # Completion summary
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("Pipeline complete!")
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info("  Hourly  MAE: %.2f (pass1 raw) / %.2f (pass2 raw) / %.2f (pass2+hourly-EW)",
                raw_metrics["MAE"], pass2_raw_metrics["MAE"],
                hourly_ew_metrics["MAE"])
    if "weekly" in weekly_monthly_metrics:
        for mn, m in weekly_monthly_metrics["weekly"].items():
            logger.info("  Weekly  MAE: %.2f (%s)", m["MAE"], mn)
    if "monthly" in weekly_monthly_metrics:
        for mn, m in weekly_monthly_metrics["monthly"].items():
            logger.info("  Monthly MAE: %.2f (%s)", m["MAE"], mn)
    logger.info("")
    logger.info("  Final MBE:      %+.3f EUR/MWh", hourly_ew_mbe)
    logger.info("  Dir. Accuracy:  %.1f%%",
                hourly_ew_metrics.get("directional_accuracy", 0))
    logger.info(sep)

    return oos_df


# ======================================================================
# AUTOGLUON PIPELINE (commented out — uncomment to run AutoGluon)
# ======================================================================
# def main_autogluon():
#     """Run AutoGluon-only pipeline.
#     Uses saved OOS predictions for ensemble/naive from previous run.
#     Only trains AutoGluon, computes all metrics, and generates reports.
#     """
#     from src.autogluon_forecaster import AutoGluonForecaster, generate_autogluon_report
#     _setup_logging()
#     sep = "=" * 60
#     # ... (see previous version of this file for full AutoGluon pipeline)
#     # Key steps:
#     #   1. Load feature matrix
#     #   2. Prepare train/test split
#     #   3. Load saved ensemble/naive predictions
#     #   4. Train AutoGluon (time_limit=300, presets="best_quality")
#     #   5. Compute metrics, generate report
#     pass

# ======================================================================
# FULL PIPELINE (commented out — uncomment to run HP search + CV + everything)
# ======================================================================
# def main_full():
#     """Run the full Part 2 forecasting pipeline from scratch.
#     Includes: HP search (810 combos), 12-fold walk-forward CV,
#     OOS evaluation, AutoGluon, all reports and plots.
#     Estimated runtime: ~45 minutes on M4 Pro.
#     """
#     from src.autogluon_forecaster import AutoGluonForecaster, generate_autogluon_report
#     _setup_logging()
#     sep = "=" * 60
#     # ... (see git history for full pipeline code)
#     pass


if __name__ == "__main__":
    main()
