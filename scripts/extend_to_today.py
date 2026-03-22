#!/usr/bin/env python3
"""
Extend the pipeline dataset to today's date for demonstration.

Fetches latest ENTSO-E data (DA prices, wind/solar, load) from where
the existing dataset ends up to today.  Gas TTF — which requires a
manual download — is forward-filled from the last available value.

Then re-runs feature engineering, model inference (using the saved
ensemble), and curve translation so the report can target today.

Usage:
    python scripts/extend_to_today.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_feature_matrix
from src.ingestion import (
    DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR, MARKET, TIMEZONE,
    fetch_da_prices, fetch_wind_solar, fetch_load_forecast,
    build_complete_hourly_index, clean_and_align, load_api_key,
)
from entsoe import EntsoePandasClient

logger = logging.getLogger("extend_to_today")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# Paths
DATASET_PATH = DATA_PROCESSED / "de_power_dataset.parquet"
OOS_PATH = DATA_PROCESSED / "oos_predictions.parquet"
FEATURE_MATRIX_PATH = DATA_PROCESSED / "feature_matrix.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_ridge_ensemble.pkl"


def main():
    sep = "=" * 60
    logger.info(sep)
    logger.info("Extending pipeline dataset to today")
    logger.info(sep)

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    logger.info("Today: %s", today_str)

    # ── 1. Load existing dataset ─────────────────────────────────────
    existing = pd.read_parquet(DATASET_PATH)
    last_date = existing.index.max()
    logger.info("Existing dataset ends at: %s", last_date)

    today_ts = pd.Timestamp(today_str, tz="UTC")

    # If dataset was previously extended with forward-filled data,
    # truncate back to the last date with real API-sourced data.
    # Detect: if last N rows of DA price are all identical → forward-filled
    if last_date >= today_ts:
        tail_prices = existing["da_price_eur_mwh"].tail(48)
        if tail_prices.nunique() <= 1:
            # Forward-filled junk — find where real data ends
            da = existing["da_price_eur_mwh"]
            # Walk backwards to find where values start changing
            diffs = da.diff().abs()
            real_end_idx = diffs[diffs > 0].index[-1]
            logger.info("Detected forward-filled extension — truncating back to %s", real_end_idx)
            existing = existing.loc[:real_end_idx]
            last_date = existing.index.max()
            existing.to_parquet(DATASET_PATH)
            logger.info("Truncated dataset saved: %d rows, ends at %s", len(existing), last_date)
        else:
            logger.info("Dataset already covers today with real data — skipping extension")
            _run_curve_translation()
            return

    # ── 2. Fetch new ENTSO-E data ────────────────────────────────────
    extend_start = (last_date + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    extend_end = today_str
    logger.info("Fetching ENTSO-E data: %s → %s", extend_start, extend_end)

    # Delete current-year cache files to force fresh API fetch
    current_year = datetime.utcnow().year
    for subdir in ["da_prices", "wind_solar", "load_forecast"]:
        cache_dir = DATA_RAW / subdir
        if cache_dir.exists():
            for f in cache_dir.glob(f"*_{current_year}.parquet"):
                f.unlink()
                logger.info("  Deleted stale cache: %s", f.name)

    api_key = load_api_key()
    client = EntsoePandasClient(api_key=api_key)

    start_tz = pd.Timestamp(extend_start, tz=TIMEZONE)
    end_tz = pd.Timestamp(f"{extend_end} 23:00", tz=TIMEZONE)

    # Fetch each series — graceful failures
    da_new = None
    wind_solar_new = None
    load_new = None

    try:
        da_new = fetch_da_prices(client, start_tz, end_tz, DATA_RAW)
        logger.info("  DA prices: %d rows", len(da_new))
    except Exception as e:
        logger.warning("  DA prices fetch failed: %s", e)

    try:
        wind_solar_new = fetch_wind_solar(client, start_tz, end_tz, DATA_RAW)
        logger.info("  Wind/Solar: %d rows", len(wind_solar_new))
    except Exception as e:
        logger.warning("  Wind/Solar fetch failed: %s", e)

    try:
        load_new = fetch_load_forecast(client, start_tz, end_tz, DATA_RAW)
        logger.info("  Load forecast: %d rows", len(load_new))
    except Exception as e:
        logger.warning("  Load forecast fetch failed: %s", e)

    if da_new is None or len(da_new) == 0:
        logger.error("Cannot extend — no DA prices available from API")
        logger.info("Falling back: will use latest available date for report")
        _run_curve_translation()
        return

    # ── 3. Build extended hourly index ───────────────────────────────
    new_index = pd.date_range(
        start=last_date + pd.Timedelta(hours=1),
        end=pd.Timestamp(f"{extend_end} 23:00", tz="UTC"),
        freq="h",
    )
    logger.info("New hourly slots: %d", len(new_index))

    # ── 4. Align new data into a DataFrame ───────────────────────────
    new_df = pd.DataFrame(index=new_index)
    new_df.index.name = "timestamp_utc"

    # DA prices (Series)
    if da_new is not None:
        new_df["da_price_eur_mwh"] = da_new.reindex(new_index)
    else:
        new_df["da_price_eur_mwh"] = np.nan

    # Wind & Solar (fetch_wind_solar returns a DataFrame with 2 cols)
    if wind_solar_new is not None:
        if isinstance(wind_solar_new, pd.DataFrame):
            for col in ["wind_forecast_mw", "solar_forecast_mw"]:
                if col in wind_solar_new.columns:
                    new_df[col] = wind_solar_new[col].reindex(new_index)
        else:
            # Series fallback
            new_df["wind_forecast_mw"] = wind_solar_new.reindex(new_index)

    if "wind_forecast_mw" not in new_df.columns:
        new_df["wind_forecast_mw"] = np.nan
    if "solar_forecast_mw" not in new_df.columns:
        new_df["solar_forecast_mw"] = np.nan

    # Load forecast (Series)
    if load_new is not None:
        new_df["load_forecast_mw"] = load_new.reindex(new_index)
    else:
        new_df["load_forecast_mw"] = np.nan

    # Gas: forward-fill from last available value
    last_gas = existing["gas_price_eur_mwh"].dropna().iloc[-1]
    new_df["gas_price_eur_mwh"] = last_gas
    logger.info("  Gas TTF forward-filled at: %.3f €/MWh", last_gas)

    # ── 5. Fill any remaining NaNs with forward-fill from existing ───
    for col in existing.columns:
        if col in new_df.columns:
            nan_count = new_df[col].isna().sum()
            if nan_count > 0:
                # Try forward-fill from the tail of existing data
                last_valid = existing[col].dropna().iloc[-1] if existing[col].notna().any() else 0
                new_df[col] = new_df[col].ffill().fillna(last_valid)
                logger.info("  %s: filled %d NaN values", col, nan_count)

    # ── 6. Concatenate with existing dataset ─────────────────────────
    extended = pd.concat([existing, new_df[existing.columns]])
    # Remove any duplicate timestamps
    extended = extended[~extended.index.duplicated(keep="first")]
    extended = extended.sort_index()

    logger.info("Extended dataset: %d rows (%s → %s)",
                len(extended), extended.index.min(), extended.index.max())

    # Save
    extended.to_parquet(DATASET_PATH)
    logger.info("Saved extended dataset to %s", DATASET_PATH)

    # ── 7. Rebuild feature matrix ────────────────────────────────────
    logger.info(sep)
    logger.info("Rebuilding feature matrix")
    feature_matrix = build_feature_matrix(extended)
    logger.info("Feature matrix: %d rows x %d cols", *feature_matrix.shape)

    # ── 8. Load saved model and predict on new rows ──────────────────
    logger.info(sep)
    logger.info("Loading saved ensemble model")

    import joblib
    model_dict = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s (keys: %s)", MODEL_PATH, list(model_dict.keys()))

    # Load existing OOS predictions
    existing_oos = pd.read_parquet(OOS_PATH)
    last_oos_ts = existing_oos.index.max()
    logger.info("Existing OOS predictions end at: %s", last_oos_ts)

    # Get new rows that need predictions
    new_rows = feature_matrix.loc[feature_matrix.index > last_oos_ts]
    if len(new_rows) == 0:
        logger.info("No new rows to predict — OOS predictions already up to date")
        _run_curve_translation()
        return

    logger.info("Predicting %d new hourly rows", len(new_rows))

    target_col = "da_price_eur_mwh"
    X_new = new_rows.drop(columns=[target_col], errors="ignore")

    # Get predictions from saved ensemble components
    lgbm_model = model_dict["lgbm_model"]
    lgbm_features = model_dict["lgbm_feature_names"]
    ridge_model = model_dict["ridge_model"]
    ridge_scaler = model_dict["ridge_scaler"]
    ridge_features = model_dict["ridge_feature_names"]
    lgbm_weight = model_dict["lgbm_weight"]

    # The model expects AR features (resid_lag_*) from Pass 2 training.
    # For new predictions without known residuals, fill with 0 (= no error info).
    ar_cols = ["resid_lag_24h", "resid_lag_48h", "resid_lag_168h",
               "resid_rolling_mean_24h", "resid_rolling_mean_168h"]
    for col in ar_cols:
        if col not in X_new.columns:
            X_new[col] = 0.0
            logger.info("  Added missing AR feature: %s = 0", col)

    # Also add any other missing features expected by the model
    for feat_list, name in [(lgbm_features, "LGBM"), (ridge_features, "Ridge")]:
        missing = [f for f in feat_list if f not in X_new.columns]
        if missing:
            logger.warning("  %s expects %d missing features: %s — filling with 0",
                           name, len(missing), missing[:5])
            for f in missing:
                X_new[f] = 0.0

    lgbm_preds = lgbm_model.predict(X_new[lgbm_features])
    X_ridge_scaled = ridge_scaler.transform(X_new[ridge_features])
    ridge_preds = ridge_model.predict(X_ridge_scaled)
    new_preds = lgbm_weight * lgbm_preds + (1 - lgbm_weight) * ridge_preds
    logger.info("  LGBM weight: %.2f, Ridge weight: %.2f", lgbm_weight, 1 - lgbm_weight)

    # Build new predictions DataFrame
    new_oos = pd.DataFrame(index=new_rows.index)
    new_oos["y_actual"] = new_rows[target_col] if target_col in new_rows.columns else np.nan
    new_oos["y_pred"] = new_preds

    # Fill optional columns that exist in the original OOS
    for col in existing_oos.columns:
        if col not in new_oos.columns:
            if col == "y_pred_naive":
                # Naive = same hour last week
                lag_col = "price_lag_168h"
                if lag_col in X_new.columns:
                    new_oos[col] = X_new[lag_col].values
                else:
                    new_oos[col] = np.nan
            else:
                new_oos[col] = np.nan

    # Concatenate
    extended_oos = pd.concat([existing_oos, new_oos])
    extended_oos = extended_oos[~extended_oos.index.duplicated(keep="first")]
    extended_oos = extended_oos.sort_index()

    logger.info("Extended OOS predictions: %d rows (%s → %s)",
                len(extended_oos), extended_oos.index.min(), extended_oos.index.max())

    extended_oos.to_parquet(OOS_PATH)
    logger.info("Saved extended OOS predictions")

    # ── 9. Re-run curve translation ──────────────────────────────────
    _run_curve_translation()

    logger.info(sep)
    logger.info("DONE — pipeline extended to %s", today_str)
    logger.info("You can now generate reports for today's date.")
    logger.info(sep)


def _run_curve_translation():
    """Re-run curve translation on the extended data."""
    logger.info("=" * 60)
    logger.info("Re-running curve translation on extended data")

    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_curve_translation.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    if result.returncode == 0:
        logger.info("Curve translation completed successfully")
    else:
        logger.error("Curve translation failed:\n%s", result.stderr[-500:] if result.stderr else "no stderr")


if __name__ == "__main__":
    main()
