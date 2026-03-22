"""
Prompt curve translation for DE_LU day-ahead electricity price forecasting.

Translates hourly DA price forecasts into tradable signals for European power
forward contracts (EEX German Power Base/Peak futures). Constructs fair-value
estimates for Week+1, Month+1, and Quarter+1 delivery periods, computes signal
premiums relative to a rolling DA curve proxy, applies confidence weighting and
invalidation logic, and produces a desk-ready signal table.

Inputs:
    data/processed/oos_predictions.parquet  (from Part 2)
    data/processed/de_power_dataset.parquet (from Part 1)

Outputs:
    outputs/curve_translation/signal_table.csv
    outputs/curve_translation/delivery_periods.csv
    outputs/curve_translation/fig_ct_01_signal_dashboard.png
    outputs/curve_translation/fig_ct_02_shape_premium.png
    outputs/curve_translation/fig_ct_03_signal_backtest.png
    outputs/curve_translation/fig_ct_04_invalidation_monitor.png
    outputs/curve_translation/fig_ct_05_confidence_bands.png
    outputs/curve_translation/curve_translation_report.txt
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Project paths (imported from ingestion to stay consistent) ────────────
from src.ingestion import PROJECT_ROOT, DATA_PROCESSED, OUTPUTS_DIR, LOGS_DIR
from src.models import OOS_TEST_START, OOS_TEST_END, PEAK_HOURS

# ── Output directory ──────────────────────────────────────────────────────
CT_OUTPUT_DIR = OUTPUTS_DIR / "curve_translation"

# ── Delivery period definitions (UTC hours) ───────────────────────────────
CT_PEAK_HOURS    = list(range(8, 20))           # 08:00–19:00 UTC (EEX peakload)
CT_OFFPEAK_HOURS = [h for h in range(24) if h not in CT_PEAK_HOURS]
CT_BASE_HOURS    = list(range(0, 24))

# ── Signal thresholds ────────────────────────────────────────────────────
SIGNAL_STRONG_BUY_THRESHOLD  =  8.0   # €/MWh above rolling mean → strong long
SIGNAL_BUY_THRESHOLD         =  3.0   # €/MWh above rolling mean → long
SIGNAL_SELL_THRESHOLD        = -3.0   # €/MWh below rolling mean → short
SIGNAL_STRONG_SELL_THRESHOLD = -8.0   # €/MWh below rolling mean → strong short
SIGNAL_NEUTRAL_BAND          = (-3.0, 3.0)  # within this range = hold/flat

# ── Invalidation trigger thresholds ──────────────────────────────────────
INVALIDATION_GAS_SPIKE_EUR_MWH        = 5.0    # €/MWh single-day gas move
INVALIDATION_GAS_SPIKE_PCT            = 0.10   # 10 % single-day gas move
INVALIDATION_NEGATIVE_PRICE_THRESHOLD = -20.0  # €/MWh — deep negative prices

# Wind and residual-load thresholds.
# NOTE: The spec's 3 GW / 5 GW values assume we can compare two different
# forecast *vintages* for the SAME delivery date (e.g. yesterday's D-2
# forecast vs today's D-1 forecast for tomorrow).  Our dataset has only the
# published D-1 forecast, so we measure the day-to-day change in realised
# weather conditions across DIFFERENT delivery days.  That quantity is
# naturally much larger (median ≈ 5 GW wind, 6 GW residual load), so the
# original thresholds fire on 60–70 % of days — ordinary weather, not
# genuine forecast surprises.
#
# Fix 1 — raised absolute thresholds to the ~90th percentile of the
# day-to-day distribution, so only truly extreme weather shifts trigger.
# Fix 2 — adaptive z-score: flag only when the day-on-day change exceeds
# 2 standard deviations of its own trailing 30-day distribution.  This
# self-calibrates across seasons (volatile winter vs calm summer).
# A day is flagged only if BOTH the absolute AND z-score tests fire.
INVALIDATION_WIND_REVISION_MW         = 15000  # MW — raised from 3 GW (see note)
INVALIDATION_RESIDUAL_LOAD_SWING_MW   = 15000  # MW — raised from 5 GW (see note)
INVALIDATION_ZSCORE_THRESHOLD         = 2.0    # flag if |z| > 2 σ of trailing 30d
INVALIDATION_ZSCORE_WINDOW            = 30     # trailing window for adaptive σ

# ── Rolling window sizes for fair-value proxy ────────────────────────────
CURVE_PROXY_WINDOW_7D  = 7 * 24    # 168 hours — prompt week proxy
CURVE_PROXY_WINDOW_30D = 30 * 24   # 720 hours — prompt month proxy
CURVE_PROXY_WINDOW_90D = 90 * 24   # 2160 hours — prompt quarter proxy

# ── Confidence signal sizing ─────────────────────────────────────────────
CONFIDENCE_HIGH_THRESHOLD   = 0.75   # model confidence ≥ 75 % → full size
CONFIDENCE_MEDIUM_THRESHOLD = 0.50   # model confidence 50–75 % → half size
# Below 50 % → no signal regardless of direction

# ── CV mean MAE from Part 2 performance report ───────────────────────────
# Source: outputs/model_performance_report.txt — best model hourly MAE.
# This is used for the high-error-regime invalidation flag.  If the report
# cannot be parsed at runtime, this fallback value is used.
CV_MEAN_MAE_FALLBACK = 13.6  # EUR/MWh — pass2_hourly_ew MAE from Part 2

# ── Gas-to-power conversion ──────────────────────────────────────────────
CCGT_EFFICIENCY = 0.47        # typical combined-cycle gas turbine
GAS_TO_POWER_FACTOR = 0.45   # simplified: 1/efficiency ≈ 2.13, but in
                              # €/MWh-gas → €/MWh-elec the factor is ~0.45

# ── Plot style ───────────────────────────────────────────────────────────
COLOR_CYCLE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
PLOT_DPI = 150

# ── Logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger("curve_translation")


# =====================================================================
# Logging setup
# =====================================================================
def _setup_logging() -> None:
    """Configure logging to both console and logs/curve_translation.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(
        LOGS_DIR / "curve_translation.log", mode="a", encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _read_cv_mean_mae() -> float:
    """Try to read the CV mean MAE from the Part 2 performance report.

    Falls back to CV_MEAN_MAE_FALLBACK if the report cannot be parsed.
    """
    report_path = OUTPUTS_DIR / "model_performance_report.txt"
    if not report_path.exists():
        logger.warning(
            "Performance report not found at %s — using fallback CV_MEAN_MAE=%.1f",
            report_path, CV_MEAN_MAE_FALLBACK,
        )
        return CV_MEAN_MAE_FALLBACK

    try:
        text = report_path.read_text(encoding="utf-8")
        # Look for "Hourly  MAE: XX.X" in the BEST MODEL section
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Hourly") and "MAE:" in stripped:
                # e.g.  "Hourly  MAE: 13.6 EUR/MWh"
                parts = stripped.split("MAE:")
                if len(parts) >= 2:
                    val_str = parts[1].strip().split()[0]
                    val = float(val_str)
                    logger.info("Parsed CV_MEAN_MAE = %.2f from performance report", val)
                    return val
    except Exception as exc:
        logger.warning("Could not parse performance report: %s", exc)

    logger.warning("Using fallback CV_MEAN_MAE = %.1f", CV_MEAN_MAE_FALLBACK)
    return CV_MEAN_MAE_FALLBACK


# =====================================================================
# Section 1: Delivery Period Aggregation
# =====================================================================
def compute_delivery_periods(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly forecasts into daily delivery-period metrics.

    Args:
        hourly_df: DataFrame with UTC DatetimeIndex and at minimum columns
            ``y_pred`` and ``y_actual``.  Optionally includes
            ``wind_forecast_mw``, ``solar_forecast_mw``,
            ``load_forecast_mw``, ``gas_price_eur_mwh``.

    Returns:
        DataFrame with one row per calendar date (UTC) containing base/peak/
        offpeak aggregations, percentile bands, and fundamental context.
    """
    logger.info("Computing delivery-period aggregations …")

    df = hourly_df.copy()
    df["delivery_date"] = df.index.date
    df["hour"] = df.index.hour

    records = []
    for date, grp in df.groupby("delivery_date"):
        rec: dict = {"delivery_date": pd.Timestamp(date)}

        # ── Price aggregations ────────────────────────────────────────
        peak_mask = grp["hour"].isin(CT_PEAK_HOURS)
        offpeak_mask = grp["hour"].isin(CT_OFFPEAK_HOURS)

        rec["base_pred"] = grp["y_pred"].mean()
        rec["peak_pred"] = grp.loc[peak_mask, "y_pred"].mean() if peak_mask.any() else np.nan
        rec["offpeak_pred"] = grp.loc[offpeak_mask, "y_pred"].mean() if offpeak_mask.any() else np.nan

        rec["base_actual"] = grp["y_actual"].mean() if grp["y_actual"].notna().any() else np.nan
        rec["peak_actual"] = (
            grp.loc[peak_mask, "y_actual"].mean()
            if peak_mask.any() and grp.loc[peak_mask, "y_actual"].notna().any()
            else np.nan
        )
        rec["offpeak_actual"] = (
            grp.loc[offpeak_mask, "y_actual"].mean()
            if offpeak_mask.any() and grp.loc[offpeak_mask, "y_actual"].notna().any()
            else np.nan
        )

        # Percentile bands
        rec["base_p10"] = np.nanpercentile(grp["y_pred"].values, 10)
        rec["base_p25"] = np.nanpercentile(grp["y_pred"].values, 25)
        rec["base_p75"] = np.nanpercentile(grp["y_pred"].values, 75)
        rec["base_p90"] = np.nanpercentile(grp["y_pred"].values, 90)

        # Intraday shape metrics
        rec["intraday_spread_pred"] = grp["y_pred"].max() - grp["y_pred"].min()
        rec["peak_base_premium_pred"] = rec["peak_pred"] - rec["base_pred"] if pd.notna(rec["peak_pred"]) else np.nan
        rec["peak_base_premium_actual"] = (
            rec["peak_actual"] - rec["base_actual"]
            if pd.notna(rec["peak_actual"]) and pd.notna(rec["base_actual"])
            else np.nan
        )

        # ── Fundamental aggregations ──────────────────────────────────
        has_wind = "wind_forecast_mw" in grp.columns
        has_solar = "solar_forecast_mw" in grp.columns
        has_load = "load_forecast_mw" in grp.columns
        has_gas = "gas_price_eur_mwh" in grp.columns

        if has_load and has_wind and has_solar:
            res_load = grp["load_forecast_mw"] - grp["wind_forecast_mw"] - grp["solar_forecast_mw"]
            rec["avg_residual_load_mw"] = res_load.mean()
            rec["avg_res_penetration"] = (
                (grp["wind_forecast_mw"] + grp["solar_forecast_mw"])
                / grp["load_forecast_mw"].replace(0, np.nan)
            ).mean()
        else:
            rec["avg_residual_load_mw"] = np.nan
            rec["avg_res_penetration"] = np.nan

        rec["avg_wind_mw"] = grp["wind_forecast_mw"].mean() if has_wind else np.nan
        rec["avg_solar_mw"] = grp["solar_forecast_mw"].mean() if has_solar else np.nan
        rec["gas_price"] = grp["gas_price_eur_mwh"].iloc[0] if has_gas else np.nan

        # Shape insight
        rec["min_hour_pred"] = int(grp.loc[grp["y_pred"].idxmin(), "hour"]) if len(grp) > 0 else np.nan
        rec["max_hour_pred"] = int(grp.loc[grp["y_pred"].idxmax(), "hour"]) if len(grp) > 0 else np.nan
        rec["n_negative_hours_pred"] = int((grp["y_pred"] < 0).sum())

        records.append(rec)

    delivery_df = pd.DataFrame(records)
    delivery_df = delivery_df.sort_values("delivery_date").reset_index(drop=True)

    logger.info(
        "Delivery periods: %d days (%s to %s)",
        len(delivery_df),
        delivery_df["delivery_date"].iloc[0].strftime("%Y-%m-%d"),
        delivery_df["delivery_date"].iloc[-1].strftime("%Y-%m-%d"),
    )

    # Save
    CT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    delivery_df.to_csv(CT_OUTPUT_DIR / "delivery_periods.csv", index=False)
    logger.info("Saved delivery_periods.csv")

    return delivery_df


# =====================================================================
# Section 2: Rolling Fair-Value Proxy
# =====================================================================
def compute_fair_value_signal(
    delivery_df: pd.DataFrame,
    raw_hourly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rolling curve proxies, forward fair values, and signal premiums.

    Args:
        delivery_df: Daily delivery-period DataFrame from
            :func:`compute_delivery_periods`.
        raw_hourly_df: The raw hourly dataset (for computing rolling DA
            averages from actual prices where available).

    Returns:
        ``delivery_df`` augmented with curve proxy, fair value, signal
        premium, and distribution band columns.
    """
    logger.info("Computing fair-value signals …")

    sig = delivery_df.copy()
    dates = sig["delivery_date"]

    # ── Step 1: trailing rolling DA averages (curve proxy) ────────────
    # Use base_actual where available, fall back to base_pred for future
    base_series = sig["base_actual"].copy()
    base_series = base_series.fillna(sig["base_pred"])

    sig["curve_proxy_7d"] = base_series.rolling(
        window=7, min_periods=3,
    ).mean()
    sig["curve_proxy_30d"] = base_series.rolling(
        window=30, min_periods=10,
    ).mean()
    sig["curve_proxy_90d"] = base_series.rolling(
        window=90, min_periods=30,
    ).mean()

    # ── Step 2: forward-looking fair value aggregations ───────────────
    n = len(sig)
    fv_week_base = np.full(n, np.nan)
    fv_month_base = np.full(n, np.nan)
    fv_quarter_base = np.full(n, np.nan)
    fv_week_peak = np.full(n, np.nan)
    fv_month_peak = np.full(n, np.nan)

    fv_week_base_p10 = np.full(n, np.nan)
    fv_week_base_p90 = np.full(n, np.nan)
    fv_month_base_p10 = np.full(n, np.nan)
    fv_month_base_p90 = np.full(n, np.nan)

    dates_arr = dates.values  # numpy datetime64 array

    for i in range(n):
        d = dates_arr[i]
        d_plus_7 = d + np.timedelta64(7, "D")
        d_plus_30 = d + np.timedelta64(30, "D")
        d_plus_90 = d + np.timedelta64(90, "D")

        # Use >= d (inclusive of current day) so that even the very last
        # date in the OOS window can produce a fair-value estimate from
        # its own predictions, rather than returning NaN / NO_DATA.
        mask_7 = (dates_arr >= d) & (dates_arr <= d_plus_7)
        mask_30 = (dates_arr >= d) & (dates_arr <= d_plus_30)
        mask_90 = (dates_arr >= d) & (dates_arr <= d_plus_90)

        if mask_7.any():
            subset = sig.loc[mask_7]
            fv_week_base[i] = subset["base_pred"].mean()
            fv_week_peak[i] = subset["peak_pred"].mean()
            fv_week_base_p10[i] = subset["base_p10"].mean()
            fv_week_base_p90[i] = subset["base_p90"].mean()

        if mask_30.any():
            subset = sig.loc[mask_30]
            fv_month_base[i] = subset["base_pred"].mean()
            fv_month_peak[i] = subset["peak_pred"].mean()
            fv_month_base_p10[i] = subset["base_p10"].mean()
            fv_month_base_p90[i] = subset["base_p90"].mean()

        if mask_90.any():
            fv_quarter_base[i] = sig.loc[mask_90, "base_pred"].mean()

    sig["fv_week_base"] = fv_week_base
    sig["fv_month_base"] = fv_month_base
    sig["fv_quarter_base"] = fv_quarter_base
    sig["fv_week_peak"] = fv_week_peak
    sig["fv_month_peak"] = fv_month_peak

    sig["fv_week_base_p10"] = fv_week_base_p10
    sig["fv_week_base_p90"] = fv_week_base_p90
    sig["fv_month_base_p10"] = fv_month_base_p10
    sig["fv_month_base_p90"] = fv_month_base_p90

    # ── Step 3: signal premium = fair value − curve proxy ─────────────
    sig["signal_week_base"] = sig["fv_week_base"] - sig["curve_proxy_7d"]
    sig["signal_month_base"] = sig["fv_month_base"] - sig["curve_proxy_30d"]
    sig["signal_quarter_base"] = sig["fv_quarter_base"] - sig["curve_proxy_90d"]
    sig["signal_week_peak"] = sig["fv_week_peak"] - sig["curve_proxy_7d"]
    sig["signal_month_peak"] = sig["fv_month_peak"] - sig["curve_proxy_30d"]

    # ── Step 4: directional signal labels ─────────────────────────────
    def _label_signal(premium: float) -> tuple:
        """Return (label, integer) for a signal premium value."""
        if pd.isna(premium):
            return ("NO_DATA", 0)
        if premium >= SIGNAL_STRONG_BUY_THRESHOLD:
            return ("STRONG_BUY", 2)
        if premium >= SIGNAL_BUY_THRESHOLD:
            return ("BUY", 1)
        if premium <= SIGNAL_STRONG_SELL_THRESHOLD:
            return ("STRONG_SELL", -2)
        if premium <= SIGNAL_SELL_THRESHOLD:
            return ("SELL", -1)
        return ("HOLD", 0)

    for col_premium, col_label, col_int in [
        ("signal_week_base", "signal_label_week_base", "signal_int_week_base"),
        ("signal_month_base", "signal_label_month_base", "signal_int_month_base"),
        ("signal_quarter_base", "signal_label_quarter_base", "signal_int_quarter_base"),
        ("signal_week_peak", "signal_label_week_peak", "signal_int_week_peak"),
        ("signal_month_peak", "signal_label_month_peak", "signal_int_month_peak"),
    ]:
        labels_ints = sig[col_premium].apply(_label_signal)
        sig[col_label] = labels_ints.apply(lambda x: x[0])
        sig[col_int] = labels_ints.apply(lambda x: x[1])

    # Log summary
    label_counts = sig["signal_label_month_base"].value_counts()
    logger.info("Month-base signal distribution:\n%s", label_counts.to_string())

    return sig


# =====================================================================
# Section 3: Confidence-Weighted Signal
# =====================================================================
def compute_confidence_score(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Add confidence scores and position-size multipliers to the signal table.

    Confidence is inversely proportional to the width of the forecast
    distribution band (p90 − p10).  Wider bands → less confidence → smaller
    position.

    Args:
        signal_df: DataFrame from :func:`compute_fair_value_signal`.

    Returns:
        DataFrame with ``week_confidence``, ``month_confidence``,
        ``week_position_size``, ``month_position_size``,
        ``week_signal_weighted``, ``month_signal_weighted`` columns added.
    """
    logger.info("Computing confidence scores …")
    sig = signal_df.copy()

    # Band widths
    sig["week_band_width"] = sig["fv_week_base_p90"] - sig["fv_week_base_p10"]
    sig["month_band_width"] = sig["fv_month_base_p90"] - sig["fv_month_base_p10"]

    # Historical median band width (over entire OOS period)
    median_week_bw = sig["week_band_width"].median()
    median_month_bw = sig["month_band_width"].median()

    # Guard against zero medians
    median_week_bw = max(median_week_bw, 1.0) if pd.notna(median_week_bw) else 1.0
    median_month_bw = max(median_month_bw, 1.0) if pd.notna(median_month_bw) else 1.0

    logger.info(
        "Median band widths: week=%.2f, month=%.2f €/MWh",
        median_week_bw, median_month_bw,
    )

    # Confidence: 1 = perfect, 0 = twice the normal uncertainty
    sig["week_confidence"] = (
        1 - (sig["week_band_width"] / median_week_bw).clip(0, 2) / 2
    )
    sig["month_confidence"] = (
        1 - (sig["month_band_width"] / median_month_bw).clip(0, 2) / 2
    )

    # Position sizing
    def _position_size(conf: float) -> float:
        if pd.isna(conf):
            return 0.0
        if conf >= CONFIDENCE_HIGH_THRESHOLD:
            return 1.0
        if conf >= CONFIDENCE_MEDIUM_THRESHOLD:
            return 0.5
        return 0.0

    sig["week_position_size"] = sig["week_confidence"].apply(_position_size)
    sig["month_position_size"] = sig["month_confidence"].apply(_position_size)

    # Weighted signal = signal integer × position size
    sig["week_signal_weighted"] = sig["signal_int_week_base"] * sig["week_position_size"]
    sig["month_signal_weighted"] = sig["signal_int_month_base"] * sig["month_position_size"]

    logger.info(
        "Confidence stats — week: mean=%.2f, month: mean=%.2f",
        sig["week_confidence"].mean(),
        sig["month_confidence"].mean(),
    )

    return sig


# =====================================================================
# Section 4: Invalidation Logic
# =====================================================================
def compute_invalidation_flags(
    signal_df: pd.DataFrame,
    raw_hourly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute invalidation flags that make the signal desk-ready.

    A signal accompanied by any True invalidation flag should not be traded —
    the desk should stand aside until the condition clears.

    Args:
        signal_df: DataFrame from :func:`compute_confidence_score`.
        raw_hourly_df: Raw hourly dataset with fundamentals.

    Returns:
        DataFrame with five boolean invalidation columns plus
        ``any_invalidation`` added.
    """
    logger.info("Computing invalidation flags …")
    sig = signal_df.copy()
    n = len(sig)

    cv_mean_mae = _read_cv_mean_mae()
    logger.info("Using CV_MEAN_MAE = %.2f for high-error-regime threshold", cv_mean_mae)

    # ── Flag 1: Gas spike ─────────────────────────────────────────────
    gas_change = sig["gas_price"].diff(1)
    gas_change_pct = sig["gas_price"].pct_change(1)
    sig["inv_gas_spike"] = (
        (gas_change.abs() > INVALIDATION_GAS_SPIKE_EUR_MWH)
        | (gas_change_pct.abs() > INVALIDATION_GAS_SPIKE_PCT)
    ).fillna(False).astype(bool)

    # ── Flag 2: Wind forecast revision ────────────────────────────────
    # We measure day-to-day change in avg wind across different delivery
    # days (not a same-day forecast revision — see constant docstring).
    # Two conditions must BOTH be true to trigger:
    #   (a) absolute change > INVALIDATION_WIND_REVISION_MW  (fixed cap)
    #   (b) absolute change > 2σ of its trailing 30-day distribution
    # This catches only genuinely anomalous weather shifts.
    if "avg_wind_mw" in sig.columns:
        wind_change = sig["avg_wind_mw"].diff(1).abs()
        wind_rolling_mean = wind_change.rolling(
            window=INVALIDATION_ZSCORE_WINDOW, min_periods=7,
        ).mean()
        wind_rolling_std = wind_change.rolling(
            window=INVALIDATION_ZSCORE_WINDOW, min_periods=7,
        ).std()
        # Guard against zero std
        wind_rolling_std = wind_rolling_std.replace(0, np.nan)
        wind_zscore = (wind_change - wind_rolling_mean) / wind_rolling_std

        abs_flag = wind_change > INVALIDATION_WIND_REVISION_MW
        zscore_flag = wind_zscore.abs() > INVALIDATION_ZSCORE_THRESHOLD
        sig["inv_wind_revision"] = (
            abs_flag & zscore_flag
        ).fillna(False).astype(bool)
    else:
        sig["inv_wind_revision"] = False

    # ── Flag 3: Residual load swing ───────────────────────────────────
    # Same dual-gate approach: absolute threshold + adaptive z-score.
    if "avg_residual_load_mw" in sig.columns:
        res_change = sig["avg_residual_load_mw"].diff(1).abs()
        res_rolling_mean = res_change.rolling(
            window=INVALIDATION_ZSCORE_WINDOW, min_periods=7,
        ).mean()
        res_rolling_std = res_change.rolling(
            window=INVALIDATION_ZSCORE_WINDOW, min_periods=7,
        ).std()
        res_rolling_std = res_rolling_std.replace(0, np.nan)
        res_zscore = (res_change - res_rolling_mean) / res_rolling_std

        abs_flag = res_change > INVALIDATION_RESIDUAL_LOAD_SWING_MW
        zscore_flag = res_zscore.abs() > INVALIDATION_ZSCORE_THRESHOLD
        sig["inv_residual_load_swing"] = (
            abs_flag & zscore_flag
        ).fillna(False).astype(bool)
    else:
        sig["inv_residual_load_swing"] = False

    # ── Flag 4: Extreme negative-price regime ─────────────────────────
    sig["inv_negative_price_regime"] = (
        sig["n_negative_hours_pred"] >= 4
    ).astype(bool)

    # ── Flag 5: High model error regime ───────────────────────────────
    # Rolling 7-day MAE where actuals are available
    daily_abs_error = (sig["base_actual"] - sig["base_pred"]).abs()
    rolling_7d_mae = daily_abs_error.rolling(window=7, min_periods=3).mean()
    # Forward-fill for dates where actuals are not yet available
    rolling_7d_mae = rolling_7d_mae.ffill()
    sig["rolling_7d_mae"] = rolling_7d_mae
    sig["inv_high_error_regime"] = (
        rolling_7d_mae > (2 * cv_mean_mae)
    ).fillna(False).astype(bool)

    # ── Combined invalidation ─────────────────────────────────────────
    sig["any_invalidation"] = (
        sig["inv_gas_spike"]
        | sig["inv_wind_revision"]
        | sig["inv_residual_load_swing"]
        | sig["inv_negative_price_regime"]
        | sig["inv_high_error_regime"]
    ).astype(bool)

    # Override position sizes and labels when invalidated
    inv_mask = sig["any_invalidation"]
    sig.loc[inv_mask, "week_position_size"] = 0.0
    sig.loc[inv_mask, "month_position_size"] = 0.0
    sig.loc[inv_mask, "week_signal_weighted"] = 0.0
    sig.loc[inv_mask, "month_signal_weighted"] = 0.0
    sig.loc[inv_mask, "signal_label_week_base"] = "INVALIDATED"
    sig.loc[inv_mask, "signal_label_month_base"] = "INVALIDATED"

    n_inv = inv_mask.sum()
    logger.info(
        "Invalidation: %d / %d days (%.1f%%)",
        n_inv, n, n_inv / n * 100 if n > 0 else 0,
    )
    for flag_col in [
        "inv_gas_spike", "inv_wind_revision", "inv_residual_load_swing",
        "inv_negative_price_regime", "inv_high_error_regime",
    ]:
        logger.info("  %s: %d days", flag_col, sig[flag_col].sum())

    return sig


# =====================================================================
# Section 5: Spark Spread Proxy
# =====================================================================
def compute_spark_spread_proxy(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simplified clean spark spread proxy (without carbon).

    The clean spark spread is the generation margin for a gas-fired plant:
    ``power_price - (gas_price × heat_rate) - (carbon_price × emission_factor)``.
    Since we lack carbon price data, we use a simplified version:
    ``spark_spread ≈ base_price - (gas_price × GAS_TO_POWER_FACTOR)``.

    Trading interpretation:
        - spark_spread_premium > 0: gas plants earning above-average margin →
          gas is price-setting → thermal regime → DA model signal is reliable
        - spark_spread_premium < −10 €/MWh: RES suppressing prices below gas
          cost → renewable oversupply regime → model signal less reliable,
          reduce position sizing

    Args:
        signal_df: DataFrame from :func:`compute_invalidation_flags`.

    Returns:
        DataFrame with spark spread columns added.
    """
    logger.info("Computing spark spread proxy …")
    sig = signal_df.copy()

    sig["spark_spread_pred"] = sig["base_pred"] - (sig["gas_price"] * GAS_TO_POWER_FACTOR)
    sig["spark_spread_actual"] = sig["base_actual"] - (sig["gas_price"] * GAS_TO_POWER_FACTOR)

    # Rolling 30-day average spark spread — the "normal" level
    sig["spark_spread_rolling_mean"] = sig["spark_spread_actual"].rolling(
        window=30, min_periods=10,
    ).mean()
    sig["spark_spread_premium"] = sig["spark_spread_actual"] - sig["spark_spread_rolling_mean"]

    # Regime classification
    sig["is_thermal_regime"] = (sig["spark_spread_actual"] > 5).astype(bool)
    sig["is_res_regime"] = (sig["spark_spread_actual"] < -5).astype(bool)

    n_thermal = sig["is_thermal_regime"].sum()
    n_res = sig["is_res_regime"].sum()
    logger.info(
        "Spark spread regimes: thermal=%d days, RES=%d days, mixed=%d days",
        n_thermal, n_res, len(sig) - n_thermal - n_res,
    )

    return sig


# =====================================================================
# Section 6: Master Signal Table
# =====================================================================
def build_signal_table(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final desk-ready signal table.

    Selects and orders the columns that a trader needs to see each morning.

    Args:
        signal_df: Fully computed DataFrame from the pipeline.

    Returns:
        Clean signal table saved to ``signal_table.csv``.
    """
    logger.info("Building master signal table …")

    output_cols = [
        # Identity
        "delivery_date",
        # Curve proxy
        "curve_proxy_7d", "curve_proxy_30d", "curve_proxy_90d",
        # Fair value estimates
        "fv_week_base", "fv_month_base", "fv_quarter_base",
        "fv_week_peak", "fv_month_peak",
        # Signal premiums (€/MWh)
        "signal_week_base", "signal_month_base", "signal_quarter_base",
        "signal_week_peak", "signal_month_peak",
        # Signal labels
        "signal_label_week_base", "signal_label_month_base",
        # Confidence & position sizing
        "week_confidence", "month_confidence",
        "week_position_size", "month_position_size",
        "week_signal_weighted", "month_signal_weighted",
        # Distribution bands
        "fv_week_base_p10", "fv_week_base_p90",
        "fv_month_base_p10", "fv_month_base_p90",
        # Invalidation flags
        "inv_gas_spike", "inv_wind_revision", "inv_residual_load_swing",
        "inv_negative_price_regime", "inv_high_error_regime",
        "any_invalidation",
        # Fundamentals context
        "avg_residual_load_mw", "avg_res_penetration", "gas_price",
        "spark_spread_actual", "spark_spread_premium",
        "is_thermal_regime", "is_res_regime",
        "peak_base_premium_pred", "n_negative_hours_pred",
        # Actuals (for backtest)
        "base_actual", "base_pred",
    ]

    # Only include columns that actually exist
    existing = [c for c in output_cols if c in signal_df.columns]
    table = signal_df[existing].copy()

    CT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CT_OUTPUT_DIR / "signal_table.csv", index=False)
    logger.info("Saved signal_table.csv (%d rows × %d cols)", len(table), len(table.columns))

    return table


# =====================================================================
# Section 7: Signal Backtest
# =====================================================================
def compute_signal_backtest(signal_df: pd.DataFrame) -> dict:
    """Run a simple directional backtest of the month-base signal.

    **IMPORTANT CAVEAT:** This backtest assumes perfect execution at the curve
    proxy price, no transaction costs, no position limits, and no slippage.
    It is for illustration only and does not constitute a live trading
    strategy.  The Sharpe figure is not annualised in a statistically rigorous
    sense given the short OOS window.

    Args:
        signal_df: Fully computed signal table.

    Returns:
        Dictionary with backtest summary statistics and a DataFrame of
        daily P&L.
    """
    logger.info("Computing signal backtest …")

    # Filter to dates with actuals and valid (non-invalidated) signals
    bt = signal_df.dropna(subset=["base_actual", "curve_proxy_30d", "signal_month_base"]).copy()

    # Signal direction and position
    bt["signal_direction"] = np.sign(bt["signal_month_base"])
    bt["position"] = bt["signal_direction"] * bt["month_position_size"]

    # Realisation: actual base vs curve proxy
    bt["realisation"] = bt["base_actual"] - bt["curve_proxy_30d"]

    # Daily P&L = position × realisation
    bt["daily_pnl"] = bt["position"] * bt["realisation"]

    # Separate invalidated days (shown but zero P&L)
    bt.loc[bt["any_invalidation"], "daily_pnl"] = 0.0

    # Cumulative P&L
    bt["cumulative_pnl"] = bt["daily_pnl"].cumsum()

    # Summary statistics (on non-invalidated, non-zero-position days)
    active = bt[(~bt["any_invalidation"]) & (bt["position"] != 0)]

    if len(active) > 0:
        hit_rate = (active["daily_pnl"] > 0).mean()
        avg_pnl = active["daily_pnl"].mean()
        total_pnl = active["daily_pnl"].sum()
        std_pnl = active["daily_pnl"].std()
        sharpe_proxy = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
    else:
        hit_rate = 0.0
        avg_pnl = 0.0
        total_pnl = 0.0
        sharpe_proxy = 0.0

    summary = {
        "hit_rate": hit_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "sharpe_proxy": sharpe_proxy,
        "n_signal_days": len(active),
        "n_total_days": len(bt),
    }

    logger.info(
        "Backtest: hit_rate=%.1f%%, avg_pnl=%.2f, total_pnl=%.1f, sharpe=%.2f",
        hit_rate * 100, avg_pnl, total_pnl, sharpe_proxy,
    )

    return {"summary": summary, "daily": bt}


# =====================================================================
# Section 8: Figures
# =====================================================================
def _apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def plot_signal_dashboard(signal_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 1: four-panel main daily trading view."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Prompt curve translation — signal dashboard", fontsize=14, fontweight="bold")

    dates = pd.to_datetime(signal_df["delivery_date"])

    # ── Panel 1 (top-left): Fair value vs actual vs curve proxy ───────
    ax1 = axes[0, 0]
    ax1.plot(dates, signal_df["base_actual"], color="grey", alpha=0.6, linewidth=0.8, label="DA actual (base)")
    ax1.plot(dates, signal_df["fv_month_base"], color=COLOR_CYCLE[0], linewidth=1.2, label="Model fair value (M+1 base)")
    ax1.plot(dates, signal_df["curve_proxy_30d"], color=COLOR_CYCLE[3], linewidth=1.0, linestyle="--", label="Curve proxy (30d rolling)")

    # Confidence band
    if "fv_month_base_p10" in signal_df.columns:
        ax1.fill_between(
            dates,
            signal_df["fv_month_base_p10"],
            signal_df["fv_month_base_p90"],
            alpha=0.15, color=COLOR_CYCLE[0], label="P10–P90 band",
        )

    # Invalidated days as vertical bands
    inv_mask = signal_df["any_invalidation"].values
    for i in range(len(dates)):
        if inv_mask[i]:
            ax1.axvspan(dates.iloc[i] - pd.Timedelta(hours=12),
                        dates.iloc[i] + pd.Timedelta(hours=12),
                        alpha=0.08, color="red", zorder=0)

    ax1.set_title("Model fair value vs DA actual and curve proxy (30d rolling average)")
    ax1.set_ylabel("Price (€/MWh)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2 (top-right): Monthly base signal premium bar chart ────
    ax2 = axes[0, 1]
    premium = signal_df["signal_month_base"].values
    colors_bar = []
    for i in range(len(premium)):
        if inv_mask[i]:
            colors_bar.append("lightgrey")
        elif pd.notna(premium[i]) and premium[i] >= 0:
            colors_bar.append(COLOR_CYCLE[0])
        else:
            colors_bar.append(COLOR_CYCLE[1])

    ax2.bar(dates, np.where(np.isnan(premium), 0, premium), color=colors_bar, width=1.0, edgecolor="none")
    ax2.axhline(SIGNAL_BUY_THRESHOLD, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(SIGNAL_SELL_THRESHOLD, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(SIGNAL_STRONG_BUY_THRESHOLD, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    ax2.axhline(SIGNAL_STRONG_SELL_THRESHOLD, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Monthly base signal premium (€/MWh)")
    ax2.set_ylabel("Premium (€/MWh)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 3 (bottom-left): Signal vs realisation scatter ──────────
    ax3 = axes[1, 0]
    valid = signal_df.dropna(subset=["signal_month_base", "base_actual", "curve_proxy_30d"])
    if len(valid) > 5:
        x_val = valid["signal_month_base"].values
        y_val = (valid["base_actual"] - valid["curve_proxy_30d"]).values

        # Colour by regime
        thermal = valid["is_thermal_regime"].values if "is_thermal_regime" in valid.columns else np.zeros(len(valid), dtype=bool)
        c_scatter = [COLOR_CYCLE[3] if t else COLOR_CYCLE[0] for t in thermal]

        ax3.scatter(x_val, y_val, c=c_scatter, alpha=0.5, s=15, edgecolors="none")

        # Trend line
        mask_finite = np.isfinite(x_val) & np.isfinite(y_val)
        if mask_finite.sum() > 5:
            slope, intercept, r_value, p_value, _ = stats.linregress(x_val[mask_finite], y_val[mask_finite])
            x_fit = np.linspace(np.nanmin(x_val), np.nanmax(x_val), 100)
            y_fit = slope * x_fit + intercept
            ax3.plot(x_fit, y_fit, color=COLOR_CYCLE[1], linewidth=1.5, linestyle="-",
                     label=f"Trend: r={r_value:.3f}")

            # 95% confidence band
            n_pts = mask_finite.sum()
            se = np.sqrt(np.sum((y_val[mask_finite] - (slope * x_val[mask_finite] + intercept))**2) / (n_pts - 2))
            x_mean = np.mean(x_val[mask_finite])
            ss_x = np.sum((x_val[mask_finite] - x_mean)**2)
            ci = stats.t.ppf(0.975, n_pts - 2) * se * np.sqrt(1/n_pts + (x_fit - x_mean)**2 / ss_x)
            ax3.fill_between(x_fit, y_fit - ci, y_fit + ci, alpha=0.15, color=COLOR_CYCLE[1])
            ax3.legend(fontsize=8)

        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.axvline(0, color="black", linewidth=0.5)

    ax3.set_title("Signal premium vs actual realisation")
    ax3.set_xlabel("Signal premium (€/MWh)")
    ax3.set_ylabel("Realisation: actual − proxy (€/MWh)")

    # ── Panel 4 (bottom-right): Confidence time series ────────────────
    ax4 = axes[1, 1]
    if "week_confidence" in signal_df.columns:
        ax4.plot(dates, signal_df["week_confidence"], color=COLOR_CYCLE[0], linewidth=0.8, label="Week confidence", alpha=0.8)
    if "month_confidence" in signal_df.columns:
        ax4.plot(dates, signal_df["month_confidence"], color=COLOR_CYCLE[2], linewidth=0.8, label="Month confidence", alpha=0.8)
    ax4.axhline(CONFIDENCE_MEDIUM_THRESHOLD, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8, label=f"Min threshold ({CONFIDENCE_MEDIUM_THRESHOLD})")
    ax4.fill_between(dates, 0, CONFIDENCE_MEDIUM_THRESHOLD, alpha=0.08, color="red")
    ax4.set_title("Model confidence score over time")
    ax4.set_ylabel("Confidence (0–1)")
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8, loc="lower right")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "fig_ct_01_signal_dashboard.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_ct_01_signal_dashboard.png")


def plot_shape_premium(signal_df: pd.DataFrame, hourly_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 2: peak/base shape analysis — three panels."""
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    dates = pd.to_datetime(signal_df["delivery_date"])

    # ── Panel 1: Peak premium time series ─────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, signal_df["peak_base_premium_pred"], color=COLOR_CYCLE[0], linewidth=0.8, label="Predicted peak premium")
    ax1.plot(dates, signal_df["peak_base_premium_actual"], color="grey", linewidth=0.8, alpha=0.7, label="Actual peak premium")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Peak premium: predicted vs actual (€/MWh)")
    ax1.set_ylabel("Peak − base (€/MWh)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Hourly price heatmap ─────────────────────────────────
    ax2 = axes[1]
    hdf = hourly_df.copy()
    hdf["date"] = hdf.index.date
    hdf["hour"] = hdf.index.hour
    pivot = hdf.pivot_table(index="hour", columns="date", values="y_pred", aggfunc="mean")

    # Subsample columns for readability (every 7th day)
    if pivot.shape[1] > 80:
        col_idx = np.linspace(0, pivot.shape[1] - 1, min(80, pivot.shape[1]), dtype=int)
        pivot_sub = pivot.iloc[:, col_idx]
    else:
        pivot_sub = pivot

    vmin = max(pivot_sub.min().min(), -50)
    vmax = min(pivot_sub.max().max(), 200)
    centre = 50.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax)

    im = ax2.imshow(
        pivot_sub.values, aspect="auto", cmap="RdYlBu_r", norm=norm,
        origin="lower", interpolation="nearest",
    )
    ax2.set_title("Forecast hourly price profile (OOS period)")
    ax2.set_ylabel("Hour of day (UTC)")
    ax2.set_xlabel("Delivery date")
    # Tick labels
    n_x = pivot_sub.shape[1]
    tick_positions = np.linspace(0, n_x - 1, min(10, n_x), dtype=int)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([str(pivot_sub.columns[i]) for i in tick_positions], rotation=30, ha="right", fontsize=7)
    ax2.set_yticks(range(0, 24, 3))
    fig.colorbar(im, ax=ax2, label="Price (€/MWh)", shrink=0.8)

    # ── Panel 3: Average intraday shape by season ─────────────────────
    ax3 = axes[2]
    hdf["month"] = hdf.index.month
    season_map = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
                  3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
                  6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
                  9: "Autumn (SON)", 10: "Autumn (SON)", 11: "Autumn (SON)"}
    hdf["season"] = hdf["month"].map(season_map)

    season_order = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]
    for idx, season in enumerate(season_order):
        subset = hdf[hdf["season"] == season]
        if len(subset) > 0:
            hourly_mean = subset.groupby("hour")["y_pred"].mean()
            ax3.plot(hourly_mean.index, hourly_mean.values,
                     color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
                     linewidth=1.5, marker="o", markersize=3, label=season)

    ax3.set_title("Average forecast intraday shape by season")
    ax3.set_xlabel("Hour of day (UTC)")
    ax3.set_ylabel("Mean predicted price (€/MWh)")
    ax3.set_xticks(range(0, 24))
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_ct_02_shape_premium.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_ct_02_shape_premium.png")


def plot_signal_backtest(backtest: dict, output_dir: Path) -> None:
    """Figure 3: signal backtest results — three panels + disclaimer."""
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    bt = backtest["daily"]
    summary = backtest["summary"]
    dates = pd.to_datetime(bt["delivery_date"])

    # ── Panel 1: Cumulative P&L ───────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, bt["cumulative_pnl"], color=COLOR_CYCLE[0], linewidth=1.2)
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_title("Hypothetical cumulative signal P&L (€/MWh, no costs)")
    ax1.set_ylabel("Cumulative P&L (€/MWh)")

    # Annotation box
    textstr = (
        f"Total P&L: {summary['total_pnl']:.1f} €/MWh\n"
        f"Hit rate: {summary['hit_rate']:.1%}\n"
        f"Sharpe proxy: {summary['sharpe_proxy']:.2f}\n"
        f"Signal days: {summary['n_signal_days']}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", bbox=props)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Daily P&L bar chart ──────────────────────────────────
    ax2 = axes[1]
    pnl_vals = bt["daily_pnl"].values
    inv_flags = bt["any_invalidation"].values
    colors_pnl = []
    for i in range(len(pnl_vals)):
        if inv_flags[i]:
            colors_pnl.append("lightgrey")
        elif pnl_vals[i] >= 0:
            colors_pnl.append(COLOR_CYCLE[2])  # green
        else:
            colors_pnl.append(COLOR_CYCLE[1])  # red

    ax2.bar(dates, pnl_vals, color=colors_pnl, width=1.0, edgecolor="none")

    # Mark invalidated days with X
    inv_dates = dates[inv_flags]
    if len(inv_dates) > 0:
        ax2.scatter(inv_dates, np.zeros(len(inv_dates)), marker="x", color="grey", s=15, alpha=0.5, zorder=5)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Daily signal P&L")
    ax2.set_ylabel("P&L (€/MWh)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 3: P&L distribution by signal direction ─────────────────
    ax3 = axes[2]
    active = bt[(~bt["any_invalidation"]) & (bt["position"] != 0)]
    buy_pnl = active.loc[active["signal_direction"] > 0, "daily_pnl"]
    sell_pnl = active.loc[active["signal_direction"] < 0, "daily_pnl"]

    if len(buy_pnl) > 3:
        ax3.hist(buy_pnl, bins=30, alpha=0.5, color=COLOR_CYCLE[0], label=f"BUY days (n={len(buy_pnl)})", density=True)
        try:
            buy_kde_x = np.linspace(buy_pnl.min() - 5, buy_pnl.max() + 5, 200)
            buy_kde = stats.gaussian_kde(buy_pnl.dropna())
            ax3.plot(buy_kde_x, buy_kde(buy_kde_x), color=COLOR_CYCLE[0], linewidth=1.5)
        except Exception:
            pass

    if len(sell_pnl) > 3:
        ax3.hist(sell_pnl, bins=30, alpha=0.5, color=COLOR_CYCLE[1], label=f"SELL days (n={len(sell_pnl)})", density=True)
        try:
            sell_kde_x = np.linspace(sell_pnl.min() - 5, sell_pnl.max() + 5, 200)
            sell_kde = stats.gaussian_kde(sell_pnl.dropna())
            ax3.plot(sell_kde_x, sell_kde(sell_kde_x), color=COLOR_CYCLE[1], linewidth=1.5)
        except Exception:
            pass

    ax3.axvline(0, color="black", linewidth=0.5)
    ax3.set_title("P&L distribution by signal direction")
    ax3.set_xlabel("Daily P&L (€/MWh)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)

    # ── Bold disclaimer ───────────────────────────────────────────────
    disclaimer = (
        "ILLUSTRATION ONLY — not a trading strategy. "
        "No transaction costs, slippage, or risk limits modelled."
    )
    fig.text(
        0.5, 0.01, disclaimer,
        ha="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#856404", alpha=0.9),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_dir / "fig_ct_03_signal_backtest.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_ct_03_signal_backtest.png")


def plot_invalidation_monitor(signal_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 4: invalidation trigger visualisation — five stacked panels."""
    _apply_style()
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    fig.suptitle("Invalidation trigger monitor", fontsize=14, fontweight="bold")

    dates = pd.to_datetime(signal_df["delivery_date"])

    # ── Panel 1: Gas price daily change ───────────────────────────────
    ax = axes[0]
    gas_change = signal_df["gas_price"].diff(1)
    ax.plot(dates, gas_change, color="grey", linewidth=0.6)
    ax.axhline(INVALIDATION_GAS_SPIKE_EUR_MWH, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8, label=f"±{INVALIDATION_GAS_SPIKE_EUR_MWH} €/MWh")
    ax.axhline(-INVALIDATION_GAS_SPIKE_EUR_MWH, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8)
    inv_gas = signal_df["inv_gas_spike"]
    ax.scatter(dates[inv_gas], gas_change[inv_gas], color=COLOR_CYCLE[1], s=20, zorder=5, label="Triggered")
    ax.set_title("Gas price daily change (€/MWh)")
    ax.set_ylabel("Δ Gas (€/MWh)")
    ax.legend(fontsize=7, loc="upper right")

    # ── Panel 2: Wind forecast revision ───────────────────────────────
    ax = axes[1]
    wind_rev = signal_df["avg_wind_mw"].diff(1).abs() if "avg_wind_mw" in signal_df.columns else pd.Series(0, index=signal_df.index)
    ax.plot(dates, wind_rev, color="grey", linewidth=0.6)
    ax.axhline(INVALIDATION_WIND_REVISION_MW, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8, label=f"{INVALIDATION_WIND_REVISION_MW} MW")
    inv_wind = signal_df["inv_wind_revision"]
    ax.scatter(dates[inv_wind], wind_rev[inv_wind], color=COLOR_CYCLE[1], s=20, zorder=5, label="Triggered")
    ax.set_title("Wind forecast revision (MW day-on-day)")
    ax.set_ylabel("|Δ Wind| (MW)")
    ax.legend(fontsize=7, loc="upper right")

    # ── Panel 3: Residual load swing ──────────────────────────────────
    ax = axes[2]
    res_swing = signal_df["avg_residual_load_mw"].diff(1).abs() if "avg_residual_load_mw" in signal_df.columns else pd.Series(0, index=signal_df.index)
    ax.plot(dates, res_swing, color="grey", linewidth=0.6)
    ax.axhline(INVALIDATION_RESIDUAL_LOAD_SWING_MW, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8, label=f"{INVALIDATION_RESIDUAL_LOAD_SWING_MW} MW")
    inv_res = signal_df["inv_residual_load_swing"]
    ax.scatter(dates[inv_res], res_swing[inv_res], color=COLOR_CYCLE[1], s=20, zorder=5, label="Triggered")
    ax.set_title("Residual load swing (MW day-on-day)")
    ax.set_ylabel("|Δ Res. load| (MW)")
    ax.legend(fontsize=7, loc="upper right")

    # ── Panel 4: Negative price hours ─────────────────────────────────
    ax = axes[3]
    neg_hours = signal_df["n_negative_hours_pred"]
    ax.bar(dates, neg_hours, color="grey", width=1.0, edgecolor="none")
    inv_neg = signal_df["inv_negative_price_regime"]
    ax.bar(dates[inv_neg], neg_hours[inv_neg], color=COLOR_CYCLE[1], width=1.0, edgecolor="none", label="≥ 4 hours")
    ax.axhline(4, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8, label="Threshold (4h)")
    ax.set_title("Predicted negative-price hours per day")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7, loc="upper right")

    # ── Panel 5: Rolling 7-day MAE ────────────────────────────────────
    ax = axes[4]
    cv_mean_mae = _read_cv_mean_mae()
    if "rolling_7d_mae" in signal_df.columns:
        ax.plot(dates, signal_df["rolling_7d_mae"], color="grey", linewidth=0.8)
        ax.axhline(2 * cv_mean_mae, color=COLOR_CYCLE[1], linestyle="--", linewidth=0.8,
                    label=f"2× CV MAE ({2 * cv_mean_mae:.1f})")
        inv_err = signal_df["inv_high_error_regime"]
        high_err_vals = signal_df.loc[inv_err, "rolling_7d_mae"]
        ax.scatter(dates[inv_err], high_err_vals, color=COLOR_CYCLE[1], s=20, zorder=5, label="Triggered")
    ax.set_title("Rolling 7-day model MAE (€/MWh)")
    ax.set_ylabel("MAE (€/MWh)")
    ax.legend(fontsize=7, loc="upper right")

    # Format x-axis
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[4].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[4].xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Summary table as text ─────────────────────────────────────────
    n_total = len(signal_df)
    n_inv = signal_df["any_invalidation"].sum()
    summary_parts = [
        f"Total signal days: {n_total}",
        f"Invalidated days: {n_inv} ({n_inv / n_total * 100:.1f}%)",
        f"Gas spike: {signal_df['inv_gas_spike'].sum()}",
        f"Wind revision: {signal_df['inv_wind_revision'].sum()}",
        f"Residual load: {signal_df['inv_residual_load_swing'].sum()}",
        f"Negative prices: {signal_df['inv_negative_price_regime'].sum()}",
        f"High error: {signal_df['inv_high_error_regime'].sum()}",
    ]
    fig.text(0.5, 0.01, "  |  ".join(summary_parts), ha="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_dir / "fig_ct_04_invalidation_monitor.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_ct_04_invalidation_monitor.png")


def plot_confidence_bands(signal_df: pd.DataFrame, output_dir: Path) -> None:
    """Figure 5: confidence band width and trading implication — two panels."""
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    dates = pd.to_datetime(signal_df["delivery_date"])

    # ── Panel 1: Fair value with confidence bands ─────────────────────
    ax1 = axes[0]
    ax1.fill_between(
        dates,
        signal_df["fv_month_base_p10"],
        signal_df["fv_month_base_p90"],
        alpha=0.2, color=COLOR_CYCLE[0], label="P10–P90 band",
    )
    ax1.plot(dates, signal_df["fv_month_base"], color=COLOR_CYCLE[0], linewidth=1.0, label="M+1 fair value (base)")

    # Actual where available
    actual_valid = signal_df["base_actual"].dropna()
    if len(actual_valid) > 0:
        actual_dates = pd.to_datetime(signal_df.loc[actual_valid.index, "delivery_date"])
        ax1.scatter(actual_dates, actual_valid, color="black", s=5, alpha=0.4, label="DA actual (base)", zorder=5)

    ax1.set_title("Monthly fair value estimate with 10th–90th percentile band")
    ax1.set_ylabel("Price (€/MWh)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Band width vs forecast error scatter ─────────────────
    ax2 = axes[1]
    valid = signal_df.dropna(subset=["fv_month_base", "base_actual", "fv_month_base_p10", "fv_month_base_p90"])
    if len(valid) > 5:
        band_width = (valid["fv_month_base_p90"] - valid["fv_month_base_p10"]).values
        abs_error = (valid["fv_month_base"] - valid["base_actual"]).abs().values

        ax2.scatter(band_width, abs_error, alpha=0.4, s=15, color=COLOR_CYCLE[0], edgecolors="none")

        # Trend line
        mask_finite = np.isfinite(band_width) & np.isfinite(abs_error)
        if mask_finite.sum() > 5:
            slope, intercept, r_value, _, _ = stats.linregress(band_width[mask_finite], abs_error[mask_finite])
            x_fit = np.linspace(np.nanmin(band_width), np.nanmax(band_width), 100)
            y_fit = slope * x_fit + intercept
            ax2.plot(x_fit, y_fit, color=COLOR_CYCLE[1], linewidth=1.5,
                     label=f"Trend: r={r_value:.3f}, slope={slope:.2f}")
            ax2.legend(fontsize=8)

    ax2.set_title("Band width vs forecast error — are confidence bands informative?")
    ax2.set_xlabel("Band width: P90 − P10 (€/MWh)")
    ax2.set_ylabel("Absolute forecast error (€/MWh)")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_ct_05_confidence_bands.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_ct_05_confidence_bands.png")


# =====================================================================
# Section 9: Curve Translation Report
# =====================================================================
def generate_curve_translation_report(
    signal_df: pd.DataFrame,
    backtest: dict,
    output_dir: Path,
) -> None:
    """Write the desk-ready curve translation report to a text file.

    Args:
        signal_df: Final signal table.
        backtest: Dictionary returned by :func:`compute_signal_backtest`.
        output_dir: Output directory (``outputs/curve_translation/``).
    """
    logger.info("Generating curve translation report …")
    summary = backtest["summary"]
    bt_daily = backtest["daily"]

    n_total = len(signal_df)
    dates = pd.to_datetime(signal_df["delivery_date"])
    date_start = dates.iloc[0].strftime("%Y-%m-%d")
    date_end = dates.iloc[-1].strftime("%Y-%m-%d")

    # Signal label distribution
    label_col = "signal_label_month_base"
    labels = signal_df[label_col].value_counts()

    def _lbl(name):
        cnt = labels.get(name, 0)
        pct = cnt / n_total * 100 if n_total > 0 else 0
        return cnt, pct

    # Signal vs realisation correlation
    valid = signal_df.dropna(subset=["signal_month_base", "base_actual", "curve_proxy_30d"])
    if len(valid) > 5:
        sig_vals = valid["signal_month_base"].values
        real_vals = (valid["base_actual"] - valid["curve_proxy_30d"]).values
        mask_f = np.isfinite(sig_vals) & np.isfinite(real_vals)
        corr = np.corrcoef(sig_vals[mask_f], real_vals[mask_f])[0, 1] if mask_f.sum() > 5 else 0.0
    else:
        corr = 0.0

    mean_signal = signal_df["signal_month_base"].mean()
    std_signal = signal_df["signal_month_base"].std()

    # Invalidation breakdown
    inv_total = signal_df["any_invalidation"].sum()

    # Spark spread regime stats
    n_thermal = signal_df["is_thermal_regime"].sum() if "is_thermal_regime" in signal_df.columns else 0
    n_res = signal_df["is_res_regime"].sum() if "is_res_regime" in signal_df.columns else 0

    # Hit rate by regime
    active_bt = bt_daily[(~bt_daily["any_invalidation"]) & (bt_daily["position"] != 0)]
    if "is_thermal_regime" in active_bt.columns and len(active_bt) > 0:
        thermal_active = active_bt[active_bt["is_thermal_regime"]]
        res_active = active_bt[active_bt["is_res_regime"]] if "is_res_regime" in active_bt.columns else pd.DataFrame()
        hit_thermal = (thermal_active["daily_pnl"] > 0).mean() if len(thermal_active) > 0 else 0.0
        hit_res = (res_active["daily_pnl"] > 0).mean() if len(res_active) > 0 else 0.0
    else:
        hit_thermal = 0.0
        hit_res = 0.0

    cv_mae = _read_cv_mean_mae()

    sb_n, sb_p = _lbl("STRONG_BUY")
    b_n, b_p = _lbl("BUY")
    h_n, h_p = _lbl("HOLD")
    s_n, s_p = _lbl("SELL")
    ss_n, ss_p = _lbl("STRONG_SELL")
    iv_n, iv_p = _lbl("INVALIDATED")

    lines = [
        "=" * 60,
        " DE_LU Power — Prompt Curve Translation Report",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f" OOS Signal Period: {date_start} to {date_end}",
        f" CV Mean MAE used: {cv_mae:.2f} EUR/MWh",
        "=" * 60,
        "",
        "METHODOLOGY",
        "-" * 60,
        "Signal construction:",
        "  - Hourly DA forecasts → daily base/peak delivery-period averages",
        "  - 7/30/90-day forward aggregations → prompt week/month/quarter fair value",
        "  - Curve proxy: trailing rolling DA average (7d/30d/90d window)",
        "  - Signal premium: fair value minus curve proxy (€/MWh)",
        f"  - Thresholds: BUY ≥ +{SIGNAL_BUY_THRESHOLD}, STRONG_BUY ≥ +{SIGNAL_STRONG_BUY_THRESHOLD}, "
        f"SELL ≤ {SIGNAL_SELL_THRESHOLD}, STRONG_SELL ≤ {SIGNAL_STRONG_SELL_THRESHOLD}",
        "",
        f"SIGNAL SUMMARY (OOS period)",
        "-" * 60,
        f"  Total dates with signal:      {n_total}",
        f"  STRONG_BUY days:              {sb_n} ({sb_p:.1f}%)",
        f"  BUY days:                     {b_n} ({b_p:.1f}%)",
        f"  HOLD days:                    {h_n} ({h_p:.1f}%)",
        f"  SELL days:                    {s_n} ({s_p:.1f}%)",
        f"  STRONG_SELL days:             {ss_n} ({ss_p:.1f}%)",
        f"  INVALIDATED days:             {iv_n} ({iv_p:.1f}%)",
        "",
        f"  Mean month base signal:       {mean_signal:.2f} €/MWh" if pd.notna(mean_signal) else "  Mean month base signal:       N/A",
        f"  Std of signal:                {std_signal:.2f} €/MWh" if pd.notna(std_signal) else "  Std of signal:                N/A",
        f"  Signal vs realisation corr:   {corr:.3f}  ← key validity metric",
        "",
        "BACKTEST SUMMARY (ILLUSTRATION ONLY)",
        "-" * 60,
        f"  Hit rate (directional):       {summary['hit_rate']:.1%}",
        f"  Mean daily P&L:               {summary['avg_pnl']:.2f} €/MWh per unit",
        f"  Cumulative P&L:               {summary['total_pnl']:.1f} €/MWh per unit",
        f"  Sharpe proxy (annualised):    {summary['sharpe_proxy']:.2f}",
        "  WARNING: No costs, no slippage, no risk limits.",
        "",
        "INVALIDATION BREAKDOWN",
        "-" * 60,
        f"  Gas spike (>{INVALIDATION_GAS_SPIKE_EUR_MWH}€/MWh or {INVALIDATION_GAS_SPIKE_PCT:.0%}):"
        f"  {signal_df['inv_gas_spike'].sum()} days ({signal_df['inv_gas_spike'].mean() * 100:.1f}%)",
        f"  Wind change (>{INVALIDATION_WIND_REVISION_MW / 1000:.0f}GW + >{INVALIDATION_ZSCORE_THRESHOLD:.0f}σ):"
        f"  {signal_df['inv_wind_revision'].sum()} days ({signal_df['inv_wind_revision'].mean() * 100:.1f}%)",
        f"  Res. load swing (>{INVALIDATION_RESIDUAL_LOAD_SWING_MW / 1000:.0f}GW + >{INVALIDATION_ZSCORE_THRESHOLD:.0f}σ):"
        f"  {signal_df['inv_residual_load_swing'].sum()} days ({signal_df['inv_residual_load_swing'].mean() * 100:.1f}%)",
        f"  Negative price regime (≥4h):"
        f" {signal_df['inv_negative_price_regime'].sum()} days ({signal_df['inv_negative_price_regime'].mean() * 100:.1f}%)",
        f"  High model error regime:"
        f"      {signal_df['inv_high_error_regime'].sum()} days ({signal_df['inv_high_error_regime'].mean() * 100:.1f}%)",
        f"  Any invalidation:"
        f"             {inv_total} days ({inv_total / n_total * 100:.1f}%)",
        "",
        "SPARK SPREAD REGIME",
        "-" * 60,
        f"  Thermal regime days (spark > 5 €/MWh):     {n_thermal} ({n_thermal / n_total * 100:.1f}%)",
        f"  RES regime days (spark < -5 €/MWh):        {n_res} ({n_res / n_total * 100:.1f}%)",
        f"  Signal hit rate in thermal regime:          {hit_thermal:.1%}",
        f"  Signal hit rate in RES regime:              {hit_res:.1%}",
        "",
        "WHAT THE DESK WOULD DO WITH THIS",
        "-" * 60,
        "  BUY signal (M+1 base):",
        "    → Go long EEX German Power Base Month+1 futures",
        "    → Size: full (confidence ≥ 0.75) or half (0.50–0.75)",
        "    → Entry: compare fair value to live EEX quote at 12:00 CET",
        "    → Stop: signal inverted OR any invalidation flag triggers",
        "",
        "  SELL signal (M+1 base):",
        "    → Go short EEX German Power Base Month+1 futures",
        "    → Same sizing logic",
        "    → Stop: same",
        "",
        "  PEAK PREMIUM signal (if fv_month_peak - fv_month_base > historical avg):",
        "    → Long Peak / Short Base spread (EEX base + peakload futures)",
        "    → Expresses shape view without outright directional exposure",
        "",
        "  QUARTER SIGNAL (if fv_quarter_base signal strong):",
        "    → Q+1 baseload futures — less reactive, longer-horizon view",
        "",
        "WHAT WOULD INVALIDATE THE SIGNAL",
        "-" * 60,
        "  1. Gas price move >10% intraday (TTF reprices entire curve)",
        f"  2. Wind change >{INVALIDATION_WIND_REVISION_MW / 1000:.0f}GW AND >{INVALIDATION_ZSCORE_THRESHOLD:.0f}σ of trailing 30d (dual-gate)",
        f"  3. Residual load swing >{INVALIDATION_RESIDUAL_LOAD_SWING_MW / 1000:.0f}GW AND >{INVALIDATION_ZSCORE_THRESHOLD:.0f}σ of trailing 30d (dual-gate)",
        "  4. Model predicting 4+ negative-price hours (RES oversupply regime)",
        f"  5. Model rolling MAE > 2× CV average ({2 * cv_mae:.1f} €/MWh threshold)",
        "  6. Major geopolitical event affecting gas supply (manual override)",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "curve_translation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    logger.info("Saved curve_translation_report.txt")


# =====================================================================
# Section 10: Master Orchestrator
# =====================================================================
def run_curve_translation_pipeline() -> dict:
    """Run the full prompt curve translation pipeline.

    Loads OOS predictions and raw dataset, computes delivery-period
    aggregations, fair-value signals, confidence scores, invalidation
    flags, spark spread proxy, assembles the signal table, runs a
    simple backtest, generates all five figures, and writes the report.

    Returns:
        Dictionary containing:
            ``signal_table``: final signal DataFrame
            ``delivery_periods``: daily delivery-period DataFrame
            ``backtest``: backtest summary and daily P&L
    """
    _setup_logging()
    sep = "=" * 60
    logger.info(sep)
    logger.info("DE_LU Prompt Curve Translation Pipeline")
    logger.info(sep)

    CT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load OOS predictions ───────────────────────────────────────
    oos_path = DATA_PROCESSED / "oos_predictions.parquet"
    if not oos_path.exists():
        raise FileNotFoundError(
            f"OOS predictions not found at {oos_path}. "
            "Run Part 2 (scripts/run_forecasting.py) first."
        )

    oos_df = pd.read_parquet(oos_path)
    logger.info("Loaded OOS predictions: %d rows (%s to %s)",
                len(oos_df), oos_df.index[0], oos_df.index[-1])

    # ── 2. Load raw dataset (for fundamentals) ────────────────────────
    raw_path = DATA_PROCESSED / "de_power_dataset.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Run Part 1 (scripts/run_ingestion.py) first."
        )

    raw_df = pd.read_parquet(raw_path)
    logger.info("Loaded raw dataset: %d rows", len(raw_df))

    # ── Merge fundamentals into OOS predictions ───────────────────────
    # Join the fundamental columns to the hourly predictions
    fundamental_cols = [
        c for c in raw_df.columns
        if c in ("wind_forecast_mw", "solar_forecast_mw", "load_forecast_mw", "gas_price_eur_mwh")
    ]
    hourly_df = oos_df.join(raw_df[fundamental_cols], how="left")
    logger.info("Merged fundamentals: %d columns available", len(hourly_df.columns))

    # ── 3. Compute delivery periods ───────────────────────────────────
    delivery_df = compute_delivery_periods(hourly_df)

    # ── 4. Compute fair-value signals ─────────────────────────────────
    signal_df = compute_fair_value_signal(delivery_df, hourly_df)

    # ── 5. Compute confidence scores ──────────────────────────────────
    signal_df = compute_confidence_score(signal_df)

    # ── 6. Compute invalidation flags ─────────────────────────────────
    signal_df = compute_invalidation_flags(signal_df, hourly_df)

    # ── 7. Compute spark spread proxy ─────────────────────────────────
    signal_df = compute_spark_spread_proxy(signal_df)

    # ── 8. Build final signal table ───────────────────────────────────
    signal_table = build_signal_table(signal_df)

    # ── 9. Compute signal backtest ────────────────────────────────────
    backtest = compute_signal_backtest(signal_df)

    # ── 10. Generate figures ──────────────────────────────────────────
    logger.info(sep)
    logger.info("Generating figures …")
    plot_signal_dashboard(signal_df, CT_OUTPUT_DIR)
    plot_shape_premium(signal_df, hourly_df, CT_OUTPUT_DIR)
    plot_signal_backtest(backtest, CT_OUTPUT_DIR)
    plot_invalidation_monitor(signal_df, CT_OUTPUT_DIR)
    plot_confidence_bands(signal_df, CT_OUTPUT_DIR)

    # ── 11. Generate report ───────────────────────────────────────────
    logger.info(sep)
    generate_curve_translation_report(signal_df, backtest, CT_OUTPUT_DIR)

    # ── Done ──────────────────────────────────────────────────────────
    logger.info(sep)
    logger.info("Curve translation pipeline complete!")
    logger.info("  Signal table: %d rows", len(signal_table))
    logger.info("  Invalidated days: %d", signal_table["any_invalidation"].sum())
    logger.info("  Outputs: %s", CT_OUTPUT_DIR)
    logger.info(sep)

    return {
        "signal_table": signal_table,
        "delivery_periods": delivery_df,
        "backtest": backtest,
    }
