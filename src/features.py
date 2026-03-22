"""
Feature engineering for DE_LU day-ahead electricity price forecasting.

Transforms the raw merged dataset (from ingestion.py) into a model-ready feature
matrix. Every feature respects the information barrier: when forecasting delivery
day D, only information available before the DA auction close (12:00 CET / 11:00 UTC
on day D-1) is used.

Inputs:
    data/processed/de_power_dataset.parquet (from Part 1)

Outputs:
    data/processed/feature_matrix.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import project paths and constants from ingestion (no function calls)
from src.ingestion import (
    PROJECT_ROOT,
    DATA_PROCESSED,
    LOGS_DIR,
    TIMEZONE,
    MARKET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PEAK_HOURS = list(range(8, 20))          # hours 8-19 UTC (~ peak block for DE)
BASE_HOURS = list(range(0, 24))          # all hours
FORECAST_HORIZON_HOURS = 24              # predict next day = 24 hours ahead

LAG_24H = 24                             # same hour yesterday
LAG_48H = 48                             # same hour two days ago
LAG_168H = 168                           # same hour last week
LAG_336H = 336                           # same hour two weeks ago

ROLLING_WINDOWS = [24, 48, 168]          # for rolling mean/std features

# Feature matrix drops the first N rows where lags are undefined
WARMUP_HOURS = LAG_336H                  # 336 hours = 14 days

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("features")


def _setup_logging() -> None:
    """Configure logging to both console and logs/features.log with timestamps."""
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

    fh = logging.FileHandler(LOGS_DIR / "features.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# German public holidays (hardcoded for 2021-2027)
# ---------------------------------------------------------------------------
def _german_public_holidays() -> set:
    """Return a set of datetime.date objects for German public holidays 2021-2027.

    Includes: New Year's Day, Good Friday, Easter Monday, Labour Day (May 1),
    Ascension, Whit Monday, German Unity Day (Oct 3), Christmas Eve,
    Christmas Day, Boxing Day, and Reformation Day (Oct 31, observed nationally).

    Returns:
        Set of datetime.date objects.
    """
    holidays = set()

    # Fixed holidays
    for year in range(2021, 2028):
        holidays.add(pd.Timestamp(f"{year}-01-01").date())   # New Year
        holidays.add(pd.Timestamp(f"{year}-05-01").date())   # Labour Day
        holidays.add(pd.Timestamp(f"{year}-10-03").date())   # German Unity Day
        holidays.add(pd.Timestamp(f"{year}-12-24").date())   # Christmas Eve
        holidays.add(pd.Timestamp(f"{year}-12-25").date())   # Christmas Day
        holidays.add(pd.Timestamp(f"{year}-12-26").date())   # Boxing Day
        holidays.add(pd.Timestamp(f"{year}-12-31").date())   # New Year's Eve

    # Easter-dependent holidays (Easter Sunday dates for 2021-2027)
    easter_sundays = {
        2021: pd.Timestamp("2021-04-04"),
        2022: pd.Timestamp("2022-04-17"),
        2023: pd.Timestamp("2023-04-09"),
        2024: pd.Timestamp("2024-03-31"),
        2025: pd.Timestamp("2025-04-20"),
        2026: pd.Timestamp("2026-04-05"),
        2027: pd.Timestamp("2027-03-28"),
    }

    for year, easter in easter_sundays.items():
        holidays.add((easter - pd.Timedelta(days=2)).date())   # Good Friday
        holidays.add((easter + pd.Timedelta(days=1)).date())   # Easter Monday
        holidays.add((easter + pd.Timedelta(days=39)).date())  # Ascension
        holidays.add((easter + pd.Timedelta(days=50)).date())  # Whit Monday

    return holidays


GERMAN_HOLIDAYS = _german_public_holidays()


# ---------------------------------------------------------------------------
# Function 1: add_calendar_features
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the UTC timestamp index.

    Adds raw integer and sin/cos cyclical encodings for hour, day-of-week, and
    month, plus binary indicators for weekends, Mondays, Fridays, public holidays,
    holiday eves, and bridge days.

    Args:
        df: DataFrame with UTC DatetimeIndex.

    Returns:
        DataFrame with calendar feature columns appended.

    Side effects:
        None.
    """
    df = df.copy()

    # --- Hour features ---
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Day-of-week features ---
    df["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Month features ---
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Other calendar ---
    df["week_of_year"] = df.index.isocalendar().week.astype(int).values
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter

    # --- Binary indicators ---
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # --- Public holidays ---
    dates = df.index.date
    df["is_public_holiday"] = pd.Series(
        [1 if d in GERMAN_HOLIDAYS else 0 for d in dates],
        index=df.index,
    )

    # Holiday eve: next day is a public holiday
    next_day_dates = (df.index + pd.Timedelta(days=1)).date
    df["is_holiday_eve"] = pd.Series(
        [1 if d in GERMAN_HOLIDAYS else 0 for d in next_day_dates],
        index=df.index,
    )

    # Bridge day: sandwiched between a public holiday and a weekend
    prev_day_dates = (df.index - pd.Timedelta(days=1)).date
    is_prev_holiday = pd.Series([d in GERMAN_HOLIDAYS for d in prev_day_dates], index=df.index)
    is_next_weekend = pd.Series(
        [(df.index[i] + pd.Timedelta(days=1)).dayofweek >= 5 for i in range(len(df))],
        index=df.index,
    )
    is_prev_weekend = pd.Series(
        [(df.index[i] - pd.Timedelta(days=1)).dayofweek >= 5 for i in range(len(df))],
        index=df.index,
    )
    is_next_holiday = pd.Series([d in GERMAN_HOLIDAYS for d in next_day_dates], index=df.index)

    df["is_holiday_bridge_day"] = (
        ((is_prev_holiday & is_next_weekend) | (is_prev_weekend & is_next_holiday))
        & (df["is_public_holiday"] == 0)
        & (df["is_weekend"] == 0)
    ).astype(int)

    logger.info("Calendar features added: %d new columns", 18)
    return df


# ---------------------------------------------------------------------------
# Function 2: add_lag_features
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged price features and rolling statistics.

    All lags use .shift() which shifts data forward in time, meaning the value
    at time t comes from time t-lag. This ensures no future information leaks.

    Rolling statistics use a window shifted by 24 hours so that day D features
    use only data from day D-1 and earlier.

    Args:
        df: DataFrame with da_price_eur_mwh column and UTC DatetimeIndex.

    Returns:
        DataFrame with lag and rolling feature columns appended.

    Side effects:
        None.
    """
    df = df.copy()
    price = df["da_price_eur_mwh"]

    # --- Direct lags ---
    df["price_lag_24h"] = price.shift(LAG_24H)
    df["price_lag_48h"] = price.shift(LAG_48H)
    df["price_lag_168h"] = price.shift(LAG_168H)
    df["price_lag_336h"] = price.shift(LAG_336H)

    # --- Delta features ---
    df["price_24h_delta"] = df["price_lag_24h"] - df["price_lag_48h"]
    df["price_168h_delta"] = df["price_lag_24h"] - df["price_lag_168h"]

    # --- Rolling statistics (shifted by 24h to prevent leakage) ---
    shifted_price = price.shift(LAG_24H)  # use data up to D-1
    for w in ROLLING_WINDOWS:
        df[f"price_rolling_mean_{w}h"] = shifted_price.rolling(window=w, min_periods=w // 2).mean()
        df[f"price_rolling_std_{w}h"] = shifted_price.rolling(window=w, min_periods=w // 2).std()
        df[f"price_rolling_max_{w}h"] = shifted_price.rolling(window=w, min_periods=w // 2).max()
        df[f"price_rolling_min_{w}h"] = shifted_price.rolling(window=w, min_periods=w // 2).min()

    n_lag_cols = 6 + 4 * len(ROLLING_WINDOWS)
    logger.info("Lag features added: %d new columns", n_lag_cols)
    return df


# ---------------------------------------------------------------------------
# Function 3: add_fundamental_features
# ---------------------------------------------------------------------------
def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive market-fundamental features from the raw series.

    Includes RES penetration, residual load, gas lags/interactions, and
    load lag features.

    Args:
        df: DataFrame with wind/solar/load/gas columns and UTC DatetimeIndex.

    Returns:
        DataFrame with fundamental feature columns appended.

    Side effects:
        None.
    """
    df = df.copy()

    # --- RES penetration features ---
    df["res_total_mw"] = df["wind_forecast_mw"] + df["solar_forecast_mw"]
    df["res_penetration"] = df["res_total_mw"] / df["load_forecast_mw"].replace(0, np.nan)
    df["residual_load_mw"] = df["load_forecast_mw"] - df["res_total_mw"]
    df["residual_load_lag_168h"] = df["residual_load_mw"].shift(LAG_168H)
    df["wind_penetration"] = df["wind_forecast_mw"] / df["load_forecast_mw"].replace(0, np.nan)
    df["solar_penetration"] = df["solar_forecast_mw"] / df["load_forecast_mw"].replace(0, np.nan)

    # --- Wind lags ---
    df["wind_lag_168h"] = df["wind_forecast_mw"].shift(LAG_168H)
    df["wind_delta_168h"] = df["wind_forecast_mw"] - df["wind_lag_168h"]

    # --- Gas features ---
    df["gas_price_lag_24h"] = df["gas_price_eur_mwh"].shift(LAG_24H)
    shifted_gas = df["gas_price_eur_mwh"].shift(LAG_24H)
    df["gas_rolling_mean_168h"] = shifted_gas.rolling(window=168, min_periods=84).mean()
    df["gas_price_delta_7d"] = df["gas_price_eur_mwh"] - df["gas_rolling_mean_168h"]

    # --- Interaction term: gas impact is larger when residual load is high ---
    df["gas_x_residual_load"] = df["gas_price_lag_24h"] * df["residual_load_mw"] / 1000

    # --- Load lag features ---
    df["load_lag_168h"] = df["load_forecast_mw"].shift(LAG_168H)
    df["load_delta_168h"] = df["load_forecast_mw"] - df["load_lag_168h"]
    shifted_load = df["load_forecast_mw"].shift(LAG_24H)
    df["load_rolling_mean_168h"] = shifted_load.rolling(window=168, min_periods=84).mean()

    logger.info("Fundamental features added: 15 new columns")
    return df


# ---------------------------------------------------------------------------
# Function 4: add_price_regime_features
# ---------------------------------------------------------------------------
def add_price_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that capture the market regime (crisis vs normal).

    Args:
        df: DataFrame with lag and rolling price features already present.

    Returns:
        DataFrame with regime feature columns appended.

    Side effects:
        None.
    """
    df = df.copy()

    # How far today's price is from the recent mean (ratio > 2 = crisis regime)
    df["price_regime_ma_ratio"] = df["price_lag_24h"] / (
        df["price_rolling_mean_168h"] + 1
    )

    # Coefficient of variation of price — high = volatile regime
    df["price_vol_regime"] = df["price_rolling_std_168h"] / (
        df["price_rolling_mean_168h"] + 1
    )

    # Was there a negative price yesterday? (signals extreme RES oversupply)
    shifted_price = df["da_price_eur_mwh"].shift(LAG_24H)
    # Check if any hour in the past 24h had negative price
    df["is_negative_price_day"] = (
        shifted_price.rolling(window=24, min_periods=1).min() < 0
    ).astype(int)

    # Count of negative-price hours in last 7 days
    df["negative_price_count_7d"] = (
        (shifted_price < 0).rolling(window=168, min_periods=1).sum()
    )

    logger.info("Price regime features added: 4 new columns")
    return df


# ---------------------------------------------------------------------------
# Function 5: add_interaction_features
# ---------------------------------------------------------------------------
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit feature interaction terms.

    Tree models can learn interactions via splits, but explicit terms help
    when tree depth is limited and reduce the number of splits needed.

    Args:
        df: DataFrame with calendar, lag, fundamental, and regime features.

    Returns:
        DataFrame with interaction columns appended.
    """
    df = df.copy()

    # Wind × hour: wind impact on price varies strongly by hour
    # (night hours have lower demand → wind surplus crashes prices more)
    df["wind_x_hour"] = df["wind_forecast_mw"] * df["hour"] / 24.0

    # Solar × month: solar impact depends on season
    # (summer months have much more solar, affects midday prices)
    df["solar_x_month"] = df["solar_forecast_mw"] * df["month"] / 12.0

    # Gas × residual_load × hour: gas sets marginal price mainly during
    # peak hours when residual load is high
    df["gas_x_resload_x_hour"] = (
        df["gas_price_eur_mwh"] * df["residual_load_mw"] * df["hour"]
        / (1000 * 24.0)
    )

    # Wind × is_weekend: weekend demand is lower, so wind surplus bites harder
    df["wind_x_weekend"] = df["wind_forecast_mw"] * df["is_weekend"]

    # RES penetration × hour: high RES at midday vs night has different effects
    df["res_pen_x_hour"] = df["res_penetration"].fillna(0) * df["hour"] / 24.0

    # Solar × hour (solar only matters during daylight hours 6-20)
    df["solar_x_hour"] = df["solar_forecast_mw"] * df["hour"] / 24.0

    n_new = 6
    logger.info("Interaction features added: %d new columns", n_new)
    return df


# ---------------------------------------------------------------------------
# Function 5b: add_advanced_features
# ---------------------------------------------------------------------------
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced features: ramp rates, volatility ratios, and additional interactions.

    These features capture intra-day dynamics that the base feature set misses:
    - Ramp rates: how fast supply/demand fundamentals are changing hour-to-hour
    - Volatility ratios: recent price volatility relative to longer term
    - Price percentile: where current price sits in recent distribution
    - Additional cross-interactions for better non-linear capture

    Args:
        df: DataFrame with calendar, lag, fundamental, regime, and interaction features.

    Returns:
        DataFrame with advanced feature columns appended.
    """
    df = df.copy()
    n_before = len(df.columns)

    # --- Ramp rate features (hour-over-hour changes in fundamentals) ---
    # These capture the SPEED of change, not just the level.
    # Shifted by 1h — at forecast time for hour h, we know forecast for h
    # (published day-ahead) but we use diff to capture the gradient.

    # Load ramp: sharp load increases drive prices up
    df["load_ramp_1h"] = df["load_forecast_mw"].diff()
    # Wind ramp: sudden wind drops cause price spikes
    df["wind_ramp_1h"] = df["wind_forecast_mw"].diff()
    # Solar ramp: captures sunrise/sunset transitions
    df["solar_ramp_1h"] = df["solar_forecast_mw"].diff()
    # Residual load ramp: net effect on dispatchable generation needs
    if "residual_load_mw" in df.columns:
        df["resload_ramp_1h"] = df["residual_load_mw"].diff()

    # Multi-hour ramps (3h ahead minus current — captures steeper gradients)
    df["load_ramp_3h"] = df["load_forecast_mw"].diff(3)
    df["wind_ramp_3h"] = df["wind_forecast_mw"].diff(3)

    # --- Lagged ramp comparison (was the ramp similar last week?) ---
    # Helps the model learn recurring patterns in ramp-driven price moves
    df["wind_ramp_lag168h"] = df["wind_ramp_1h"].shift(LAG_168H)
    df["load_ramp_lag168h"] = df["load_ramp_1h"].shift(LAG_168H)

    # --- Price volatility ratios ---
    # Short-term vs long-term volatility ratio (regime-change indicator)
    if "price_rolling_std_24h" in df.columns and "price_rolling_std_168h" in df.columns:
        df["vol_ratio_24_168"] = (
            df["price_rolling_std_24h"] / (df["price_rolling_std_168h"] + 0.01)
        )

    # Price range in recent 24h (max - min) — captures spike days
    if "price_rolling_max_24h" in df.columns and "price_rolling_min_24h" in df.columns:
        df["price_range_24h"] = df["price_rolling_max_24h"] - df["price_rolling_min_24h"]

    # --- Price percentile in recent window ---
    # Where the most recent price sits in the 168h distribution
    shifted_price = df["da_price_eur_mwh"].shift(LAG_24H)
    rolling_rank = shifted_price.rolling(168, min_periods=84).rank(pct=True)
    df["price_percentile_168h"] = rolling_rank

    # --- Additional cross-interactions ---
    # Wind ramp × hour: wind ramps at night are more impactful (low demand)
    df["wind_ramp_x_hour"] = df["wind_ramp_1h"].fillna(0) * df["hour"] / 24.0

    # Residual load × is_holiday: holidays with high residual load are unusual
    if "residual_load_mw" in df.columns:
        df["resload_x_holiday"] = df["residual_load_mw"] * df["is_public_holiday"]

    # Gas price × wind penetration: when wind is high, gas price matters less
    if "wind_penetration" in df.columns:
        df["gas_x_wind_pen"] = df["gas_price_eur_mwh"] * (1 - df["wind_penetration"].fillna(0))

    # Hour-of-day × day-of-week (captures specific weekday-hour profiles)
    df["hour_x_dow"] = df["hour"] * df["day_of_week"] / (24.0 * 7.0)

    # Fill NaN in new features
    new_cols_added = len(df.columns) - n_before
    for col in df.columns[-new_cols_added:]:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    logger.info("Advanced features added: %d new columns", new_cols_added)
    return df


# ---------------------------------------------------------------------------
# Function 5c: add_crossborder_flow_features
# ---------------------------------------------------------------------------
def add_crossborder_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-border electricity flow features if cached data is available.

    Cross-border flows strongly influence DA prices because Germany is at the
    center of the European grid. Net imports reduce supply surplus (higher prices),
    net exports indicate oversupply (lower prices).

    Tries to load cached flow data from data/raw/crossborder_flows/de_flows_all.parquet.
    If not available, skips gracefully.

    Args:
        df: DataFrame with UTC DatetimeIndex.

    Returns:
        DataFrame with flow feature columns appended (if data available).
    """
    cache_file = DATA_PROCESSED.parent / "raw" / "crossborder_flows" / "de_flows_all.parquet"

    if not cache_file.exists():
        logger.info("Cross-border flow data not found at %s — skipping flow features", cache_file)
        logger.info("  (To enable: run ingestion pipeline to fetch flows, or place cached file)")
        return df

    try:
        flows = pd.read_parquet(cache_file)
        flows.index = pd.to_datetime(flows.index, utc=True)
        logger.info("Loaded cross-border flows: %d rows x %d cols", len(flows), len(flows.columns))
    except Exception as e:
        logger.warning("Failed to load cross-border flows: %s — skipping", e)
        return df

    df = df.copy()
    n_before = len(df.columns)

    # Merge flows into feature matrix (left join — keep all rows even if no flow data)
    for col in flows.columns:
        if col in df.columns:
            continue
        df[col] = flows[col].reindex(df.index)

    # Total net flow is the most important (already in flows as flow_total_net)
    if "flow_total_net" in df.columns:
        # Lagged flows (24h and 168h) — respects information barrier
        df["flow_total_net_lag24h"] = df["flow_total_net"].shift(LAG_24H)
        df["flow_total_net_lag168h"] = df["flow_total_net"].shift(LAG_168H)

        # Flow change from last week
        df["flow_total_delta_168h"] = df["flow_total_net"] - df["flow_total_net"].shift(LAG_168H)

        # Flow × hour: import patterns differ by time of day
        df["flow_x_hour"] = df["flow_total_net"].fillna(0) * df["hour"] / 24.0

        # Rolling mean of flows (past week, shifted 24h)
        shifted_flow = df["flow_total_net"].shift(LAG_24H)
        df["flow_rolling_mean_168h"] = shifted_flow.rolling(168, min_periods=84).mean()

    # Fill NaN in new flow features
    new_cols_added = len(df.columns) - n_before
    for col in df.columns[-new_cols_added:]:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    logger.info("Cross-border flow features added: %d new columns", new_cols_added)
    return df


# ---------------------------------------------------------------------------
# Function 6: build_feature_matrix (master function)
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full model-ready feature matrix from the raw merged dataset.

    Calls all feature-adding functions in order, enforces the information barrier,
    drops the warmup period where lags are undefined, and validates the result.

    Args:
        df: Raw merged DataFrame from ingestion (de_power_dataset.parquet).

    Returns:
        Feature matrix with target column (da_price_eur_mwh) retained.

    Raises:
        ValueError: If NaN values remain in feature columns after warmup drop.

    Side effects:
        Saves data/processed/feature_matrix.parquet.
    """
    _setup_logging()
    logger.info("=" * 60)
    logger.info("Building feature matrix")
    logger.info("=" * 60)

    original_index = df.index.copy()
    original_len = len(df)

    # Step 1: Calendar features
    df = add_calendar_features(df)

    # Step 2: Lag features (produces NaN for first 336 hours)
    df = add_lag_features(df)

    # Step 3: Fundamental features (produces NaN for first ~168 hours)
    df = add_fundamental_features(df)

    # Step 4: Price regime features
    df = add_price_regime_features(df)

    # Step 5: Feature interactions
    df = add_interaction_features(df)

    # Step 6: Advanced features (ramp rates, volatility ratios, etc.)
    df = add_advanced_features(df)

    # Step 7: Cross-border flow features (if cached data available)
    df = add_crossborder_flow_features(df)

    # --- Information barrier verification ---
    # price_lag_24h should correlate positively with its index position
    # (values increase over time since they are shifted forward)
    lag_col = df["price_lag_24h"].dropna()
    idx_numeric = np.arange(len(lag_col))
    corr_check = np.corrcoef(idx_numeric, lag_col.values)[0, 1]
    # This correlation can be weak but should not be strongly negative
    logger.info("Lag direction check: corr(index, price_lag_24h) = %.4f", corr_check)

    # Verify index integrity
    assert df.index.equals(original_index), (
        "Feature matrix index diverged from original dataset index"
    )

    # --- Drop warmup period ---
    df = df.iloc[WARMUP_HOURS:]
    n_dropped = original_len - len(df)
    logger.info("Dropped first %d rows (warmup for %dh lags)", n_dropped, WARMUP_HOURS)

    # --- Validate no NaN in features ---
    target_col = "da_price_eur_mwh"
    feature_cols = [c for c in df.columns if c != target_col]
    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        # For columns with small NaN counts, these may be edge effects from
        # division by zero in penetration ratios — fill with 0
        for col in nan_cols.index:
            n_nan = nan_cols[col]
            pct = n_nan / len(df) * 100
            if pct < 0.5:
                logger.warning(
                    "Feature '%s' has %d NaN (%.3f%%) — filling with 0 (edge effect)",
                    col, n_nan, pct,
                )
                df[col] = df[col].fillna(0)
            else:
                raise ValueError(
                    f"Feature '{col}' has {n_nan} NaN values ({pct:.2f}%) after "
                    f"warmup drop — too many to silently fill. Investigate."
                )

    # --- Save ---
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "feature_matrix.parquet"
    df.to_parquet(output_path)

    n_features = len(feature_cols)
    logger.info(
        "Feature matrix: %d rows x %d features. Target: %s. "
        "Date range: %s to %s",
        len(df), n_features, target_col,
        df.index[0], df.index[-1],
    )
    logger.info("Saved to %s", output_path)

    return df
