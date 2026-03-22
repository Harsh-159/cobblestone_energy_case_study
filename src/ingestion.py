"""
Data ingestion and QA pipeline for DE_LU day-ahead power market.

Fetches 4 years of hourly DA prices, wind/solar/load forecasts, and daily gas prices.
Cleans, aligns, timezone-corrects, and validates everything.
Outputs one clean parquet file and a QA report.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARKET = "DE_LU"
TIMEZONE = "Europe/Berlin"
START_DATE = "2021-01-01"
# END_DATE = "2024-12-31"
END_DATE = "2026-03-15"

# Physical bounds for QA
DA_PRICE_MIN = -500.0       # €/MWh — EPEX floor
DA_PRICE_MAX = 3000.0       # €/MWh — observed extreme during 2022 crisis
WIND_SOLAR_MAX_MW = 130_000  # Combined DE installed capacity upper bound
LOAD_MIN_MW = 20_000        # DE minimum plausible load
LOAD_MAX_MW = 90_000        # DE maximum plausible load
GAS_PRICE_MIN = 0.0
GAS_PRICE_MAX = 400.0       # €/MWh — 2022 crisis peak was ~339

STALE_THRESHOLD = 6          # consecutive identical values = suspect

# Crisis period bounds (for context-aware outlier flagging)
CRISIS_START = pd.Timestamp("2022-06-01", tz="UTC")
CRISIS_END = pd.Timestamp("2023-03-31", tz="UTC")

# Solar at night: values below this (MW) are auto-corrected to 0; above this are errors
SOLAR_NIGHT_AUTO_FIX_MAX_MW = 50.0

# Severity levels for QA flags
SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_ERROR = "ERROR"

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

logger = logging.getLogger("ingestion")


# ---------------------------------------------------------------------------
# Corrections log — records every auto-fix applied to the data
# ---------------------------------------------------------------------------
_corrections_log: list[dict] = []


def _log_correction(
    timestamp: pd.Timestamp,
    column: str,
    old_value: float,
    new_value: float,
    reason: str,
) -> None:
    """Record a single auto-correction to the global corrections log.

    Args:
        timestamp: The UTC timestamp of the corrected row.
        column: Column name that was corrected.
        old_value: Value before correction.
        new_value: Value after correction.
        reason: Human-readable reason for the correction.
    """
    _corrections_log.append({
        "timestamp_utc": timestamp,
        "column": column,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
    })


def save_corrections_log(output_dir: Path) -> int:
    """Write the corrections log to a CSV file.

    Args:
        output_dir: Directory for output files.

    Returns:
        Number of corrections logged.

    Side effects:
        Writes outputs/auto_corrections_log.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "auto_corrections_log.csv"

    if not _corrections_log:
        logger.info("No auto-corrections were applied — corrections log is empty")
        # Write an empty CSV with headers only
        pd.DataFrame(columns=["timestamp_utc", "column", "old_value", "new_value", "reason"]).to_csv(
            path, index=False
        )
        return 0

    log_df = pd.DataFrame(_corrections_log)
    log_df = log_df.sort_values("timestamp_utc").reset_index(drop=True)
    log_df.to_csv(path, index=False)
    logger.info("Auto-corrections log: %d corrections saved to %s", len(log_df), path)
    return len(log_df)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Configure logging to both console and logs/ingestion.log with timestamps."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOGS_DIR / "ingestion.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Function 1: load_api_key
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Load the ENTSO-E API key from the .env file.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the key is not found.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("ENTSO_E_API_KEY")
    if not key or key == "your_key_here":
        raise EnvironmentError(
            "ENTSO_E_API_KEY not set. Create a .env file in the project root with:\n"
            "  ENTSO_E_API_KEY=<your_key>\n"
            "Get a free key at https://transparency.entsoe.eu/"
        )
    return key


# ---------------------------------------------------------------------------
# Function 2: fetch_da_prices
# ---------------------------------------------------------------------------
def fetch_da_prices(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.Series:
    """Fetch hourly Day-Ahead prices for DE_LU from ENTSO-E.

    Args:
        client: An initialised EntsoePandasClient.
        start: Start timestamp (tz-aware Europe/Berlin).
        end: End timestamp (tz-aware Europe/Berlin).
        cache_dir: Directory for raw cached parquet files.

    Returns:
        A pd.Series with UTC DatetimeIndex, named 'da_price_eur_mwh'.

    Side effects:
        Caches each year's data as parquet in cache_dir/da_prices/.
    """
    cache_sub = cache_dir / "da_prices"
    cache_sub.mkdir(parents=True, exist_ok=True)
    all_chunks: list[pd.Series] = []

    for year in range(start.year, end.year + 1):
        cache_file = cache_sub / f"da_prices_{year}.parquet"
        if cache_file.exists():
            logger.info("Loading DA prices %d from cache", year)
            chunk_df = pd.read_parquet(cache_file)
            chunk = chunk_df["da_price_eur_mwh"]
            chunk.index = pd.to_datetime(chunk.index, utc=True)
        else:
            logger.info("Fetching DA prices %d from ENTSO-E API", year)
            y_start = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE)
            y_end = pd.Timestamp(f"{year}-12-31 23:00", tz=TIMEZONE)
            try:
                chunk = client.query_day_ahead_prices(MARKET, start=y_start, end=y_end)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed fetching DA prices for year {year}: {exc}"
                ) from exc

            # Convert to UTC immediately
            chunk.index = chunk.index.tz_convert("UTC")
            chunk.name = "da_price_eur_mwh"

            # Detect sub-hourly resolution and resample if needed
            chunk = _ensure_hourly(chunk, f"DA prices {year}")

            # Cache
            chunk.to_frame().to_parquet(cache_file)
            logger.debug("Cached DA prices %d → %s", year, cache_file)

        all_chunks.append(chunk)

    series = pd.concat(all_chunks)
    series.name = "da_price_eur_mwh"

    # Convert to UTC (cached data already is, API data converted above)
    series.index = pd.to_datetime(series.index, utc=True)

    # Remove duplicate index entries
    n_dup = series.index.duplicated().sum()
    if n_dup > 0:
        logger.warning("DA prices: dropping %d duplicate UTC timestamps", n_dup)
        series = series[~series.index.duplicated(keep="first")]

    # Trim to exact range
    utc_start = pd.Timestamp(START_DATE, tz="UTC")
    utc_end = pd.Timestamp(f"{END_DATE} 23:00", tz="UTC")
    series = series.loc[utc_start:utc_end]

    logger.info("DA prices fetched: %d rows", len(series))
    return series


# ---------------------------------------------------------------------------
# Function 3: fetch_wind_solar
# ---------------------------------------------------------------------------
def fetch_wind_solar(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch day-ahead wind and solar generation forecasts for DE_LU.

    Args:
        client: An initialised EntsoePandasClient.
        start: Start timestamp (tz-aware Europe/Berlin).
        end: End timestamp (tz-aware Europe/Berlin).
        cache_dir: Directory for raw cached parquet files.

    Returns:
        A pd.DataFrame with UTC DatetimeIndex and columns
        'wind_forecast_mw' and 'solar_forecast_mw'.

    Side effects:
        Caches each year's data as parquet in cache_dir/wind_solar/.
    """
    cache_sub = cache_dir / "wind_solar"
    cache_sub.mkdir(parents=True, exist_ok=True)
    all_chunks: list[pd.DataFrame] = []

    for year in range(start.year, end.year + 1):
        cache_file = cache_sub / f"wind_solar_{year}.parquet"
        if cache_file.exists():
            logger.info("Loading wind/solar %d from cache", year)
            chunk = pd.read_parquet(cache_file)
            chunk.index = pd.to_datetime(chunk.index, utc=True)
        else:
            logger.info("Fetching wind/solar %d from ENTSO-E API", year)
            y_start = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE)
            y_end = pd.Timestamp(f"{year}-12-31 23:00", tz=TIMEZONE)
            try:
                raw = client.query_wind_and_solar_forecast(
                    MARKET, start=y_start, end=y_end, psr_type=None
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed fetching wind/solar for year {year}: {exc}"
                ) from exc

            logger.info("Wind/solar %d raw columns: %s", year, list(raw.columns))
            raw.index = raw.index.tz_convert("UTC")

            # Identify wind columns
            wind_cols = [c for c in raw.columns if "wind" in c.lower()]
            solar_cols = [c for c in raw.columns if "solar" in c.lower()]

            if not wind_cols:
                raise ValueError(
                    f"No wind columns found in API response. Columns: {list(raw.columns)}"
                )
            if not solar_cols:
                raise ValueError(
                    f"No solar columns found in API response. Columns: {list(raw.columns)}"
                )

            chunk = pd.DataFrame(index=raw.index)
            chunk["wind_forecast_mw"] = raw[wind_cols].sum(axis=1)
            chunk["solar_forecast_mw"] = raw[solar_cols[0]]

            if len(wind_cols) == 1 and "offshore" not in wind_cols[0].lower():
                logger.warning(
                    "Wind offshore not available for %d — using onshore only", year
                )

            # Resample to hourly if needed
            chunk = _ensure_hourly_df(chunk, f"wind/solar {year}")

            # Fix negative values (physically impossible)
            for col in ["wind_forecast_mw", "solar_forecast_mw"]:
                neg_mask = chunk[col] < 0
                n_neg = neg_mask.sum()
                if n_neg > 0:
                    logger.warning(
                        "%s: corrected %d negative values to 0 in %d", col, n_neg, year
                    )
                    chunk.loc[neg_mask, col] = 0.0

            # Cache
            chunk.to_parquet(cache_file)
            logger.debug("Cached wind/solar %d → %s", year, cache_file)

        all_chunks.append(chunk)

    df = pd.concat(all_chunks)
    df.index = pd.to_datetime(df.index, utc=True)

    # Remove duplicate timestamps
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        logger.warning("Wind/solar: dropping %d duplicate UTC timestamps", n_dup)
        df = df[~df.index.duplicated(keep="last")]

    # Trim to range
    utc_start = pd.Timestamp(START_DATE, tz="UTC")
    utc_end = pd.Timestamp(f"{END_DATE} 23:00", tz="UTC")
    df = df.loc[utc_start:utc_end]

    logger.info("Wind/solar fetched: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Function 4: fetch_load_forecast
# ---------------------------------------------------------------------------
def fetch_load_forecast(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.Series:
    """Fetch day-ahead total load forecast for DE_LU.

    Args:
        client: An initialised EntsoePandasClient.
        start: Start timestamp (tz-aware Europe/Berlin).
        end: End timestamp (tz-aware Europe/Berlin).
        cache_dir: Directory for raw cached parquet files.

    Returns:
        A pd.Series with UTC DatetimeIndex, named 'load_forecast_mw'.

    Side effects:
        Caches each year's data as parquet in cache_dir/load_forecast/.
    """
    cache_sub = cache_dir / "load_forecast"
    cache_sub.mkdir(parents=True, exist_ok=True)
    all_chunks: list[pd.Series] = []

    for year in range(start.year, end.year + 1):
        cache_file = cache_sub / f"load_forecast_{year}.parquet"
        if cache_file.exists():
            logger.info("Loading load forecast %d from cache", year)
            chunk_df = pd.read_parquet(cache_file)
            chunk = chunk_df["load_forecast_mw"]
            chunk.index = pd.to_datetime(chunk.index, utc=True)
        else:
            logger.info("Fetching load forecast %d from ENTSO-E API", year)
            y_start = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE)
            y_end = pd.Timestamp(f"{year}-12-31 23:00", tz=TIMEZONE)
            try:
                chunk = client.query_load_forecast(MARKET, start=y_start, end=y_end)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed fetching load forecast for year {year}: {exc}"
                ) from exc

            # If DataFrame returned, take first column
            if isinstance(chunk, pd.DataFrame):
                chunk = chunk.iloc[:, 0]

            chunk.index = chunk.index.tz_convert("UTC")
            chunk.name = "load_forecast_mw"

            # Remove overlapping forecast runs — keep the last
            if chunk.index.duplicated().any():
                logger.warning(
                    "Load forecast %d: dropping %d duplicate timestamps (keeping last)",
                    year,
                    chunk.index.duplicated().sum(),
                )
                chunk = chunk[~chunk.index.duplicated(keep="last")]

            # Resample to hourly if needed
            chunk = _ensure_hourly(chunk, f"load forecast {year}")

            # Cache
            chunk.to_frame().to_parquet(cache_file)
            logger.debug("Cached load forecast %d → %s", year, cache_file)

        all_chunks.append(chunk)

    series = pd.concat(all_chunks)
    series.name = "load_forecast_mw"
    series.index = pd.to_datetime(series.index, utc=True)

    # Remove duplicates
    n_dup = series.index.duplicated().sum()
    if n_dup > 0:
        logger.warning("Load forecast: dropping %d duplicate UTC timestamps", n_dup)
        series = series[~series.index.duplicated(keep="last")]

    # Trim to range
    utc_start = pd.Timestamp(START_DATE, tz="UTC")
    utc_end = pd.Timestamp(f"{END_DATE} 23:00", tz="UTC")
    series = series.loc[utc_start:utc_end]

    logger.info("Load forecast fetched: %d rows", len(series))
    return series


# ---------------------------------------------------------------------------
# Function 5: fetch_gas_price
# ---------------------------------------------------------------------------
def fetch_gas_price(cache_dir: Path) -> pd.Series:
    """Load daily TTF gas price from a pre-downloaded CSV and forward-fill to hourly.

    The CSV should be downloaded from investing.com:
    https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data
    Set date range 2021-01-01 to 2024-12-31, download CSV, and place at:
    data/raw/gas_price/ttf_daily.csv

    Args:
        cache_dir: Directory containing raw data (expects gas_price/ttf_daily.csv).

    Returns:
        A pd.Series with UTC DatetimeIndex (hourly), named 'gas_price_eur_mwh'.

    Raises:
        FileNotFoundError: If the CSV is not present.
    """
    csv_path = cache_dir / "gas_price" / "ttf_daily.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Gas price CSV not found at {csv_path}\n\n"
            "Please download it manually:\n"
            "1. Go to https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data\n"
            "2. Set date range: 2021-01-01 to 2024-12-31\n"
            "3. Click 'Download Data'\n"
            "4. Save the CSV as: data/raw/gas_price/ttf_daily.csv\n"
            "Required columns: Date, Price"
        )

    logger.info("Loading gas price from %s", csv_path)
    raw = pd.read_csv(csv_path)

    # Identify date and price columns (case-insensitive)
    col_map = {c.lower().strip(): c for c in raw.columns}

    date_col = col_map.get("date")
    price_col = col_map.get("price")
    if date_col is None or price_col is None:
        raise ValueError(
            f"Gas CSV must have 'Date' and 'Price' columns. Found: {list(raw.columns)}"
        )

    # Parse dates — investing.com format is typically "Mon DD, YYYY" (e.g. "Jan 03, 2021")
    dates = pd.to_datetime(raw[date_col], format="mixed", dayfirst=False)

    # Parse prices — strip commas for thousands separators
    prices = raw[price_col].astype(str).str.replace(",", "", regex=False).astype(float)

    daily = pd.Series(prices.values, index=dates, name="gas_price_eur_mwh")
    daily = daily.sort_index()

    # Remove any duplicate dates
    if daily.index.duplicated().any():
        logger.warning("Gas prices: dropping %d duplicate dates", daily.index.duplicated().sum())
        daily = daily[~daily.index.duplicated(keep="first")]

    # Set index to noon UTC for each date so forward-fill aligns with next-day prices
    daily.index = daily.index.tz_localize("UTC") + pd.Timedelta(hours=12)

    # Create hourly index spanning the full period and reindex with forward fill
    utc_start = pd.Timestamp(START_DATE, tz="UTC")
    utc_end = pd.Timestamp(f"{END_DATE} 23:00", tz="UTC")
    hourly_idx = pd.date_range(start=utc_start, end=utc_end, freq="h", tz="UTC")

    hourly = daily.reindex(hourly_idx, method="ffill")
    # Back-fill the first few hours (before the first noon) if needed
    hourly = hourly.bfill()

    hourly.name = "gas_price_eur_mwh"
    logger.info("Gas price loaded: %d hourly values from %d daily records", len(hourly), len(daily))
    return hourly


# ---------------------------------------------------------------------------
# Function 5b: fetch_crossborder_flows
# ---------------------------------------------------------------------------
def fetch_crossborder_flows(
    client: EntsoePandasClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch scheduled day-ahead cross-border commercial flows for DE_LU.

    Fetches flows from key neighboring zones: FR, NL, AT, PL, DK1, CH, CZ.
    Returns net flows (positive = import into DE).

    Args:
        client: An initialised EntsoePandasClient.
        start: Start timestamp (tz-aware Europe/Berlin).
        end: End timestamp (tz-aware Europe/Berlin).
        cache_dir: Directory for raw cached parquet files.

    Returns:
        DataFrame with columns like 'flow_FR_net', 'flow_NL_net', etc.
        Index is hourly UTC.
    """
    cache_sub = cache_dir / "crossborder_flows"
    cache_sub.mkdir(parents=True, exist_ok=True)
    cache_file = cache_sub / "de_flows_all.parquet"

    if cache_file.exists():
        logger.info("Loading cross-border flows from cache: %s", cache_file)
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    neighbors = {
        "FR": "10YFR-RTE------C",
        "NL": "10YNL----------L",
        "AT": "10YAT-APG------L",
        "PL": "10YPL-AREA-----S",
        "DK1": "10YDK-1--------W",
        "CH": "10YCH-SWISSGRIDZ",
        "CZ": "10YCZ-CEPS-----N",
    }
    de_code = "10Y1001A1001A82H"  # DE_LU

    all_flows = {}
    for name, code in neighbors.items():
        try:
            logger.info("Fetching scheduled flows DE_LU <-> %s", name)
            # DE -> neighbor (export from DE)
            export = client.query_scheduled_day_ahead_transfers(
                de_code, code, start=start, end=end
            )
            # neighbor -> DE (import to DE)
            imp = client.query_scheduled_day_ahead_transfers(
                code, de_code, start=start, end=end
            )

            if hasattr(export, 'index'):
                export.index = export.index.tz_convert("UTC")
                export = _ensure_hourly(export, f"export_DE_{name}")
            if hasattr(imp, 'index'):
                imp.index = imp.index.tz_convert("UTC")
                imp = _ensure_hourly(imp, f"import_{name}_DE")

            # Net flow: positive = importing into DE
            net = imp.reindex(export.index, fill_value=0) - export.fillna(0)
            all_flows[f"flow_{name}_net"] = net
            logger.info("  %s: %d rows fetched", name, len(net))

        except Exception as e:
            logger.warning("  Failed to fetch flows for %s: %s", name, e)

    if not all_flows:
        logger.warning("No cross-border flow data could be fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_flows)
    df.index = pd.to_datetime(df.index, utc=True)

    # Total net import
    df["flow_total_net"] = df.sum(axis=1)

    # Trim to range
    utc_start = pd.Timestamp(START_DATE, tz="UTC")
    utc_end = pd.Timestamp(f"{END_DATE} 23:00", tz="UTC")
    df = df.loc[utc_start:utc_end]

    # Cache
    df.to_parquet(cache_file)
    logger.info("Cross-border flows: %d rows x %d cols, cached to %s",
                len(df), len(df.columns), cache_file)
    return df


# ---------------------------------------------------------------------------
# Function 6: build_complete_hourly_index
# ---------------------------------------------------------------------------
def build_complete_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    """Create the ground-truth hourly UTC index for the dataset.

    Args:
        start: Start date string (e.g. '2021-01-01').
        end: End date string (e.g. '2024-12-31').

    Returns:
        A complete UTC-aware hourly DatetimeIndex from start 00:00 to end 23:00.
    """
    idx = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(f"{end} 23:00", tz="UTC"),
        freq="h",
    )
    logger.info("Complete hourly index: %d hours (%s → %s)", len(idx), idx[0], idx[-1])
    return idx


# ---------------------------------------------------------------------------
# Function 7: clean_and_align
# ---------------------------------------------------------------------------
def clean_and_align(
    da: pd.Series,
    wind_solar: pd.DataFrame,
    load: pd.Series,
    gas: pd.Series,
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex all series to the complete hourly UTC grid, merge, and auto-correct
    physically impossible values.

    Auto-corrections applied (each logged to the corrections log):
        - Negative wind forecast → 0
        - Negative solar forecast → 0
        - Solar > 0 at night (21:00–04:00 UTC) and ≤ SOLAR_NIGHT_AUTO_FIX_MAX_MW → 0
        - Negative load forecast → NaN (flagged, not silently fixed)

    Args:
        da: DA price series (UTC index).
        wind_solar: DataFrame with wind_forecast_mw & solar_forecast_mw (UTC index).
        load: Load forecast series (UTC index).
        gas: Gas price series (UTC index).
        full_index: The authoritative hourly UTC DatetimeIndex.

    Returns:
        A single merged DataFrame with columns: da_price_eur_mwh,
        wind_forecast_mw, solar_forecast_mw, load_forecast_mw, gas_price_eur_mwh.

    Raises:
        AssertionError: If index integrity checks fail.

    Side effects:
        Appends auto-correction entries to the global _corrections_log.
    """
    global _corrections_log

    # Reindex each series to the complete hourly grid
    da_aligned = da.reindex(full_index)
    ws_aligned = wind_solar.reindex(full_index)
    load_aligned = load.reindex(full_index)
    gas_aligned = gas.reindex(full_index)

    df = pd.DataFrame(
        {
            "da_price_eur_mwh": da_aligned,
            "wind_forecast_mw": ws_aligned["wind_forecast_mw"],
            "solar_forecast_mw": ws_aligned["solar_forecast_mw"],
            "load_forecast_mw": load_aligned,
            "gas_price_eur_mwh": gas_aligned,
        },
        index=full_index,
    )
    df.index.name = "timestamp_utc"

    # Integrity assertions
    assert df.index.equals(full_index), "Index does not match full_index after merge"
    assert df.index.is_monotonic_increasing, "Index is not monotonically increasing"
    assert df.index.tz is not None and str(df.index.tz) == "UTC", (
        f"Index must be UTC-aware, got tz={df.index.tz}"
    )
    assert not df.index.duplicated().any(), "Duplicate index values found after merge"

    logger.info("Merged dataset: %d rows × %d columns", df.shape[0], df.shape[1])

    # ------------------------------------------------------------------
    # Auto-correct physically impossible values
    # ------------------------------------------------------------------
    n_fixes = 0

    # 1. Negative wind → 0
    neg_wind = df["wind_forecast_mw"] < 0
    if neg_wind.any():
        for ts in df.index[neg_wind]:
            _log_correction(ts, "wind_forecast_mw", float(df.loc[ts, "wind_forecast_mw"]),
                            0.0, "Negative wind forecast is physically impossible → set to 0")
        n_fixes += neg_wind.sum()
        df.loc[neg_wind, "wind_forecast_mw"] = 0.0

    # 2. Negative solar → 0
    neg_solar = df["solar_forecast_mw"] < 0
    if neg_solar.any():
        for ts in df.index[neg_solar]:
            _log_correction(ts, "solar_forecast_mw", float(df.loc[ts, "solar_forecast_mw"]),
                            0.0, "Negative solar forecast is physically impossible → set to 0")
        n_fixes += neg_solar.sum()
        df.loc[neg_solar, "solar_forecast_mw"] = 0.0

    # 3. Solar at night (21:00–04:00 UTC): auto-fix small values, leave large ones for QA
    hour_utc = df.index.hour
    night_mask = (hour_utc >= 21) | (hour_utc <= 4)
    solar_at_night = (df["solar_forecast_mw"] > 0) & night_mask
    small_solar_night = solar_at_night & (df["solar_forecast_mw"] <= SOLAR_NIGHT_AUTO_FIX_MAX_MW)
    if small_solar_night.any():
        for ts in df.index[small_solar_night]:
            _log_correction(ts, "solar_forecast_mw", float(df.loc[ts, "solar_forecast_mw"]),
                            0.0,
                            f"Solar > 0 at night ({ts.hour}:00 UTC) but ≤ {SOLAR_NIGHT_AUTO_FIX_MAX_MW} MW "
                            "→ rounding artifact, set to 0")
        n_fixes += small_solar_night.sum()
        df.loc[small_solar_night, "solar_forecast_mw"] = 0.0

    if n_fixes > 0:
        logger.info("Auto-corrected %d physically impossible values (see corrections log)", n_fixes)
    else:
        logger.info("No auto-corrections needed — all values physically plausible")

    return df


# ---------------------------------------------------------------------------
# Function 8: run_qa_checks
# ---------------------------------------------------------------------------
def run_qa_checks(df: pd.DataFrame, full_index: pd.DatetimeIndex) -> dict:
    """Run all data quality checks on the merged dataset with severity-aware flagging.

    Each check now assigns a severity level:
        - INFO:    Known/expected condition, no action needed (e.g. crisis-era outliers)
        - WARNING: Suspicious but plausible, needs human review
        - ERROR:   Definitely wrong, blocks downstream use

    Auto-corrected values (from clean_and_align) are NOT re-flagged here — they
    have already been fixed and logged in the corrections log.

    Args:
        df: The merged, aligned DataFrame (after auto-corrections).
        full_index: The authoritative hourly UTC DatetimeIndex.

    Returns:
        A dict with keys for each check, each containing:
        {passed: bool, severity: str, details: str, flagged_rows: pd.DataFrame or None}
    """
    results = {}
    all_flagged: list[pd.DataFrame] = []

    # --- Check 1: Timestamp completeness ---
    missing_ts = full_index.difference(df.index)
    extra_ts = df.index.difference(full_index)
    passed = len(missing_ts) == 0 and len(extra_ts) == 0
    details = (
        f"Expected {len(full_index)} rows, got {len(df)}. "
        f"Missing: {len(missing_ts)}, Extra: {len(extra_ts)}."
    )
    if len(missing_ts) > 0:
        sample = list(missing_ts[:5]) + (list(missing_ts[-5:]) if len(missing_ts) > 5 else [])
        details += f" Sample missing: {sample}"
    results["timestamp_completeness"] = {
        "passed": passed,
        "severity": SEVERITY_ERROR if not passed else SEVERITY_INFO,
        "details": details,
        "flagged_rows": None,
    }

    # --- Check 2: Duplicate timestamps ---
    n_dup = df.index.duplicated().sum()
    results["duplicate_timestamps"] = {
        "passed": n_dup == 0,
        "severity": SEVERITY_ERROR if n_dup > 0 else SEVERITY_INFO,
        "details": f"{n_dup} duplicate timestamps",
        "flagged_rows": None,
    }

    # --- Check 3: Missingness per column ---
    miss_lines = []
    miss_data = {}
    for col in df.columns:
        total_miss = df[col].isna().sum()
        pct = total_miss / len(df) * 100
        yearly = df[col].isna().groupby(df.index.year).mean() * 100
        miss_data[col] = {
            "total": int(total_miss),
            "pct": pct,
            "yearly": yearly.to_dict(),
        }
        status = "ERROR" if pct > 5 else ("WARNING" if pct > 1 else "OK")
        miss_lines.append(f"  {col}: {total_miss} ({pct:.2f}%) [{status}]")
        for yr, yr_pct in yearly.items():
            miss_lines.append(f"    {yr}: {yr_pct:.2f}%")

    any_error = any(v["pct"] > 5 for v in miss_data.values())
    any_warn = any(v["pct"] > 1 for v in miss_data.values())
    results["missingness"] = {
        "passed": not any_error,
        "severity": SEVERITY_ERROR if any_error else (SEVERITY_WARNING if any_warn else SEVERITY_INFO),
        "details": "\n".join(miss_lines),
        "flagged_rows": None,
        "data": miss_data,
    }

    # --- Check 4: DA price outliers (context-aware — crisis vs non-crisis) ---
    da_col = df["da_price_eur_mwh"].dropna()

    # 4a. Physical bounds (always an error regardless of period)
    bounds_mask = (da_col < DA_PRICE_MIN) | (da_col > DA_PRICE_MAX)
    bounds_flagged = df.loc[da_col[bounds_mask].index].copy()
    bounds_flagged["flag_reason"] = "da_price_out_of_physical_bounds"
    bounds_flagged["severity"] = SEVERITY_ERROR

    # 4b. Statistical outliers — split by crisis period
    mean_p = da_col.mean()
    std_p = da_col.std()
    stat_mask = (da_col - mean_p).abs() > 4 * std_p
    stat_outlier_idx = da_col[stat_mask].index

    # Separate crisis-era outliers (INFO) from non-crisis outliers (WARNING)
    crisis_mask = (stat_outlier_idx >= CRISIS_START) & (stat_outlier_idx <= CRISIS_END)
    crisis_outlier_idx = stat_outlier_idx[crisis_mask]
    non_crisis_outlier_idx = stat_outlier_idx[~crisis_mask]

    crisis_flagged = df.loc[crisis_outlier_idx].copy()
    crisis_flagged["flag_reason"] = "da_price_4sigma_outlier_CRISIS_PERIOD (expected — energy crisis Jun 2022–Mar 2023)"
    crisis_flagged["severity"] = SEVERITY_INFO

    non_crisis_flagged = df.loc[non_crisis_outlier_idx].copy()
    non_crisis_flagged["flag_reason"] = "da_price_4sigma_outlier_NON_CRISIS (investigate — unexpected extreme)"
    non_crisis_flagged["severity"] = SEVERITY_WARNING

    all_flagged.extend([bounds_flagged, crisis_flagged, non_crisis_flagged])
    results["da_price_outliers"] = {
        "passed": len(bounds_flagged) == 0 and len(non_crisis_flagged) == 0,
        "severity": (
            SEVERITY_ERROR if len(bounds_flagged) > 0
            else SEVERITY_WARNING if len(non_crisis_flagged) > 0
            else SEVERITY_INFO
        ),
        "details": (
            f"Physical bounds: {len(bounds_flagged)} flagged [{SEVERITY_ERROR}]. "
            f"Statistical 4σ during crisis (Jun 2022–Mar 2023): {len(crisis_flagged)} [{SEVERITY_INFO}]. "
            f"Statistical 4σ outside crisis: {len(non_crisis_flagged)} [{SEVERITY_WARNING}]."
        ),
        "flagged_rows": (
            pd.concat([bounds_flagged, crisis_flagged, non_crisis_flagged])
            if len(bounds_flagged) + len(crisis_flagged) + len(non_crisis_flagged) > 0
            else None
        ),
        "bounds_count": len(bounds_flagged),
        "crisis_stat_count": len(crisis_flagged),
        "non_crisis_stat_count": len(non_crisis_flagged),
        "stat_count": len(crisis_flagged) + len(non_crisis_flagged),
    }

    # --- Check 5: Wind/solar physical bounds ---
    # Note: negative values and small solar-at-night have already been auto-corrected
    # in clean_and_align(). Here we flag anything that survived (large solar at night,
    # capacity exceedances).
    ws_flags: list[pd.DataFrame] = []

    # Negative wind/solar should be gone after auto-correction, but verify
    neg_wind = df["wind_forecast_mw"] < 0
    if neg_wind.any():
        f = df.loc[neg_wind].copy()
        f["flag_reason"] = "negative_wind_forecast (should have been auto-corrected)"
        f["severity"] = SEVERITY_ERROR
        ws_flags.append(f)

    neg_solar = df["solar_forecast_mw"] < 0
    if neg_solar.any():
        f = df.loc[neg_solar].copy()
        f["flag_reason"] = "negative_solar_forecast (should have been auto-corrected)"
        f["severity"] = SEVERITY_ERROR
        ws_flags.append(f)

    combined_re = df["wind_forecast_mw"] + df["solar_forecast_mw"]
    over_cap = combined_re > WIND_SOLAR_MAX_MW
    if over_cap.any():
        f = df.loc[over_cap].copy()
        f["flag_reason"] = "wind_solar_exceeds_installed_capacity"
        f["severity"] = SEVERITY_WARNING
        ws_flags.append(f)

    # Large solar at night — NOT auto-corrected, flagged as ERROR
    hour_utc = df.index.hour
    night_mask = (hour_utc >= 21) | (hour_utc <= 4)
    large_solar_night = (df["solar_forecast_mw"] > SOLAR_NIGHT_AUTO_FIX_MAX_MW) & night_mask
    if large_solar_night.any():
        f = df.loc[large_solar_night].copy()
        f["flag_reason"] = (
            f"solar > {SOLAR_NIGHT_AUTO_FIX_MAX_MW} MW at night — too large to auto-fix, "
            "possible data error"
        )
        f["severity"] = SEVERITY_ERROR
        ws_flags.append(f)

    ws_total = sum(len(x) for x in ws_flags)
    all_flagged.extend(ws_flags)
    has_error = any((x["severity"] == SEVERITY_ERROR).any() for x in ws_flags) if ws_flags else False
    results["wind_solar_bounds"] = {
        "passed": ws_total == 0,
        "severity": SEVERITY_ERROR if has_error else (SEVERITY_WARNING if ws_total > 0 else SEVERITY_INFO),
        "details": f"{ws_total} rows flagged for wind/solar bound violations",
        "flagged_rows": pd.concat(ws_flags) if ws_flags else None,
    }

    # --- Check 6: Load bounds ---
    load_col = df["load_forecast_mw"].dropna()
    load_oob = (load_col < LOAD_MIN_MW) | (load_col > LOAD_MAX_MW)
    load_flagged = df.loc[load_col[load_oob].index].copy()
    load_flagged["flag_reason"] = "load_out_of_bounds"
    load_flagged["severity"] = SEVERITY_WARNING

    # Holidays get downgraded to INFO
    holidays_mask = _is_low_load_holiday(load_flagged.index)
    if holidays_mask.any():
        load_flagged.loc[holidays_mask, "flag_reason"] = "load_out_of_bounds (holiday — expected low load)"
        load_flagged.loc[holidays_mask, "severity"] = SEVERITY_INFO

    non_holiday_oob = len(load_flagged) - holidays_mask.sum() if len(load_flagged) > 0 else 0
    if len(load_flagged) > 0:
        all_flagged.append(load_flagged)
    results["load_bounds"] = {
        "passed": non_holiday_oob == 0,
        "severity": SEVERITY_WARNING if non_holiday_oob > 0 else SEVERITY_INFO,
        "details": (
            f"{len(load_flagged)} rows flagged: "
            f"{non_holiday_oob} non-holiday [{SEVERITY_WARNING}], "
            f"{holidays_mask.sum() if len(load_flagged) > 0 else 0} holiday [{SEVERITY_INFO}]"
        ),
        "flagged_rows": load_flagged if len(load_flagged) > 0 else None,
    }

    # --- Check 7: Gas price bounds and coverage ---
    gas_col = df["gas_price_eur_mwh"].dropna()
    gas_oob = (gas_col < GAS_PRICE_MIN) | (gas_col > GAS_PRICE_MAX)
    gas_flagged = df.loc[gas_col[gas_oob].index].copy()
    gas_flagged["flag_reason"] = "gas_price_out_of_bounds"
    gas_flagged["severity"] = SEVERITY_ERROR

    # Stale gas check — more than 72 consecutive identical values
    gas_runs = _find_runs(df["gas_price_eur_mwh"], threshold=72)
    gas_stale_details = ""
    if gas_runs:
        gas_stale_details = f" {len(gas_runs)} stale sequences >72h detected [{SEVERITY_WARNING}]."

    if len(gas_flagged) > 0:
        all_flagged.append(gas_flagged)
    results["gas_bounds"] = {
        "passed": len(gas_flagged) == 0 and not gas_runs,
        "severity": SEVERITY_ERROR if len(gas_flagged) > 0 else (SEVERITY_WARNING if gas_runs else SEVERITY_INFO),
        "details": f"{len(gas_flagged)} rows out of bounds.{gas_stale_details}",
        "flagged_rows": gas_flagged if len(gas_flagged) > 0 else None,
    }

    # --- Check 8: Stale data detection ---
    stale_details = []
    for col in ["da_price_eur_mwh", "wind_forecast_mw", "solar_forecast_mw", "load_forecast_mw"]:
        runs = _find_runs(df[col], threshold=STALE_THRESHOLD, exclude_zero=(col == "solar_forecast_mw"))
        for run_start, run_len, run_val in runs:
            stale_details.append(f"  {col}: {run_len} identical values ({run_val}) starting {run_start}")

    results["stale_data"] = {
        "passed": len(stale_details) == 0,
        "severity": SEVERITY_WARNING if stale_details else SEVERITY_INFO,
        "details": "\n".join(stale_details) if stale_details else "No stale sequences detected",
        "flagged_rows": None,
    }

    # --- Check 9: Cross-series alignment ---
    cross_details = []
    cross_severity = SEVERITY_INFO

    # High-solar hour NaN check (10–14 UTC, Apr–Sep)
    summer_mask = df.index.month.isin([4, 5, 6, 7, 8, 9])
    solar_hours_mask = df.index.hour.isin([10, 11, 12, 13, 14])
    summer_solar = df.loc[summer_mask & solar_hours_mask, "solar_forecast_mw"]
    solar_nan_pct = summer_solar.isna().mean() * 100
    if solar_nan_pct > 50:
        cross_details.append(f"ERROR: Summer solar (10-14 UTC, Apr-Sep) has {solar_nan_pct:.1f}% NaN")
        cross_severity = SEVERITY_ERROR

    # Wind-price correlation
    valid_mask = df["da_price_eur_mwh"].notna() & df["wind_forecast_mw"].notna()
    if valid_mask.sum() > 100:
        corr = df.loc[valid_mask, "da_price_eur_mwh"].corr(df.loc[valid_mask, "wind_forecast_mw"])
        cross_details.append(f"Wind-price correlation: {corr:.3f}")
        if corr > 0.1:
            cross_details.append(f"  WARNING: Positive wind-price correlation (>0.1) — unusual")
            if cross_severity != SEVERITY_ERROR:
                cross_severity = SEVERITY_WARNING
    else:
        corr = float("nan")
        cross_details.append("Wind-price correlation: insufficient data")

    # Gas completeness
    gas_nan = df["gas_price_eur_mwh"].isna().sum()
    cross_details.append(f"Gas completeness: {100 - gas_nan / len(df) * 100:.1f}%")
    if gas_nan > 0:
        cross_details.append(f"  WARNING: {gas_nan} gas NaN values after forward-fill")
        if cross_severity != SEVERITY_ERROR:
            cross_severity = SEVERITY_WARNING

    results["cross_series"] = {
        "passed": solar_nan_pct <= 50 and gas_nan == 0,
        "severity": cross_severity,
        "details": "\n".join(cross_details),
        "flagged_rows": None,
        "wind_price_corr": corr if valid_mask.sum() > 100 else None,
        "gas_nan_count": gas_nan,
    }

    # --- Check 10: DST boundary check ---
    # dst_dates = _get_dst_transition_dates(start_year=2021, end_year=2024)
    dst_dates = _get_dst_transition_dates(start_year=2021, end_year=2026)
    dst_details = []
    dst_ok = True
    for date, direction in dst_dates:
        ts_02 = pd.Timestamp(f"{date} 02:00", tz="UTC")
        if ts_02 in df.index:
            dst_details.append(f"  {date} ({direction}): 02:00 UTC present ✓")
        else:
            dst_details.append(f"  {date} ({direction}): 02:00 UTC MISSING")
            dst_ok = False

    results["dst_boundaries"] = {
        "passed": dst_ok,
        "severity": SEVERITY_ERROR if not dst_ok else SEVERITY_INFO,
        "details": f"{len(dst_dates)} transition dates checked:\n" + "\n".join(dst_details),
        "flagged_rows": None,
    }

    # Consolidate all flagged rows
    if all_flagged:
        non_empty = [f for f in all_flagged if len(f) > 0]
        if non_empty:
            combined = pd.concat(non_empty)
            combined = combined[~combined.index.duplicated(keep="first")]
            results["_all_flagged"] = combined
        else:
            results["_all_flagged"] = pd.DataFrame()
    else:
        results["_all_flagged"] = pd.DataFrame()

    return results


# ---------------------------------------------------------------------------
# Function 9: generate_qa_report
# ---------------------------------------------------------------------------
def generate_qa_report(qa_results: dict, df: pd.DataFrame, output_dir: Path) -> None:
    """Write the QA summary to both stdout and qa_report_summary.txt.

    Now includes severity-aware formatting:
        - INFO items show ✓ (known/expected)
        - WARNING items show ⚠ (needs review)
        - ERROR items show ✗ (must fix)

    Also reports auto-corrections summary from the global corrections log.

    Args:
        qa_results: The dict returned by run_qa_checks().
        df: The final merged DataFrame.
        output_dir: Directory for output files.

    Side effects:
        Writes outputs/qa_report_summary.txt and outputs/qa_flagged_rows.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _severity_icon(severity: str) -> str:
        return {"INFO": "✓", "WARNING": "⚠", "ERROR": "✗"}.get(severity, "?")

    def _check_icon(check: dict) -> str:
        sev = check.get("severity", SEVERITY_INFO)
        if check["passed"]:
            return "✓"
        return _severity_icon(sev)

    sep = "=" * 60
    lines = [
        sep,
        f" DE_LU Power Dataset — Data Quality Report",
        f" Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f" Dataset: {START_DATE} to {END_DATE} UTC",
        sep,
        "",
        "DATASET OVERVIEW",
        f"  Total rows:     {len(df):,} (expected: {35_064:,}) "
        + ("✓" if len(df) == 35_064 else "✗"),
        f"  Total columns:  {df.shape[1]}",
        f"  Date range:     {df.index[0]} → {df.index[-1]}",
        "",
    ]

    # Auto-corrections summary
    lines.append("AUTO-CORRECTIONS APPLIED")
    n_corrections = len(_corrections_log)
    if n_corrections > 0:
        # Summarise by reason
        reasons = {}
        for entry in _corrections_log:
            r = entry["reason"].split("→")[0].strip() if "→" in entry["reason"] else entry["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        lines.append(f"  Total auto-corrections: {n_corrections}")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append(f"    {count:>5d} × {reason}")
        lines.append(f"  Full log: outputs/auto_corrections_log.csv")
    else:
        lines.append("  No auto-corrections needed  ✓")
    lines.append("")

    # Timestamp checks
    lines.append("TIMESTAMP CHECKS")
    tc = qa_results["timestamp_completeness"]
    lines.append(f"  Completeness:   {'PASS' if tc['passed'] else 'FAIL'} — {tc['details']}  {_check_icon(tc)}")
    dt = qa_results["duplicate_timestamps"]
    lines.append(f"  Duplicates:     {'PASS' if dt['passed'] else 'FAIL'} — {dt['details']}  {_check_icon(dt)}")
    dst = qa_results["dst_boundaries"]
    lines.append(f"  DST boundaries: {'PASS' if dst['passed'] else 'FAIL'} — {len(_get_dst_transition_dates(2021, 2026))} transition dates verified  {_check_icon(dst)}")
    lines.append("")

    # Missingness
    lines.append("MISSINGNESS")
    miss = qa_results["missingness"]["data"]
    for col, info in miss.items():
        status = "✓" if info["pct"] <= 1 else ("⚠" if info["pct"] <= 5 else "✗")
        lines.append(f"  {col + ':':30s} {info['total']:>6d} missing ({info['pct']:.2f}%)  {status}")
    lines.append("")

    # Outliers — now with crisis/non-crisis breakdown
    lines.append("OUTLIERS & PHYSICAL BOUNDS")
    da_out = qa_results["da_price_outliers"]
    lines.append(
        f"  DA price (physical bounds):     {da_out['bounds_count']} flagged  "
        + ("✓" if da_out["bounds_count"] == 0 else f"✗ [{SEVERITY_ERROR}]")
    )
    lines.append(
        f"  DA price (4σ, crisis period):   {da_out.get('crisis_stat_count', 0)} flagged  "
        f"✓ [{SEVERITY_INFO} — expected during Jun 2022–Mar 2023]"
    )
    lines.append(
        f"  DA price (4σ, non-crisis):      {da_out.get('non_crisis_stat_count', 0)} flagged  "
        + ("✓" if da_out.get("non_crisis_stat_count", 0) == 0 else f"⚠ [{SEVERITY_WARNING} — investigate]")
    )
    ws = qa_results["wind_solar_bounds"]
    lines.append(
        f"  Wind/solar bound violations:    {ws['details'].split()[0]} flagged  "
        + ("✓" if ws["passed"] else f"{_severity_icon(ws['severity'])} [{ws['severity']}]")
    )
    lb = qa_results["load_bounds"]
    lines.append(
        f"  Load out of bounds:             {lb['details'].split()[0]} flagged  "
        + ("✓" if lb["passed"] else f"{_severity_icon(lb['severity'])} [{lb['severity']}]")
    )
    gb = qa_results["gas_bounds"]
    lines.append(
        f"  Gas price out of bounds:        {gb['details'].split()[0]} flagged  "
        + ("✓" if gb["passed"] else f"{_severity_icon(gb['severity'])} [{gb['severity']}]")
    )
    lines.append("")

    # Stale data
    lines.append("STALE DATA")
    stale = qa_results["stale_data"]
    if stale["passed"]:
        lines.append("  No stale sequences detected  ✓")
    else:
        lines.append(f"  {_severity_icon(stale['severity'])} [{stale['severity']}]")
        lines.append(stale["details"])
    lines.append("")

    # Cross-series
    lines.append("CROSS-SERIES CHECKS")
    cs = qa_results["cross_series"]
    lines.append(f"  {cs['details']}")
    lines.append("")

    # Flagged rows summary with severity breakdown
    all_flagged = qa_results.get("_all_flagged", pd.DataFrame())
    n_flagged = len(all_flagged)
    lines.append("FLAGGED ROWS")
    lines.append(f"  Total flagged: {n_flagged}")

    if n_flagged > 0 and "severity" in all_flagged.columns:
        sev_counts = all_flagged["severity"].value_counts()
        for sev in [SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO]:
            if sev in sev_counts.index:
                lines.append(f"    {_severity_icon(sev)} {sev}: {sev_counts[sev]}")

    flagged_csv = output_dir / "qa_flagged_rows.csv"
    if n_flagged > 0:
        all_flagged.to_csv(flagged_csv)
        lines.append(f"  Saved to: {flagged_csv.relative_to(PROJECT_ROOT)}")
    else:
        lines.append("  No rows flagged")
    lines.append("")

    # Overall status — based on severity counts
    n_errors = sum(
        1 for k, v in qa_results.items()
        if k != "_all_flagged" and isinstance(v, dict) and v.get("severity") == SEVERITY_ERROR
    )
    n_warnings = sum(
        1 for k, v in qa_results.items()
        if k != "_all_flagged" and isinstance(v, dict) and v.get("severity") == SEVERITY_WARNING
    )
    n_info = sum(
        1 for k, v in qa_results.items()
        if k != "_all_flagged" and isinstance(v, dict) and v.get("severity") == SEVERITY_INFO
    )

    if n_errors > 0:
        overall = "FAIL"
    elif n_warnings > 0:
        overall = "PASS WITH WARNINGS"
    else:
        overall = "PASS"

    lines.append(f"OVERALL STATUS: {overall}")
    lines.append(f"  ✗ Errors:   {n_errors}")
    lines.append(f"  ⚠ Warnings: {n_warnings}")
    lines.append(f"  ✓ Info:     {n_info}")
    lines.append(f"  Auto-fixes: {n_corrections}")
    lines.append(sep)

    report_text = "\n".join(lines)

    # Write to file
    report_path = output_dir / "qa_report_summary.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("QA report written to %s", report_path)

    # Print to console
    print(report_text)


# ---------------------------------------------------------------------------
# Function 10: run_ingestion_pipeline
# ---------------------------------------------------------------------------
def run_ingestion_pipeline() -> pd.DataFrame:
    """Master orchestrator — runs the full ingestion and QA pipeline.

    Returns:
        The final cleaned and validated DataFrame.

    Side effects:
        - Fetches data from ENTSO-E API (or loads from cache)
        - Writes cached raw data to data/raw/
        - Writes processed dataset to data/processed/de_power_dataset.parquet
        - Writes QA report to outputs/
        - Logs to both console and logs/ingestion.log
    """
    _setup_logging()
    logger.info("=" * 60)
    logger.info("Starting DE_LU power data ingestion pipeline")
    logger.info("=" * 60)

    try:
        # 0. Clear any corrections from previous runs
        global _corrections_log
        _corrections_log = []

        # 1. Load API key
        api_key = load_api_key()
        logger.info("API key loaded successfully")

        # 2. Initialise ENTSO-E client
        client = EntsoePandasClient(api_key=api_key)

        # 3. Define date range (Europe/Berlin as required by entsoe-py)
        start = pd.Timestamp(START_DATE, tz=TIMEZONE)
        end = pd.Timestamp(f"{END_DATE} 23:00", tz=TIMEZONE)

        # 4. Build authoritative hourly UTC index
        full_index = build_complete_hourly_index(START_DATE, END_DATE)

        # 5. Fetch each data series
        da = fetch_da_prices(client, start, end, DATA_RAW)
        wind_solar = fetch_wind_solar(client, start, end, DATA_RAW)
        load = fetch_load_forecast(client, start, end, DATA_RAW)
        gas = fetch_gas_price(DATA_RAW)

        # 5b. Fetch cross-border flows (optional — graceful fallback)
        try:
            flows = fetch_crossborder_flows(client, start, end, DATA_RAW)
            if not flows.empty:
                logger.info("Cross-border flows fetched: %d rows x %d cols",
                            len(flows), len(flows.columns))
        except Exception as e:
            logger.warning("Cross-border flow fetch failed (non-critical): %s", e)

        # 6. Clean, align, and auto-correct physically impossible values
        df = clean_and_align(da, wind_solar, load, gas, full_index)

        # 7. QA checks (severity-aware)
        qa_results = run_qa_checks(df, full_index)

        # 8. Save corrections log (before QA report so report can reference it)
        save_corrections_log(OUTPUTS_DIR)

        # 9. Generate QA report
        generate_qa_report(qa_results, df, OUTPUTS_DIR)

        # 10. Save final dataset
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        output_path = DATA_PROCESSED / "de_power_dataset.parquet"
        df.to_parquet(output_path)
        logger.info("Pipeline complete. Dataset saved to %s", output_path)

        return df

    except Exception:
        logger.exception("Pipeline failed with unrecoverable error")
        raise


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _ensure_hourly(series: pd.Series, label: str) -> pd.Series:
    """Resample a Series to hourly frequency if sub-hourly data is detected.

    Args:
        series: A time-indexed pd.Series.
        label: Human-readable label for log messages.

    Returns:
        Hourly-resampled Series (mean) if sub-hourly, otherwise unchanged.
    """
    if len(series) < 2:
        return series
    deltas = series.index.to_series().diff().dropna()
    median_delta = deltas.median()
    if median_delta < pd.Timedelta(hours=1):
        logger.warning(
            "%s: sub-hourly resolution detected (median delta=%s). Resampling to hourly mean.",
            label,
            median_delta,
        )
        series = series.resample("h").mean()
    return series


def _ensure_hourly_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Resample a DataFrame to hourly frequency if sub-hourly data is detected.

    Args:
        df: A time-indexed pd.DataFrame.
        label: Human-readable label for log messages.

    Returns:
        Hourly-resampled DataFrame (mean) if sub-hourly, otherwise unchanged.
    """
    if len(df) < 2:
        return df
    deltas = df.index.to_series().diff().dropna()
    median_delta = deltas.median()
    if median_delta < pd.Timedelta(hours=1):
        logger.warning(
            "%s: sub-hourly resolution detected (median delta=%s). Resampling to hourly mean.",
            label,
            median_delta,
        )
        df = df.resample("h").mean()
    return df


def _find_runs(
    series: pd.Series,
    threshold: int = STALE_THRESHOLD,
    exclude_zero: bool = False,
) -> list[tuple[pd.Timestamp, int, float]]:
    """Find runs of consecutive identical values in a Series.

    Args:
        series: A time-indexed pd.Series.
        threshold: Minimum run length to report.
        exclude_zero: If True, skip runs of zero values (e.g. solar at night).

    Returns:
        List of (start_timestamp, run_length, value) tuples for runs >= threshold.
    """
    s = series.dropna()
    if len(s) == 0:
        return []

    groups = s.ne(s.shift()).cumsum()
    results = []
    for _, grp in s.groupby(groups):
        if len(grp) >= threshold:
            val = grp.iloc[0]
            if exclude_zero and val == 0:
                continue
            results.append((grp.index[0], len(grp), val))
    return results


def _is_low_load_holiday(index: pd.DatetimeIndex) -> pd.Series:
    """Check if dates fall on known low-load holidays (Christmas, Easter Sunday).

    Args:
        index: A DatetimeIndex to check.

    Returns:
        A boolean Series indicating holiday dates.
    """
    holidays = set()
    for year in range(2021, 2027):
        holidays.add(pd.Timestamp(f"{year}-12-25", tz="UTC").date())
        holidays.add(pd.Timestamp(f"{year}-12-26", tz="UTC").date())
        # Easter Sundays (hardcoded for 2021-2024)
        easter_dates = {
            2021: "2021-04-04",
            2022: "2022-04-17",
            2023: "2023-04-09",
            2024: "2024-03-31",
            2025: "2025-04-20",
            2026: "2026-04-05",

        }
        holidays.add(pd.Timestamp(easter_dates[year]).date())

    dates = index.date if hasattr(index, "date") else pd.DatetimeIndex(index).date
    return pd.Series([d in holidays for d in dates], index=index)


def _get_dst_transition_dates(
    start_year: int, end_year: int
) -> list[tuple[str, str]]:
    """Get DST transition dates for Europe/Berlin.

    Args:
        start_year: First year to check.
        end_year: Last year to check.

    Returns:
        List of (date_string, direction) tuples where direction is
        'spring_forward' or 'fall_back'.
    """
    import pytz

    tz = pytz.timezone("Europe/Berlin")
    transitions = []
    for year in range(start_year, end_year + 1):
        # Last Sunday of March (spring forward)
        march_end = pd.Timestamp(f"{year}-03-31")
        while march_end.weekday() != 6:  # Sunday = 6
            march_end -= pd.Timedelta(days=1)
        transitions.append((march_end.strftime("%Y-%m-%d"), "spring_forward"))

        # Last Sunday of October (fall back)
        oct_end = pd.Timestamp(f"{year}-10-31")
        while oct_end.weekday() != 6:
            oct_end -= pd.Timedelta(days=1)
        transitions.append((oct_end.strftime("%Y-%m-%d"), "fall_back"))

    return transitions
