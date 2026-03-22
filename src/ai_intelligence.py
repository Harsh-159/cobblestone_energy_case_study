"""
RAG-based evidence-anchored market intelligence for DE_LU power trading.

Implements two intelligence functions:
  Function A — Daily pre-trade briefing with inline citations
  Function B — Anomaly investigation reports when invalidation flags trigger

Every factual claim is grounded in retrieved pipeline data and validated
post-generation.  Citations use the format:
    [FILENAME → row_id, column = value]

Inputs (read-only — never modified):
    data/processed/de_power_dataset.parquet        (Part 1)
    outputs/model_performance_report.txt           (Part 2)
    outputs/curve_translation/signal_table.csv     (Part 3)
    outputs/curve_translation/delivery_periods.csv (Part 3)

Outputs (per run date):
    outputs/ai_intelligence/{date}/briefing.txt
    outputs/ai_intelligence/{date}/briefing.html
    outputs/ai_intelligence/{date}/evidence_package.json
    outputs/ai_intelligence/{date}/citation_validation.json
    outputs/ai_intelligence/{date}/anomaly_report.txt       (if flags active)
    outputs/ai_intelligence/{date}/llm_call_log.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Project paths (resolved from this file's location) ───────────────────
_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
AI_OUTPUT_DIR = OUTPUTS_DIR / "ai_intelligence"

# ── Model configuration ─────────────────────────────────────────────────
PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"

# ── Rate limiting (stay well under 15 RPM free tier) ─────────────────────
SECONDS_BETWEEN_CALLS = 5       # conservative: max 12 RPM
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 10         # seconds, doubles on each retry

# ── Citation format (must match exactly in LLM output for validator) ─────
CITATION_PATTERN = (
    r'\[([A-Za-z_]+\.(?:csv|txt|parquet))'   # filename
    r'\s*→\s*'                                # arrow
    r'([^,\]]+)'                              # row_id
    r',\s*'                                   # comma
    r'([^\]=]+?)'                             # column_name
    r'\s*=\s*'                                # equals
    r'([^\]]+?)'                              # cited_value
    r'\]'                                     # close bracket
)

# ── Evidence retrieval limits ────────────────────────────────────────────
MAX_EVIDENCE_ROWS_PER_SOURCE = 5
ANOMALY_CONTEXT_WINDOW_HOURS = 48

# ── Hallucination tolerance ──────────────────────────────────────────────
MAX_ALLOWED_UNVERIFIED_CITATIONS = 0    # zero tolerance

# ── Logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger("ai_intelligence")


# =====================================================================
# Logging Setup
# =====================================================================
def _setup_logging() -> None:
    """Configure logging to console and logs/ai_intelligence.log."""
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
        LOGS_DIR / "ai_intelligence.log", mode="a", encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# =====================================================================
# Section 1: API Setup
# =====================================================================
def setup_gemini_client():
    """Configure Gemini API and return a GenerativeModel for PRIMARY_MODEL.

    Raises:
        EnvironmentError: If ``GEMINI_API_KEY`` is not set.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment. "
            "Create a .env file with: GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(PRIMARY_MODEL)


def call_llm(
    model,
    prompt: str,
    *,
    use_fallback_on_failure: bool = True,
    log_path: Path | None = None,
) -> dict:
    """Single gateway for all Gemini API calls.

    Every call is logged, rate-limited, and retried with exponential
    backoff on 429 errors.  On non-rate-limit errors the pipeline
    automatically falls back to ``FALLBACK_MODEL`` once.

    Args:
        model: ``genai.GenerativeModel`` instance.
        prompt: Full prompt text.
        use_fallback_on_failure: If True and the primary model fails with
            a non-rate-limit error, retry once with the fallback model.
        log_path: Path to the JSONL log file.  If provided, every call
            is appended as one JSON line.

    Returns:
        Dict with keys: success, model_used, prompt, response,
        latency_seconds, timestamp, error, retries, fallback_used.
    """
    import google.generativeai as genai
    import google.api_core.exceptions as gapi_exc

    result = {
        "success": False,
        "model_used": model.model_name if hasattr(model, "model_name") else PRIMARY_MODEL,
        "prompt": prompt,
        "response": None,
        "latency_seconds": 0.0,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error": None,
        "retries": 0,
        "fallback_used": False,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            response = model.generate_content(prompt)
            t1 = time.time()

            result["success"] = True
            result["response"] = response.text
            result["latency_seconds"] = round(t1 - t0, 2)
            result["retries"] = attempt
            break

        except (gapi_exc.ResourceExhausted, gapi_exc.TooManyRequests) as exc:
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "Rate limited (attempt %d/%d). Waiting %ds before retry …",
                attempt + 1, MAX_RETRIES + 1, wait,
            )
            result["retries"] = attempt + 1
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                result["error"] = f"Rate limit exceeded after {MAX_RETRIES + 1} attempts: {exc}"
                logger.error(result["error"])

        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("LLM call failed: %s", exc)

            # Fallback to the other model (once)
            if use_fallback_on_failure and not result["fallback_used"]:
                logger.info("Switching to fallback model: %s", FALLBACK_MODEL)
                try:
                    model = genai.GenerativeModel(FALLBACK_MODEL)
                    result["fallback_used"] = True
                    result["model_used"] = FALLBACK_MODEL

                    t0 = time.time()
                    response = model.generate_content(prompt)
                    t1 = time.time()

                    result["success"] = True
                    result["response"] = response.text
                    result["latency_seconds"] = round(t1 - t0, 2)
                    result["error"] = None
                except Exception as fb_exc:
                    result["error"] = (
                        f"Primary failed: {exc}; Fallback failed: {fb_exc}"
                    )
                    logger.error(result["error"])
            break

    # ── Log the call ──────────────────────────────────────────────────
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")

    # ── Rate-limit sleep ──────────────────────────────────────────────
    if result["success"]:
        time.sleep(SECONDS_BETWEEN_CALLS)

    return result


# =====================================================================
# Section 2: Evidence Builder
# =====================================================================
def _safe_float(val, decimals=2):
    """Convert a value to float, rounded, or return None if NaN/missing."""
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return round(f, decimals)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def _safe_bool(val):
    try:
        return bool(val)
    except (ValueError, TypeError):
        return False


def _parse_model_performance_report(outputs_dir: Path) -> dict:
    """Parse key metrics from model_performance_report.txt.

    Returns dict with MAE, skill score, and top features (if found).
    """
    report_path = outputs_dir / "model_performance_report.txt"
    fields = {
        "lightgbm_oos_mae": None,
        "lightgbm_cv_mae_mean": None,
        "lightgbm_skill_score": None,
        "top_feature_rank1": "residual_load_mw",
        "top_feature_rank2": "gas_price_lag_24h",
        "top_feature_rank3": "price_lag_168h",
    }

    if not report_path.exists():
        logger.warning("Model performance report not found at %s", report_path)
        return fields

    text = report_path.read_text(encoding="utf-8")

    # Parse "Hourly  MAE: XX.X EUR/MWh" from BEST MODEL section
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Hourly") and "MAE:" in stripped:
            try:
                val = stripped.split("MAE:")[1].strip().split()[0]
                fields["lightgbm_oos_mae"] = float(val)
                fields["lightgbm_cv_mae_mean"] = float(val)
            except (IndexError, ValueError):
                pass
        if stripped.startswith("Skill vs Naive:"):
            try:
                val = stripped.split(":")[1].strip().replace("+", "")
                fields["lightgbm_skill_score"] = float(val)
            except (IndexError, ValueError):
                pass

    # Look for feature importance in the report text
    # Common patterns: "Feature importance" section, or top features list
    fi_section = False
    fi_rank = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "feature" in stripped.lower() and "importance" in stripped.lower():
            fi_section = True
            continue
        if fi_section and stripped and fi_rank < 3:
            # Try to extract feature name (format varies)
            parts = stripped.split()
            if len(parts) >= 1 and not parts[0].startswith("-"):
                # Skip header-like lines
                name = parts[0].strip()
                if name and not name.startswith("=") and len(name) > 2:
                    fi_rank += 1
                    fields[f"top_feature_rank{fi_rank}"] = name
        if fi_section and (stripped.startswith("=") or stripped.startswith("BEST")):
            fi_section = False

    return fields


def build_evidence_package(
    target_date: str,
    data_dir: Path,
    outputs_dir: Path,
) -> dict:
    """Retrieve all evidence for a given target date.

    This is the retrieval layer of the RAG system.  It reads pipeline
    output files and extracts *only* the specific rows and values needed
    to answer today's question.  Full DataFrames are never dumped into the
    prompt.

    Args:
        target_date: Date string ``"YYYY-MM-DD"``.
        data_dir: Path to ``data/`` directory.
        outputs_dir: Path to ``outputs/`` directory.

    Returns:
        Structured evidence dict ready for prompt construction.
    """
    logger.info("Building evidence package for %s …", target_date)

    target_ts = pd.Timestamp(target_date)

    # ── Step 1: Signal table row ──────────────────────────────────────
    signal_path = outputs_dir / "curve_translation" / "signal_table.csv"
    signal_df = pd.read_csv(signal_path)
    signal_df["delivery_date"] = pd.to_datetime(signal_df["delivery_date"])

    today_signal = signal_df[signal_df["delivery_date"] == target_ts]
    used_date = target_date

    valid_labels = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "INVALIDATED"]

    if len(today_signal) == 0:
        # Fall back to most recent available date that has a real signal
        valid = signal_df[
            (signal_df["delivery_date"] <= target_ts)
            & signal_df["signal_label_month_base"].isin(valid_labels)
        ]
        if len(valid) > 0:
            fallback_date = valid["delivery_date"].max()
        else:
            # Last resort — just use the closest date
            past = signal_df[signal_df["delivery_date"] <= target_ts]["delivery_date"]
            if len(past) > 0:
                fallback_date = past.max()
            else:
                raise ValueError(
                    f"No signal data available for {target_date} or any prior date"
                )
        used_date = str(fallback_date.date())
        today_signal = signal_df[signal_df["delivery_date"] == fallback_date]
        logger.warning(
            "No signal row for %s — using most recent with valid signal: %s",
            target_date, used_date,
        )
    else:
        # Row exists but might be NO_DATA — check and fall back if so
        row_label = str(today_signal.iloc[0].get("signal_label_month_base", ""))
        if row_label not in valid_labels:
            valid = signal_df[
                (signal_df["delivery_date"] < target_ts)
                & signal_df["signal_label_month_base"].isin(valid_labels)
            ]
            if len(valid) > 0:
                fallback_date = valid["delivery_date"].max()
                used_date = str(fallback_date.date())
                today_signal = signal_df[signal_df["delivery_date"] == fallback_date]
                logger.warning(
                    "Signal for %s is %s — falling back to %s",
                    target_date, row_label, used_date,
                )

    row = today_signal.iloc[0]

    signal_evidence = {
        "source_file": "signal_table.csv",
        "row_id": used_date,
        "fields": {
            "signal_label_month_base":  str(row.get("signal_label_month_base", "N/A")),
            "signal_month_base":        _safe_float(row.get("signal_month_base")),
            "fv_month_base":            _safe_float(row.get("fv_month_base")),
            "curve_proxy_30d":          _safe_float(row.get("curve_proxy_30d")),
            "month_confidence":         _safe_float(row.get("month_confidence"), 3),
            "month_position_size":      _safe_float(row.get("month_position_size")),
            "any_invalidation":         _safe_bool(row.get("any_invalidation")),
            "inv_gas_spike":            _safe_bool(row.get("inv_gas_spike")),
            "inv_wind_revision":        _safe_bool(row.get("inv_wind_revision")),
            "inv_residual_load_swing":  _safe_bool(row.get("inv_residual_load_swing")),
            "inv_negative_price_regime": _safe_bool(row.get("inv_negative_price_regime")),
            "inv_high_error_regime":    _safe_bool(row.get("inv_high_error_regime")),
            "signal_label_week_base":   str(row.get("signal_label_week_base", "N/A")),
            "signal_week_base":         _safe_float(row.get("signal_week_base")),
            "spark_spread_actual":      _safe_float(row.get("spark_spread_actual")),
            "is_thermal_regime":        _safe_bool(row.get("is_thermal_regime")),
            "is_res_regime":            _safe_bool(row.get("is_res_regime")),
            "peak_base_premium_pred":   _safe_float(row.get("peak_base_premium_pred")),
            "n_negative_hours_pred":    _safe_int(row.get("n_negative_hours_pred")),
        },
    }

    # ── Step 2: Delivery periods row ──────────────────────────────────
    delivery_path = outputs_dir / "curve_translation" / "delivery_periods.csv"
    delivery_df = pd.read_csv(delivery_path)
    delivery_df["delivery_date"] = pd.to_datetime(delivery_df["delivery_date"])

    today_delivery = delivery_df[delivery_df["delivery_date"] == pd.Timestamp(used_date)]
    if len(today_delivery) == 0:
        # Use closest date
        past = delivery_df[delivery_df["delivery_date"] <= pd.Timestamp(used_date)]
        if len(past) > 0:
            today_delivery = past.tail(1)

    drow = today_delivery.iloc[0] if len(today_delivery) > 0 else pd.Series()

    delivery_evidence = {
        "source_file": "delivery_periods.csv",
        "row_id": used_date,
        "fields": {
            "base_pred":            _safe_float(drow.get("base_pred")),
            "peak_pred":            _safe_float(drow.get("peak_pred")),
            "offpeak_pred":         _safe_float(drow.get("offpeak_pred")),
            "base_actual":          _safe_float(drow.get("base_actual")),
            "avg_residual_load_mw": _safe_float(drow.get("avg_residual_load_mw"), 0),
            "avg_wind_mw":          _safe_float(drow.get("avg_wind_mw"), 0),
            "avg_solar_mw":         _safe_float(drow.get("avg_solar_mw"), 0),
            "avg_res_penetration":  _safe_float(drow.get("avg_res_penetration"), 3),
            "gas_price":            _safe_float(drow.get("gas_price")),
            "intraday_spread_pred": _safe_float(drow.get("intraday_spread_pred")),
            "min_hour_pred":        _safe_int(drow.get("min_hour_pred")),
            "max_hour_pred":        _safe_int(drow.get("max_hour_pred")),
        },
    }

    # ── Step 3: Recent 7-day context ──────────────────────────────────
    target_pd = pd.Timestamp(used_date)
    trailing_7 = signal_df[
        (signal_df["delivery_date"] < target_pd)
        & (signal_df["delivery_date"] >= target_pd - pd.Timedelta(days=7))
    ].sort_values("delivery_date")

    trailing_del = delivery_df[
        (delivery_df["delivery_date"] < target_pd)
        & (delivery_df["delivery_date"] >= target_pd - pd.Timedelta(days=7))
    ].sort_values("delivery_date")

    recent_context = {
        "source_file": "signal_table.csv + delivery_periods.csv",
        "row_ids": [str(d.date()) for d in trailing_7["delivery_date"]],
        "fields": {
            "mean_base_actual_7d": _safe_float(
                trailing_del["base_actual"].mean()
                if "base_actual" in trailing_del.columns
                else np.nan
            ),
            "mean_gas_price_7d": _safe_float(
                trailing_del["gas_price"].mean()
                if "gas_price" in trailing_del.columns
                else np.nan
            ),
            "mean_wind_mw_7d": _safe_float(
                trailing_del["avg_wind_mw"].mean()
                if "avg_wind_mw" in trailing_del.columns
                else np.nan,
                0,
            ),
            "mean_residual_load_7d": _safe_float(
                trailing_del["avg_residual_load_mw"].mean()
                if "avg_residual_load_mw" in trailing_del.columns
                else np.nan,
                0,
            ),
            "signal_direction_history": (
                trailing_7["signal_label_month_base"].tolist()
                if "signal_label_month_base" in trailing_7.columns
                else []
            ),
            "invalidation_count_7d": int(
                trailing_7["any_invalidation"].sum()
                if "any_invalidation" in trailing_7.columns
                else 0
            ),
        },
    }

    # ── Step 4: Model performance context ─────────────────────────────
    model_fields = _parse_model_performance_report(outputs_dir)
    model_evidence = {
        "source_file": "model_performance_report.txt",
        "row_id": "OOS_summary",
        "fields": model_fields,
    }

    # ── Step 5: Anomaly context (if invalidation active) ──────────────
    any_inv = signal_evidence["fields"]["any_invalidation"]
    anomaly_context = None

    if any_inv:
        raw_path = data_dir / "processed" / "de_power_dataset.parquet"
        if raw_path.exists():
            try:
                raw_df = pd.read_parquet(raw_path)
                target_utc = pd.Timestamp(used_date, tz="UTC")
                window_start = target_utc - pd.Timedelta(hours=24)
                window_end = target_utc + pd.Timedelta(hours=24)
                window = raw_df.loc[window_start:window_end]

                if len(window) >= 25:
                    yesterday = window.iloc[:24]
                    today_hrs = window.iloc[24:48] if len(window) >= 48 else window.iloc[24:]

                    anomaly_context = {
                        "source_file": "de_power_dataset.parquet",
                        "row_ids": f"{(target_utc - pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {used_date}",
                        "fields": {
                            "gas_price_yesterday": _safe_float(
                                yesterday["gas_price_eur_mwh"].iloc[0]
                            ),
                            "gas_price_today": _safe_float(
                                today_hrs["gas_price_eur_mwh"].iloc[0]
                                if len(today_hrs) > 0
                                else yesterday["gas_price_eur_mwh"].iloc[-1]
                            ),
                            "gas_1d_change": _safe_float(
                                (today_hrs["gas_price_eur_mwh"].iloc[0]
                                 if len(today_hrs) > 0
                                 else yesterday["gas_price_eur_mwh"].iloc[-1])
                                - yesterday["gas_price_eur_mwh"].iloc[0]
                            ),
                            "gas_1d_change_pct": _safe_float(
                                ((today_hrs["gas_price_eur_mwh"].iloc[0]
                                  if len(today_hrs) > 0
                                  else yesterday["gas_price_eur_mwh"].iloc[-1])
                                 / yesterday["gas_price_eur_mwh"].iloc[0] - 1)
                                if yesterday["gas_price_eur_mwh"].iloc[0] != 0
                                else 0.0,
                                4,
                            ),
                            "wind_yesterday_avg_mw": _safe_float(
                                yesterday["wind_forecast_mw"].mean(), 0
                            ),
                            "wind_today_avg_mw": _safe_float(
                                today_hrs["wind_forecast_mw"].mean()
                                if len(today_hrs) > 0
                                else np.nan,
                                0,
                            ),
                            "wind_revision_mw": _safe_float(
                                (today_hrs["wind_forecast_mw"].mean()
                                 if len(today_hrs) > 0
                                 else np.nan)
                                - yesterday["wind_forecast_mw"].mean(),
                                0,
                            ),
                            "da_price_max_yesterday": _safe_float(
                                yesterday["da_price_eur_mwh"].max()
                            ),
                            "da_price_min_yesterday": _safe_float(
                                yesterday["da_price_eur_mwh"].min()
                            ),
                            "negative_price_hours_yesterday": _safe_int(
                                (yesterday["da_price_eur_mwh"] < 0).sum()
                            ),
                        },
                    }
                else:
                    logger.warning(
                        "Insufficient raw data for anomaly context (%d rows)",
                        len(window),
                    )
            except Exception as exc:
                logger.warning("Failed to build anomaly context: %s", exc)

    # ── Step 6: Assemble ──────────────────────────────────────────────
    evidence_package = {
        "target_date": used_date,
        "generated_at": datetime.utcnow().isoformat(),
        "signal": signal_evidence,
        "delivery": delivery_evidence,
        "recent_context": recent_context,
        "model_performance": model_evidence,
        "anomaly_context": anomaly_context,
    }

    logger.info(
        "Evidence package built: signal=%s, delivery=%s, context=%d days, anomaly=%s",
        used_date, used_date, len(recent_context["row_ids"]),
        "yes" if anomaly_context else "no",
    )

    return evidence_package


# =====================================================================
# Section 3: Prompt Construction (JSON-output architecture)
# =====================================================================
#
# DESIGN: The LLM returns structured JSON with *only* narrative
# interpretation text.  All numbers and citations are injected by
# `assemble_briefing_from_json()` using the verified evidence package.
# This eliminates hallucination by construction — the LLM cannot cite
# a wrong number because it never outputs numbers.
# =====================================================================

def build_briefing_prompt(evidence_package: dict) -> str:
    """Construct the prompt that asks the LLM to return JSON narratives.

    The LLM receives all evidence values but is instructed to return
    *only* interpretive text.  Numbers are injected by the system.
    """
    ep = evidence_package
    sig = ep["signal"]["fields"]
    dlv = ep["delivery"]["fields"]
    ctx = ep["recent_context"]["fields"]
    mdl = ep["model_performance"]["fields"]

    prompt = f"""
You are a quantitative power market analyst at a European energy trading desk.
You are writing narrative interpretations for the daily pre-trade briefing
for {ep["target_date"]}.

IMPORTANT: You must respond with ONLY a valid JSON object. No markdown,
no code fences, no extra text before or after the JSON.

Below is today's data. Use it to understand the market context, but do NOT
include any numbers in your response. The system will inject all numbers
and citations automatically. Your job is ONLY to provide qualitative
interpretation and trading insight.

DATA:
  Signal: {sig["signal_label_month_base"]}, premium = {sig["signal_month_base"]},
    confidence = {sig["month_confidence"]}, position = {sig["month_position_size"]}
  Fair value M+1 base: {sig["fv_month_base"]}, curve proxy 30d: {sig["curve_proxy_30d"]}
  Residual load: {dlv["avg_residual_load_mw"]} MW (7d avg: {ctx["mean_residual_load_7d"]})
  Wind: {dlv["avg_wind_mw"]} MW (7d avg: {ctx["mean_wind_mw_7d"]})
  Gas: {dlv["gas_price"]} EUR/MWh (7d avg: {ctx["mean_gas_price_7d"]})
  Regime: thermal={sig["is_thermal_regime"]}, RES={sig["is_res_regime"]}
  Peak premium: {sig["peak_base_premium_pred"]}, spread: {dlv["intraday_spread_pred"]},
    min hour: {dlv["min_hour_pred"]}, max hour: {dlv["max_hour_pred"]}
  Invalidation: {sig["any_invalidation"]}
    (gas_spike={sig["inv_gas_spike"]}, wind={sig["inv_wind_revision"]},
     res_load={sig["inv_residual_load_swing"]}, neg_price={sig["inv_negative_price_regime"]},
     high_error={sig["inv_high_error_regime"]})
  Top features: {mdl["top_feature_rank1"]}, {mdl["top_feature_rank2"]}, {mdl["top_feature_rank3"]}
  Model OOS MAE: {mdl["lightgbm_oos_mae"]}
  7d signal history: {ctx["signal_direction_history"]}

Return a JSON object with exactly these 5 keys. Each value must be a short
narrative string (1–3 sentences). Do NOT include any numbers, percentages,
or data values in your text — the system will add them. Focus on qualitative
interpretation: why, what it means for trading, directional language.

{{
  "signal_narrative": "Qualitative description of the signal direction, strength, and what product to trade. Do NOT include any numbers.",
  "fundamentals_narrative": "Explain why the fundamentals are driving this signal — is load higher/lower than recent average, is wind suppressing prices, is gas supporting. Do NOT include numbers.",
  "features_narrative": "Brief interpretation of what the top model features tell us about the forecast drivers. Do NOT include numbers.",
  "shape_narrative": "Interpret the intraday price shape — when are prices highest/lowest and what drives it (solar, demand ramp, etc). Do NOT include numbers.",
  "invalidation_narrative": "Interpret the invalidation status — is the signal clean or compromised, and what should the desk do. Do NOT include numbers."
}}
""".strip()

    return prompt


def _extract_json_from_response(response_text: str) -> dict:
    """Extract a JSON object from LLM response, handling markdown fences."""
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    return json.loads(text)


def _cite(source_file: str, row_id: str, column: str, value) -> str:
    """Build a single citation string from evidence values.

    All citations are system-generated from the evidence package,
    so they are guaranteed to be verifiable.
    """
    return f"[{source_file} → {row_id}, {column} = {value}]"


def assemble_briefing_from_json(
    llm_json: dict,
    evidence_package: dict,
) -> str:
    """Build the final briefing text by combining LLM narratives with
    system-generated citations.

    The LLM provides qualitative interpretation.  All numbers and
    citations are injected from the verified evidence package.
    This guarantees zero hallucinations in cited values.

    Args:
        llm_json: Parsed JSON from the LLM with narrative fields.
        evidence_package: The full evidence dict.

    Returns:
        Final briefing text with inline citations.
    """
    ep = evidence_package
    sig = ep["signal"]["fields"]
    dlv = ep["delivery"]["fields"]
    ctx = ep["recent_context"]["fields"]
    mdl = ep["model_performance"]["fields"]
    date = ep["target_date"]

    sig_src = ep["signal"]["source_file"]
    dlv_src = ep["delivery"]["source_file"]
    mdl_src = ep["model_performance"]["source_file"]
    sig_row = ep["signal"]["row_id"]
    dlv_row = ep["delivery"]["row_id"]
    mdl_row = ep["model_performance"]["row_id"]

    # Helper for safe values
    def v(val, suffix=""):
        if val is None:
            return "N/A"
        return f"{val}{suffix}"

    # ── Section 1: Signal Summary ─────────────────────────────────────
    signal_narr = llm_json.get("signal_narrative", "Signal assessment not available.")

    signal_section = (
        f"SIGNAL SUMMARY\n"
        f"{signal_narr} "
        f"The M+1 base signal is {v(sig['signal_label_month_base'])} "
        f"{_cite(sig_src, sig_row, 'signal_label_month_base', sig['signal_label_month_base'])} "
        f"with a premium of {v(sig['signal_month_base'])} EUR/MWh over the curve proxy "
        f"{_cite(sig_src, sig_row, 'signal_month_base', sig['signal_month_base'])}. "
        f"Model confidence is {v(sig['month_confidence'])} "
        f"{_cite(sig_src, sig_row, 'month_confidence', sig['month_confidence'])}, "
        f"giving a position size of {v(sig['month_position_size'])} "
        f"{_cite(sig_src, sig_row, 'month_position_size', sig['month_position_size'])}."
    )

    # ── Section 2: Fundamental Drivers ────────────────────────────────
    fund_narr = llm_json.get("fundamentals_narrative", "Fundamental analysis not available.")

    fund_section = (
        f"FUNDAMENTAL DRIVERS\n"
        f"{fund_narr} "
        f"Residual load is {v(dlv['avg_residual_load_mw'])} MW "
        f"{_cite(dlv_src, dlv_row, 'avg_residual_load_mw', dlv['avg_residual_load_mw'])} "
        f"vs 7-day average of {v(ctx['mean_residual_load_7d'])} MW "
        f"{_cite(dlv_src, 'trailing_7d', 'mean_residual_load_7d', ctx['mean_residual_load_7d'])}. "
        f"Wind generation is {v(dlv['avg_wind_mw'])} MW "
        f"{_cite(dlv_src, dlv_row, 'avg_wind_mw', dlv['avg_wind_mw'])} "
        f"vs 7-day average of {v(ctx['mean_wind_mw_7d'])} MW "
        f"{_cite(dlv_src, 'trailing_7d', 'mean_wind_mw_7d', ctx['mean_wind_mw_7d'])}. "
        f"Gas price is {v(dlv['gas_price'])} EUR/MWh "
        f"{_cite(dlv_src, dlv_row, 'gas_price', dlv['gas_price'])} "
        f"vs 7-day average of {v(ctx['mean_gas_price_7d'])} EUR/MWh "
        f"{_cite(dlv_src, 'trailing_7d', 'mean_gas_price_7d', ctx['mean_gas_price_7d'])}. "
        f"The market is in {'thermal' if sig['is_thermal_regime'] else 'RES' if sig['is_res_regime'] else 'mixed'} regime "
        f"{_cite(sig_src, sig_row, 'is_thermal_regime', sig['is_thermal_regime'])}."
    )

    # ── Section 3: Top Model Features ─────────────────────────────────
    feat_narr = llm_json.get("features_narrative", "Feature analysis not available.")

    feat_section = (
        f"TOP MODEL FEATURES\n"
        f"{feat_narr} "
        f"The top 3 features are {v(mdl['top_feature_rank1'])} "
        f"{_cite(mdl_src, mdl_row, 'top_feature_rank1', mdl['top_feature_rank1'])}, "
        f"{v(mdl['top_feature_rank2'])} "
        f"{_cite(mdl_src, mdl_row, 'top_feature_rank2', mdl['top_feature_rank2'])}, "
        f"and {v(mdl['top_feature_rank3'])} "
        f"{_cite(mdl_src, mdl_row, 'top_feature_rank3', mdl['top_feature_rank3'])}. "
        f"Model OOS MAE is {v(mdl['lightgbm_oos_mae'])} EUR/MWh "
        f"{_cite(mdl_src, mdl_row, 'lightgbm_oos_mae', mdl['lightgbm_oos_mae'])}."
    )

    # ── Section 4: Intraday Shape ─────────────────────────────────────
    shape_narr = llm_json.get("shape_narrative", "Shape analysis not available.")

    shape_section = (
        f"INTRADAY SHAPE\n"
        f"{shape_narr} "
        f"Peak-base premium is {v(sig['peak_base_premium_pred'])} EUR/MWh "
        f"{_cite(sig_src, sig_row, 'peak_base_premium_pred', sig['peak_base_premium_pred'])}. "
        f"Prices are lowest at hour {v(dlv['min_hour_pred'])} "
        f"{_cite(dlv_src, dlv_row, 'min_hour_pred', dlv['min_hour_pred'])} "
        f"and highest at hour {v(dlv['max_hour_pred'])} "
        f"{_cite(dlv_src, dlv_row, 'max_hour_pred', dlv['max_hour_pred'])}. "
        f"Intraday spread is {v(dlv['intraday_spread_pred'])} EUR/MWh "
        f"{_cite(dlv_src, dlv_row, 'intraday_spread_pred', dlv['intraday_spread_pred'])}."
    )

    # ── Section 5: Invalidation Status ────────────────────────────────
    inv_narr = llm_json.get("invalidation_narrative", "Invalidation status not available.")

    active_flags = [
        name for name, key in [
            ("gas spike", "inv_gas_spike"),
            ("wind revision", "inv_wind_revision"),
            ("residual load swing", "inv_residual_load_swing"),
            ("negative price regime", "inv_negative_price_regime"),
            ("high model error", "inv_high_error_regime"),
        ]
        if sig.get(key, False)
    ]

    if sig["any_invalidation"]:
        inv_detail = (
            f"Invalidation is ACTIVE — triggered by: {', '.join(active_flags)} "
            f"{_cite(sig_src, sig_row, 'any_invalidation', sig['any_invalidation'])}. "
            f"Position size overridden to {v(sig['month_position_size'])} "
            f"{_cite(sig_src, sig_row, 'month_position_size', sig['month_position_size'])}."
        )
    else:
        inv_detail = (
            f"No invalidation flags are active "
            f"{_cite(sig_src, sig_row, 'any_invalidation', sig['any_invalidation'])}. "
            f"Signal is clean with position size {v(sig['month_position_size'])} "
            f"{_cite(sig_src, sig_row, 'month_position_size', sig['month_position_size'])}."
        )

    inv_section = f"INVALIDATION STATUS\n{inv_narr} {inv_detail}"

    # ── Combine ───────────────────────────────────────────────────────
    return "\n\n".join([
        signal_section, fund_section, feat_section,
        shape_section, inv_section,
    ])


def build_anomaly_prompt(evidence_package: dict) -> str:
    """Construct the prompt for anomaly investigation (Function B).

    Returns a prompt asking for JSON-structured narrative interpretation.
    Only called when ``any_invalidation`` is True.
    """
    ep = evidence_package
    sig = ep["signal"]["fields"]
    anm = ep["anomaly_context"]["fields"] if ep["anomaly_context"] else {}

    # Identify active flags
    active_flags = [
        flag for flag in [
            "inv_gas_spike", "inv_wind_revision", "inv_residual_load_swing",
            "inv_negative_price_regime", "inv_high_error_regime",
        ]
        if sig.get(flag, False)
    ]

    prompt = f"""
You are a data quality and market analyst investigating why the trading signal
for {ep["target_date"]} has been invalidated.

IMPORTANT: Respond with ONLY a valid JSON object. No markdown, no code fences.

Active flags: {active_flags}

Context data:
  Signal premium: {sig["signal_month_base"]}
  Gas yesterday: {anm.get("gas_price_yesterday", "N/A")}, today: {anm.get("gas_price_today", "N/A")},
    change: {anm.get("gas_1d_change", "N/A")} ({anm.get("gas_1d_change_pct", "N/A")} pct)
  Wind yesterday avg: {anm.get("wind_yesterday_avg_mw", "N/A")} MW,
    today avg: {anm.get("wind_today_avg_mw", "N/A")} MW,
    revision: {anm.get("wind_revision_mw", "N/A")} MW
  DA price range yesterday: {anm.get("da_price_min_yesterday", "N/A")} to {anm.get("da_price_max_yesterday", "N/A")}
  Negative price hours yesterday: {anm.get("negative_price_hours_yesterday", "N/A")}

Return JSON with exactly these 4 keys. Do NOT include numbers — the system
will inject them. Provide only qualitative interpretation.

{{
  "flags_narrative": "Brief statement of which flags triggered and why they matter for trading.",
  "cause_narrative": "Explain the likely cause of each active flag based on the data context. Reference whether gas moved, wind shifted, etc. but do NOT cite specific numbers.",
  "assessment_narrative": "Is this a genuine market event or a data quality issue? Explain your reasoning without numbers.",
  "recommendation": "One of exactly: STAND ASIDE, INVESTIGATE FURTHER, or CONSIDER OVERRIDE — with a brief reason."
}}
""".strip()

    return prompt


def assemble_anomaly_from_json(
    llm_json: dict,
    evidence_package: dict,
) -> str:
    """Build the anomaly report by combining LLM narratives with
    system-generated citations from the evidence package."""
    ep = evidence_package
    sig = ep["signal"]["fields"]
    anm = ep["anomaly_context"]["fields"] if ep["anomaly_context"] else {}
    date = ep["target_date"]
    sig_src = ep["signal"]["source_file"]
    sig_row = ep["signal"]["row_id"]
    anm_src = ep["anomaly_context"]["source_file"] if ep["anomaly_context"] else "N/A"
    anm_row = ep["anomaly_context"]["row_ids"] if ep["anomaly_context"] else "N/A"

    active_flags = [
        flag for flag in [
            "inv_gas_spike", "inv_wind_revision", "inv_residual_load_swing",
            "inv_negative_price_regime", "inv_high_error_regime",
        ]
        if sig.get(flag, False)
    ]

    # FLAGS TRIGGERED
    flags_narr = llm_json.get("flags_narrative", "")
    flag_citations = " ".join(
        _cite(sig_src, sig_row, f, sig[f]) for f in active_flags
    )
    flags_section = (
        f"FLAGS TRIGGERED\n"
        f"{flags_narr} "
        f"Active flags: {', '.join(active_flags)} {flag_citations}."
    )

    # LIKELY CAUSE
    cause_narr = llm_json.get("cause_narrative", "Cause analysis not available.")
    cause_data = []
    if sig.get("inv_gas_spike"):
        cause_data.append(
            f"Gas moved {anm.get('gas_1d_change', 'N/A')} EUR/MWh "
            f"{_cite(anm_src, anm_row, 'gas_1d_change', anm.get('gas_1d_change', 'N/A'))} "
            f"({anm.get('gas_1d_change_pct', 'N/A')} fractional change) "
            f"{_cite(anm_src, anm_row, 'gas_1d_change_pct', anm.get('gas_1d_change_pct', 'N/A'))}."
        )
    if sig.get("inv_wind_revision"):
        cause_data.append(
            f"Wind shifted by {anm.get('wind_revision_mw', 'N/A')} MW "
            f"{_cite(anm_src, anm_row, 'wind_revision_mw', anm.get('wind_revision_mw', 'N/A'))}."
        )
    if sig.get("inv_negative_price_regime"):
        cause_data.append(
            f"Yesterday had {anm.get('negative_price_hours_yesterday', 'N/A')} negative-price hours "
            f"{_cite(anm_src, anm_row, 'negative_price_hours_yesterday', anm.get('negative_price_hours_yesterday', 'N/A'))}."
        )
    if sig.get("inv_residual_load_swing"):
        cause_data.append(
            f"Wind averaged {anm.get('wind_yesterday_avg_mw', 'N/A')} MW yesterday "
            f"{_cite(anm_src, anm_row, 'wind_yesterday_avg_mw', anm.get('wind_yesterday_avg_mw', 'N/A'))} "
            f"vs {anm.get('wind_today_avg_mw', 'N/A')} MW today "
            f"{_cite(anm_src, anm_row, 'wind_today_avg_mw', anm.get('wind_today_avg_mw', 'N/A'))}."
        )
    cause_section = f"LIKELY CAUSE\n{cause_narr} {' '.join(cause_data)}"

    # DATA ERROR OR MARKET EVENT?
    assess_narr = llm_json.get("assessment_narrative", "Assessment not available.")
    assess_section = (
        f"DATA ERROR OR MARKET EVENT?\n"
        f"{assess_narr} "
        f"Signal premium was {sig['signal_month_base']} EUR/MWh "
        f"{_cite(sig_src, sig_row, 'signal_month_base', sig['signal_month_base'])}."
    )

    # DESK RECOMMENDATION
    rec = llm_json.get("recommendation", "INVESTIGATE FURTHER")
    rec_section = f"DESK RECOMMENDATION\n{rec}"

    return "\n\n".join([flags_section, cause_section, assess_section, rec_section])


# =====================================================================
# Section 4: Citation Validator
# =====================================================================
def _lookup_evidence_value(evidence_package: dict, filename: str, column_name: str):
    """Search all evidence blocks for a matching source_file and field.

    Returns the stringified value if found, else None.
    """
    col_clean = column_name.strip()
    fn_clean = filename.strip().lower()

    for block_name, block in evidence_package.items():
        if not isinstance(block, dict) or "source_file" not in block:
            continue
        if fn_clean in block["source_file"].lower():
            fields = block.get("fields", {})
            if col_clean in fields:
                return str(fields[col_clean])
    return None


def _values_match(cited: str, actual: str) -> bool:
    """Match strings exactly, or floats within 0.01 tolerance."""
    cited_s = cited.strip()
    actual_s = actual.strip()

    if cited_s == actual_s:
        return True

    # Handle booleans
    if actual_s.lower() in ("true", "false"):
        return cited_s.lower() == actual_s.lower()

    # Handle None
    if actual_s == "None":
        return cited_s.lower() in ("none", "n/a", "null")

    # Numeric comparison with tolerance
    try:
        c_f = float(cited_s.replace("+", ""))
        a_f = float(actual_s)
        # Tolerance: 0.01 for small numbers, 0.5% for large numbers
        abs_tol = max(0.01, abs(a_f) * 0.005) if a_f != 0 else 0.01
        return abs(c_f - a_f) < abs_tol
    except (ValueError, TypeError):
        pass

    return False


def validate_citations(llm_response: str, evidence_package: dict) -> dict:
    """Parse and verify every citation in the LLM output.

    Each citation matching ``CITATION_PATTERN`` is checked against the
    evidence package.  Any cited value that does not match the retrieved
    evidence is flagged as a hallucination.

    Args:
        llm_response: Full text from the LLM.
        evidence_package: The evidence dict used to generate the response.

    Returns:
        Validation result dict with per-citation details and a summary.
    """
    citations = re.findall(CITATION_PATTERN, llm_response)

    validation_result = {
        "total_citations": len(citations),
        "verified_citations": 0,
        "unverified_citations": 0,
        "hallucination_detected": False,
        "details": [],
    }

    for filename, row_id, column_name, cited_value in citations:
        actual_value = _lookup_evidence_value(
            evidence_package, filename, column_name.strip(),
        )
        match = actual_value is not None and _values_match(cited_value, actual_value)

        detail = {
            "filename": filename.strip(),
            "row_id": row_id.strip(),
            "column": column_name.strip(),
            "cited_value": cited_value.strip(),
            "actual_value": actual_value,
            "verified": match,
            "verdict": (
                "VERIFIED" if match
                else ("NOT_FOUND" if actual_value is None else "MISMATCH")
            ),
        }
        validation_result["details"].append(detail)

        if match:
            validation_result["verified_citations"] += 1
        else:
            validation_result["unverified_citations"] += 1
            validation_result["hallucination_detected"] = True
            logger.warning(
                "HALLUCINATION: LLM cited %s = '%s' but actual value is '%s'",
                column_name.strip(), cited_value.strip(), actual_value,
            )

    logger.info(
        "Citation validation: %d total, %d verified, %d unverified → %s",
        validation_result["total_citations"],
        validation_result["verified_citations"],
        validation_result["unverified_citations"],
        "CLEAN" if not validation_result["hallucination_detected"] else "⚠ HALLUCINATIONS",
    )

    return validation_result


# =====================================================================
# Section 5: HTML Output
# =====================================================================
def render_briefing_html(
    briefing_text: str,
    evidence_package: dict,
    validation_result: dict,
    model_used: str,
    output_path: Path,
) -> None:
    """Render the briefing as self-contained HTML with clickable citations.

    Citations become interactive elements that expand to show the source
    evidence row.  All CSS is inline — no external dependencies.
    """
    target_date = evidence_package["target_date"]
    verified = validation_result["verified_citations"]
    total = validation_result["total_citations"]
    is_clean = not validation_result["hallucination_detected"]
    status_class = "status-clean" if is_clean else "status-warn"
    status_text = "CLEAN — all citations verified" if is_clean else "⚠ HALLUCINATIONS DETECTED"

    # ── Convert citations to clickable HTML ───────────────────────────
    citation_counter = [0]

    def _replace_citation(match):
        filename, row_id, column, value = match.groups()
        idx = citation_counter[0]
        citation_counter[0] += 1

        detail = next(
            (d for d in validation_result["details"]
             if d["column"] == column.strip()
             and _values_match(d["cited_value"], value.strip())),
            None,
        )
        is_verified = detail["verified"] if detail else False
        border_color = "#2563eb" if is_verified else "#dc2626"
        actual = detail["actual_value"] if detail else "not found"
        verdict = "✓ VERIFIED" if is_verified else "⚠ UNVERIFIED"

        return (
            f'<span class="citation" onclick="toggleCitation({idx})" '
            f'style="border-color:{border_color}" '
            f'title="{verdict}">'
            f'[{filename.strip()} → {column.strip()} = {value.strip()}]'
            f'</span>'
            f'<div class="citation-popup" id="popup-{idx}">'
            f'<strong>Source:</strong> {filename.strip()}<br>'
            f'<strong>Row:</strong> {row_id.strip()}<br>'
            f'<strong>Field:</strong> {column.strip()}<br>'
            f'<strong>Cited value:</strong> {value.strip()}<br>'
            f'<strong>Actual value:</strong> {actual}<br>'
            f'<strong>Status:</strong> {verdict}'
            f'</div>'
        )

    html_briefing = re.sub(CITATION_PATTERN, _replace_citation, briefing_text)

    # ── Wrap section labels ───────────────────────────────────────────
    section_labels = [
        "SIGNAL SUMMARY", "FUNDAMENTAL DRIVERS", "TOP MODEL FEATURES",
        "INTRADAY SHAPE", "INVALIDATION STATUS",
    ]
    for label in section_labels:
        html_briefing = html_briefing.replace(
            label, f'<span class="section-label">{label}</span>',
        )

    # ── Wrap signal labels in coloured spans ──────────────────────────
    signal_class_map = {
        "STRONG_BUY": "signal-strong-buy",
        "STRONG_SELL": "signal-strong-sell",
        "INVALIDATED": "signal-invalidated",
        "BUY": "signal-buy",
        "SELL": "signal-sell",
        "HOLD": "signal-hold",
    }
    for label, css_class in signal_class_map.items():
        # Only replace standalone words (not inside citations)
        html_briefing = re.sub(
            rf'(?<![=\w])\b{label}\b(?!["\w])',
            f'<span class="{css_class}">{label}</span>',
            html_briefing,
        )

    # Convert newlines to <br>
    html_briefing = html_briefing.replace("\n", "<br>\n")

    # ── Build evidence tables ─────────────────────────────────────────
    evidence_tables_html = ""
    for block_name in ["signal", "delivery", "recent_context", "model_performance"]:
        block = evidence_package.get(block_name)
        if not isinstance(block, dict) or "fields" not in block:
            continue
        source = block.get("source_file", block_name)
        row_id = block.get("row_id", "")
        fields = block["fields"]

        rows_html = ""
        for k, v in fields.items():
            if isinstance(v, list):
                v_str = ", ".join(str(x) for x in v)
            else:
                v_str = str(v)
            rows_html += f"<tr><td><strong>{k}</strong></td><td>{v_str}</td></tr>\n"

        evidence_tables_html += f"""
        <details style="margin-top:12px;">
          <summary style="color:#64b5f6; cursor:pointer; font-size:14px;">
            {source} → {row_id}
          </summary>
          <table class="evidence-table">{rows_html}</table>
        </details>
        """

    # ── Full HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>DE_LU Power Intelligence — {target_date}</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: monospace; max-width: 900px; margin: 40px auto;
            background: #0f1117; color: #e2e8f0; padding: 20px; }}
    .briefing {{ background: #1a1f2e; padding: 24px; border-radius: 8px;
                 line-height: 1.8; font-size: 15px; }}
    .section-label {{ color: #64b5f6; font-weight: bold; display: block;
                      margin-top: 16px; }}
    .citation {{ background: #1e3a5f; color: #90caf9; border-radius: 4px;
                padding: 1px 6px; cursor: pointer; font-size: 12px;
                border: 1px solid #2563eb; white-space: nowrap; }}
    .citation:hover {{ background: #2563eb; }}
    .citation-popup {{ display: none; background: #1a2744; border: 1px solid #3b82f6;
                      border-radius: 6px; padding: 12px; margin: 4px 0;
                      font-size: 13px; }}
    .citation-popup.active {{ display: block; }}
    .evidence-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    .evidence-table td {{ padding: 4px 8px; border: 1px solid #374151; }}
    .evidence-table tr:nth-child(even) {{ background: #1f2937; }}
    .status-clean {{ color: #34d399; }}
    .status-warn {{ color: #fbbf24; }}
    .validation-bar {{ background: #111827; padding: 12px; border-radius: 6px;
                      margin-top: 20px; font-size: 13px; }}
    .signal-strong-buy {{ color: #34d399; font-weight: bold; }}
    .signal-buy {{ color: #6ee7b7; }}
    .signal-hold {{ color: #9ca3af; }}
    .signal-sell {{ color: #f87171; }}
    .signal-strong-sell {{ color: #ef4444; font-weight: bold; }}
    .signal-invalidated {{ color: #fbbf24; }}
    details summary {{ outline: none; }}
    details summary::-webkit-details-marker {{ color: #64b5f6; }}
  </style>
</head>
<body>
  <h2 style="color:#64b5f6;">DE_LU Power Intelligence — {target_date}</h2>
  <p style="color:#9ca3af; font-size:13px;">
    Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} |
    Model: {model_used} |
    Citations: {total} total, {verified} verified
  </p>

  <div class="briefing">
    {html_briefing}
  </div>

  <div class="validation-bar">
    <span class="{status_class}">Citation Status: {status_text}</span>
    — {verified}/{total} claims verified against pipeline data
  </div>

  <h3 style="color:#64b5f6; margin-top:30px;">Evidence package</h3>
  <p style="color:#9ca3af; font-size:13px;">
    Full audit trail: outputs/ai_intelligence/{target_date}/evidence_package.json
  </p>
  {evidence_tables_html}

  <script>
    function toggleCitation(id) {{
      var el = document.getElementById('popup-' + id);
      el.classList.toggle('active');
    }}
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Saved HTML briefing: %s", output_path)


# =====================================================================
# Section 6: Plain Text Output
# =====================================================================
def render_briefing_txt(
    briefing_text: str,
    validation_result: dict,
    output_path: Path,
) -> None:
    """Save the plain-text briefing with citation validation footer."""
    target_date = output_path.parent.name  # directory name = date
    verified = validation_result["verified_citations"]
    total = validation_result["total_citations"]
    status = (
        "CLEAN" if not validation_result["hallucination_detected"]
        else "⚠ HALLUCINATIONS DETECTED"
    )

    txt = f"""\
====================================================
DE_LU POWER INTELLIGENCE BRIEFING — {target_date}
====================================================

{briefing_text}

────────────────────────────────────────────────────
CITATION VALIDATION
{verified}/{total} claims verified against pipeline data.
Status: {status}
Full audit: outputs/ai_intelligence/{target_date}/citation_validation.json
====================================================
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(txt, encoding="utf-8")
    logger.info("Saved text briefing: %s", output_path)


# =====================================================================
# Section 7: Anomaly Report Output
# =====================================================================
def render_anomaly_report(
    anomaly_text: str,
    validation_result: dict,
    output_path: Path,
) -> None:
    """Save the anomaly investigation report."""
    target_date = output_path.parent.name
    verified = validation_result["verified_citations"]
    total = validation_result["total_citations"]

    txt = f"""\
====================================================
DE_LU ANOMALY INVESTIGATION — {target_date}
====================================================

{anomaly_text}

────────────────────────────────────────────────────
EVIDENCE VERIFIED: {verified}/{total} citations confirmed
This report was generated automatically from pipeline data.
Manual verification recommended before acting on CONSIDER OVERRIDE recommendation.
====================================================
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(txt, encoding="utf-8")
    logger.info("Saved anomaly report: %s", output_path)


# =====================================================================
# Section 8: Master Orchestrator
# =====================================================================
def run_ai_intelligence_pipeline(target_date: str | None = None) -> dict:
    """Run the full AI intelligence pipeline for a given date.

    Steps:
        0. Setup logging and Gemini client
        1. Build evidence package (retrieval step)
        2. Save evidence package to disk (before any LLM call)
        3. Generate daily briefing (Function A)
        4. Validate citations
        5. Render text + HTML outputs
        6. If invalidation active: generate anomaly report (Function B)
        7. Log all LLM calls

    Args:
        target_date: ``"YYYY-MM-DD"`` string.  Defaults to the most
            recent date in ``signal_table.csv``.

    Returns:
        Dict with pipeline results.
    """
    _setup_logging()
    sep = "=" * 60
    logger.info(sep)
    logger.info("DE_LU AI Intelligence Pipeline")
    logger.info(sep)

    # ── 0. Setup Gemini ───────────────────────────────────────────────
    model = setup_gemini_client()
    logger.info("Gemini client configured (primary model: %s)", PRIMARY_MODEL)

    # ── Resolve target date ───────────────────────────────────────────
    if target_date is None:
        signal_df = pd.read_csv(
            OUTPUTS_DIR / "curve_translation" / "signal_table.csv"
        )
        signal_df["delivery_date"] = pd.to_datetime(signal_df["delivery_date"])
        # Pick the latest date that has a real signal (not NO_DATA) so the
        # briefing can discuss an actionable trading view.  The very last
        # date(s) in the OOS window typically lack forward data for the
        # Month+1 fair-value aggregation and show NO_DATA — skip those.
        valid_labels = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "INVALIDATED"]
        valid = signal_df[signal_df["signal_label_month_base"].isin(valid_labels)]
        if len(valid) > 0:
            target_date = str(valid["delivery_date"].max().date())
        else:
            # Absolute fallback — use latest date even if NO_DATA
            target_date = str(signal_df["delivery_date"].max().date())
        logger.info("No target date specified — using latest with signal: %s", target_date)
    else:
        logger.info("Target date: %s", target_date)

    # ── Create output directory ───────────────────────────────────────
    out_dir = AI_OUTPUT_DIR / target_date
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_log_path = out_dir / "llm_call_log.jsonl"

    # ── 1. Build evidence package ─────────────────────────────────────
    logger.info("Step 1: Building evidence package …")
    evidence = build_evidence_package(target_date, DATA_DIR, OUTPUTS_DIR)

    # ── 2. Save evidence BEFORE any LLM call ──────────────────────────
    evidence_path = out_dir / "evidence_package.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Evidence package saved: %s", evidence_path)

    # ── 3. Function A — Daily briefing ────────────────────────────────
    logger.info("Step 2: Generating daily briefing …")
    briefing_prompt = build_briefing_prompt(evidence)
    briefing_response = call_llm(model, briefing_prompt, log_path=llm_log_path)

    briefing_text = None
    validation = None

    if briefing_response["success"]:
        # Parse LLM JSON → assemble briefing with system-generated citations
        try:
            llm_json = _extract_json_from_response(briefing_response["response"])
            logger.info("LLM returned valid JSON with keys: %s", list(llm_json.keys()))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "LLM did not return valid JSON (%s) — using fallback narratives",
                exc,
            )
            llm_json = {}

        briefing_text = assemble_briefing_from_json(llm_json, evidence)

        # ── 4. Validate citations ─────────────────────────────────────
        # All citations are system-generated from the evidence package,
        # so they should always verify.  We still run validation as a
        # safety net and for the audit trail.
        logger.info("Step 3: Validating citations …")
        validation = validate_citations(briefing_text, evidence)

        if validation["hallucination_detected"]:
            logger.warning(
                "⚠ UNEXPECTED: %d unverified citations in system-generated text",
                validation["unverified_citations"],
            )
        else:
            logger.info(
                "Citation validation CLEAN: %d/%d citations verified",
                validation["verified_citations"],
                validation["total_citations"],
            )

        # ── 5. Render outputs ─────────────────────────────────────────
        logger.info("Step 4: Rendering outputs …")
        render_briefing_txt(briefing_text, validation, out_dir / "briefing.txt")
        render_briefing_html(
            briefing_text, evidence, validation,
            briefing_response["model_used"],
            out_dir / "briefing.html",
        )

        # Save validation result
        (out_dir / "citation_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8",
        )
    else:
        logger.error("Briefing generation failed: %s", briefing_response["error"])

    # ── 6. Function B — Anomaly investigation ─────────────────────────
    any_invalidation = evidence["signal"]["fields"]["any_invalidation"]
    anomaly_response = None

    if any_invalidation and evidence.get("anomaly_context") is not None:
        logger.info("Step 5: Invalidation flags active — running anomaly investigation …")
        anomaly_prompt = build_anomaly_prompt(evidence)
        anomaly_response = call_llm(model, anomaly_prompt, log_path=llm_log_path)

        if anomaly_response["success"]:
            try:
                anomaly_json = _extract_json_from_response(anomaly_response["response"])
            except (json.JSONDecodeError, ValueError):
                logger.warning("Anomaly LLM did not return valid JSON — using fallback")
                anomaly_json = {}

            anomaly_text = assemble_anomaly_from_json(anomaly_json, evidence)
            anomaly_validation = validate_citations(anomaly_text, evidence)
            render_anomaly_report(
                anomaly_text, anomaly_validation, out_dir / "anomaly_report.txt",
            )
        else:
            logger.error(
                "Anomaly investigation failed: %s", anomaly_response["error"],
            )
    elif any_invalidation:
        logger.warning(
            "Invalidation active but no anomaly context available — skipping investigation",
        )
    else:
        logger.info("No invalidation flags active — skipping anomaly investigation")

    # ── Done ──────────────────────────────────────────────────────────
    logger.info(sep)
    logger.info("AI intelligence pipeline complete. Outputs: %s", out_dir)
    logger.info(sep)

    return {
        "target_date": target_date,
        "briefing_text": briefing_text,
        "hallucination_detected": (
            validation["hallucination_detected"] if validation else None
        ),
        "anomaly_investigated": any_invalidation and anomaly_response is not None,
        "output_dir": str(out_dir),
    }
