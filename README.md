# Cobblestone Energy — DE_LU Day-Ahead Power Price Forecasting

A five-component daily-run Python pipeline for German day-ahead electricity price forecasting, signal generation, AI-powered market intelligence, and automated PDF reporting. Built for the Cobblestone Energy technical case study.

**Best model**: LGBM+Ridge ensemble with 2-pass AR residual correction + hourly-EW bias adjustment
- Hourly MAE: **13.6 €/MWh** | Weekly MAE: **5.79 €/MWh** | Monthly MAE: **4.00 €/MWh**
- Directional accuracy: **86.7%** | Skill vs naive: **+0.61**
- OOS period: Nov 2024 – Mar 2026 (11,999 hours)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys in .env
echo "ENTSO_E_API_KEY=your_key_here" > .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# 3. Download TTF gas price data (manual step — see below)

# 4. Run the full pipeline (Tasks 1–5)
python scripts/run_ingestion.py          # Task 1: Ingest + QA
python scripts/run_forecasting.py        # Task 2: Train models + OOS predictions
python -m src.curve_translation          # Task 3: Signal generation
python scripts/run_ai_intelligence.py    # Task 4: AI briefing (requires Gemini key)
python scripts/start_server.py           # Task 5: Web dashboard + PDF report
```

## Gas Price Data (Manual Download Required)

TTF natural gas prices are not available via a free API. Download manually:

1. Go to https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data
2. Set date range: **2021-01-01** to **2026-03-15**
3. Click **Download Data**
4. Save the CSV as: `data/raw/gas_price/ttf_daily.csv`

Required columns: `Date`, `Price`

---

## Pipeline Overview

| Task | Module | What it does | Entry point |
|------|--------|-------------|-------------|
| **1. Ingestion & QA** | `src/ingestion.py` | Fetch ENTSO-E + gas data, run 10 QA checks, auto-correct, flag anomalies | `python scripts/run_ingestion.py` |
| **2. Forecasting** | `src/models.py`, `src/features.py` | 60+ features, LGBM+Ridge ensemble, walk-forward CV, OOS evaluation | `python scripts/run_forecasting.py` |
| **3. Curve Translation** | `src/curve_translation.py` | Convert forecasts to BUY/SELL/HOLD signals with confidence + invalidation | `python -m src.curve_translation` |
| **4. AI Intelligence** | `src/ai_intelligence.py` | RAG-based market briefings via Gemini with citation validation | `python scripts/run_ai_intelligence.py` |
| **5. Daily Reporting** | `src/report_generator.py`, `server/` | PDF reports, Excel workbooks, web dashboard, email delivery | `python scripts/start_server.py` |

---

## Data Sources

| Dataset | Source | Method | URL |
|---------|--------|--------|-----|
| DA Prices | ENTSO-E Transparency Platform | `entsoe-py` `query_day_ahead_prices` | https://transparency.entsoe.eu |
| Wind+Solar Forecast | ENTSO-E Transparency Platform | `query_wind_and_solar_forecast` | https://transparency.entsoe.eu |
| Load Forecast | ENTSO-E Transparency Platform | `query_load_forecast` | https://transparency.entsoe.eu |
| Gas Price (TTF) | Investing.com | Manual CSV download | https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data |

---

## Project Structure

```
cobblestone_project/
│
├── src/                                    # Core pipeline modules
│   ├── __init__.py
│   ├── ingestion.py                        # Task 1: API fetch, QA checks, auto-corrections
│   ├── features.py                         # Feature engineering (60+ features)
│   ├── models.py                           # Walk-forward CV, LGBM+Ridge ensemble, metrics
│   ├── curve_translation.py                # Task 3: Fair-value signals, invalidation logic
│   ├── ai_intelligence.py                  # Task 4: RAG briefings, citation validation
│   ├── report_generator.py                 # Task 5: PDF/Excel/figure generation
│   ├── autogluon_forecaster.py             # AutoGluon tabular benchmark
│   └── autogluon_ts_forecaster.py          # AutoGluon time-series benchmark
│
├── scripts/                                # Entry points & standalone tools
│   ├── run_ingestion.py                    # Run Task 1
│   ├── run_forecasting.py                  # Run Task 2 (full training pipeline)
│   ├── run_curve_translation.py            # Run Task 3
│   ├── run_ai_intelligence.py              # Run Task 4
│   ├── run_model_comparison.py             # Reproduce model comparison table (all baselines)
│   ├── run_autogluon_tabular.py            # AutoGluon benchmark (standalone)
│   ├── run_autogluon_cv.py                 # AutoGluon cross-validation
│   ├── run_autogluon_oos.py                # AutoGluon OOS evaluation
│   ├── extend_to_today.py                  # Extend predictions to today via ENTSO-E API
│   └── start_server.py                     # Launch web dashboard (Task 5)
│
├── server/                                 # Web dashboard & scheduling
│   ├── __init__.py
│   ├── app.py                              # Flask web application
│   ├── scheduler.py                        # Full pipeline orchestrator (daily runs)
│   ├── mailer.py                           # Email delivery of PDF reports
│   └── templates/
│       ├── index.html                      # Dashboard UI
│       ├── report.html                     # PDF report template (Jinja2)
│       └── email_body.html                 # Email template
│
├── data/
│   ├── raw/                                # Cached API responses (never modified)
│   │   ├── da_prices/                      #   Day-ahead price parquets (per year)
│   │   ├── wind_solar/                     #   Wind+solar forecast parquets
│   │   ├── load_forecast/                  #   Load forecast parquets
│   │   └── gas_price/
│   │       └── ttf_daily.csv               #   Manual TTF gas download
│   └── processed/
│       ├── de_power_dataset.parquet        # Clean hourly dataset (Task 1 output)
│       ├── feature_matrix.parquet          # 60+ engineered features (Task 2 input)
│       └── oos_predictions.parquet         # OOS predictions with all model variants
│
├── models/                                 # Serialised model artefacts
│   ├── lightgbm_final.pkl                  # Trained LightGBM model
│   ├── lgbm_ridge_ensemble.pkl             # Full ensemble (LGBM+Ridge, both passes)
│   ├── xgboost_final.pkl                   # XGBoost model
│   ├── autogluon/                          # AutoGluon tabular model (no bagging)
│   ├── ag_5min/                            # AutoGluon 5-min benchmark (8-fold bagging)
│   └── ag_tabular_eval/                    # AutoGluon evaluation model (4-fold bagging)
│
├── outputs/                                # All pipeline outputs
│   ├── model_performance_report.txt        # Full model metrics (hourly/weekly/monthly)
│   ├── submission.csv                      # Submission-format predictions
│   ├── multi_granularity_accuracy.csv      # MAE/RMSE/MAPE at all granularities
│   ├── delivery_period_aggregations.csv    # Aggregated delivery-period forecasts
│   │
│   ├── qa_report_summary.txt              # QA check results (Task 1)
│   ├── qa_flagged_rows.csv                # All flagged rows with reasons
│   ├── auto_corrections_log.csv           # 591 solar-at-night corrections
│   │
│   ├── fig_forecast_vs_actual.png          # OOS forecast vs actual timeseries
│   ├── fig_feature_importance.png          # Top 25 LightGBM feature importances
│   ├── fig_model_comparison.png            # Model comparison chart
│   ├── fig_walk_forward_errors.png         # Walk-forward CV error distribution
│   │
│   ├── curve_translation/                  # Task 3 outputs
│   │   ├── signal_table.csv               #   Daily signal table (BUY/SELL/HOLD per delivery)
│   │   ├── delivery_periods.csv           #   Delivery period fair values
│   │   ├── curve_translation_report.txt   #   Summary report
│   │   ├── fig_ct_01_signal_dashboard.png #   Signal overview dashboard
│   │   ├── fig_ct_02_shape_premium.png    #   Base vs peak shape premium
│   │   ├── fig_ct_03_signal_backtest.png  #   Historical signal backtest
│   │   ├── fig_ct_04_invalidation_monitor.png  # Invalidation flag timeline
│   │   └── fig_ct_05_confidence_bands.png #   Confidence band visualisation
│   │
│   ├── ai_intelligence/                    # Task 4 outputs (per date)
│   │   └── YYYY-MM-DD/
│   │       ├── briefing.txt               #   Plain-text market briefing
│   │       ├── briefing.html              #   HTML-formatted briefing
│   │       ├── evidence_package.json      #   RAG evidence cards
│   │       ├── citation_validation.json   #   Citation audit results
│   │       ├── anomaly_report.txt         #   Anomaly investigation (if flags active)
│   │       └── llm_call_log.jsonl         #   Full LLM call audit trail
│   │
│   ├── reports/                            # Task 5 outputs (per date)
│   │   └── YYYY-MM-DD/
│   │       ├── daily_report.pdf           #   Desk-ready PDF report
│   │       ├── fig_forecast_tomorrow.png  #   24h forecast chart
│   │       ├── fig_signal_dashboard.png   #   Signal dashboard chart
│   │       └── raw_data.xlsx              #   Excel workbook (5 sheets)
│   │
│   └── plots/                              # Exploratory data analysis plots
│       ├── 01_full_timeseries.png
│       ├── 02_price_distribution_by_year.png
│       ├── 03_daily_profiles_by_season.png
│       ├── 04_monthly_price_heatmap.png
│       ├── 05_renewables_vs_price.png
│       ├── 06_gas_vs_da_price.png
│       ├── 07_missing_data.png
│       ├── 08_renewable_penetration.png
│       ├── 09_weekly_patterns.png
│       └── 10_summary_dashboard.png
│
├── plots/model_plots/                      # Detailed model analysis plots
│   ├── fig_12fold_confidence.png           #   12-fold CV confidence intervals
│   ├── fig_cumulative_error.png            #   Cumulative error over OOS period
│   ├── fig_ensemble_hp_analysis.png        #   Ensemble hyperparameter sensitivity
│   ├── fig_error_heatmaps.png              #   Error heatmap by hour × month
│   ├── fig_feature_importance.png          #   Feature importance (detailed)
│   ├── fig_forecast_vs_actual.png          #   Forecast vs actual (detailed)
│   ├── fig_full_model_comparison.png       #   All models side-by-side
│   ├── fig_hyperparameter_analysis.png     #   HP tuning analysis
│   ├── fig_model_comparison.png            #   Model comparison bars
│   ├── fig_walk_forward_errors.png         #   Walk-forward error per fold
│   ├── 12fold_confidence_report.txt        #   CV confidence report
│   ├── 12fold_mae_per_fold.csv             #   Per-fold MAE values
│   ├── comprehensive_report.txt            #   Full model analysis report
│   ├── ensemble_hp_report.txt              #   Ensemble HP report
│   └── hyperparameter_report.txt           #   HP tuning report
│
├── logs/                                   # Runtime logs
│   └── ingestion.log
│
├── report.tex                              # LaTeX report source
├── sample_daily_report.pdf                 # Example daily PDF report (Task 5 output)
├── requirements.txt
├── server_config.json                      # Dashboard config (email, schedule)
├── .env                                    # API keys (not committed)
├── .gitignore
└── README.md
```

---
# Task 1
## Output Dataset

The pipeline produces `data/processed/de_power_dataset.parquet` with these columns:

| Column | Unit | Resolution | Source |
|--------|------|------------|--------|
| `da_price_eur_mwh` | €/MWh | Hourly | ENTSO-E |
| `wind_forecast_mw` | MW | Hourly | ENTSO-E |
| `solar_forecast_mw` | MW | Hourly | ENTSO-E |
| `load_forecast_mw` | MW | Hourly | ENTSO-E |
| `gas_price_eur_mwh` | €/MWh | Daily→Hourly (ffill) | Investing.com |

- **Index**: `timestamp_utc` — hourly, UTC-aware, 2021-01-01 to 2026-03-15
- **Rows**: 45,600 hourly observations

---

## QA Checks

The pipeline runs 10 automated checks:

1. **Timestamp completeness** — no gaps in the hourly grid
2. **Duplicate timestamps** — none after UTC conversion
3. **Missingness per column** — warns >1%, errors >5%
4. **DA price outliers** — physical bounds (−500 to 3000 €/MWh) + statistical 4σ with crisis-period awareness
5. **Wind/solar bounds** — no negatives, capacity cap, no solar at night (591 auto-corrections applied)
6. **Load bounds** — 20–90 GW range, holiday exceptions
7. **Gas price bounds** — 0–400 €/MWh, stale detection >72h
8. **Stale data detection** — ≥6 consecutive identical values
9. **Cross-series alignment** — wind-price correlation (−0.339), gas completeness
10. **DST boundary verification** — all transition dates validated

**QA results**: 1,621 rows flagged (1,183 errors, 42 warnings, 396 informational), 591 auto-corrections. See `outputs/qa_report_summary.txt` and `outputs/qa_flagged_rows.csv`.

---
# Task 2
## Model Architecture

The forecasting engine uses a **LGBM+Ridge ensemble** with two-pass training:

1. **Pass 1 (base)**: LightGBM (800 trees, 63 leaves, lr=0.01) blended 90/10 with Ridge regression. Recency weighting (halflife=365d) downweights the 2022 energy crisis.
2. **Pass 2 (AR correction)**: Residuals from Pass 1 are lagged (24h, 48h, 168h) and a second ensemble captures persistent forecast biases. Reduces MAE by ~2 €/MWh.
3. **Hourly-EW bias correction**: Hour-stratified exponentially-weighted expanding mean (halflife=672h). Trades +0.33 hourly MAE for MBE reduction from +4.86 to −0.50 €/MWh.

**Target transform**: Multi-scale asinh (c ∈ {0.5, 1.0, 2.0}) for variance stabilisation across the −100 to +500 €/MWh price range.

### Model Comparison (OOS: Nov 2024 – Mar 2026)

| Model | Hourly MAE | Hourly RMSE | Dir.Acc | Weekly MAE | Monthly MAE |
|-------|-----------|-------------|---------|-----------|------------|
| Seasonal Naive (24h) | 27.16 | 51.92 | — | 10.62 | 5.29 |
| Seasonal Naive (168h) | 34.84 | 57.43 | — | 18.93 | 5.67 |
| Ridge Regression | 18.74 | 32.13 | 81.6% | 8.35 | 5.09 |
| LightGBM (standalone) | 14.50 | 28.01 | 85.1% | 6.72 | 4.42 |
| **LGBM+Ridge+AR+EW** | **13.60** | **27.01** | **86.7%** | **5.79** | **4.00** |

Reproduce with: `python scripts/run_model_comparison.py`

---
# Task 3
## Curve Translation 

Converts hourly forecasts into trading signals for Week+1, Month+1, Quarter+1 delivery periods:

- **Signal tiers**: STRONG_BUY (>+8€), BUY (+3–8€), HOLD (±3€), SELL (−3–8€), STRONG_SELL (<−8€)
- **Confidence sizing**: ≥75% → full position, 50–75% → half, <50% → no position
- **5 invalidation flags**: gas spike, wind revision, residual load swing, negative prices, high recent error

---
# Task 4
## AI Intelligence 

RAG-based market briefings using Google Gemini (`gemini-2.0-flash`):

1. **Evidence assembly** — structured "evidence cards" from pipeline outputs
2. **Constrained generation** — must cite specific numbers, no fabrication
3. **Citation validation** — regex-based post-generation audit, zero tolerance for hallucinations
4. **Anomaly reports** — auto-generated when invalidation flags are active

Requires `GEMINI_API_KEY` in `.env`.

---

## Daily Reporting (Additional feature)

Automated daily report generation with web dashboard:

- **PDF report** — signal header, 24h forecast chart, signal dashboard, AI briefing, model diagnostics
- **Excel workbook** — 5 sheets with raw data, predictions, signals, delivery periods, model metrics
- **Web dashboard** — Flask app at `localhost:5000` for triggering runs and viewing reports
- **Email delivery** — configurable recipient and schedule via `server_config.json`
- **Extend to today** — `scripts/extend_to_today.py` fetches latest ENTSO-E data and re-predicts using saved models

Run with: `python scripts/start_server.py`

---

## Reproducibility

All results can be reproduced from scratch:

```bash
# Full pipeline
python scripts/run_ingestion.py              # Task 1: ~5 min (API calls)
python scripts/run_forecasting.py            # Task 2: ~10 min (training)
python -m src.curve_translation              # Task 3: ~1 min
python scripts/run_ai_intelligence.py        # Task 4: ~30 sec (requires Gemini key)
python scripts/start_server.py               # Task 5: starts dashboard

# Standalone verification
python scripts/run_model_comparison.py       # Reproduce Table 1 metrics
python scripts/run_autogluon_tabular.py      # AutoGluon benchmark (~5 min)
python scripts/extend_to_today.py            # Extend predictions to today
```

---

## Caching

Raw API responses are cached to `data/raw/` as parquet files (one per year per series). Re-running the pipeline skips API calls for cached years. Delete the cache files to force a fresh fetch.
