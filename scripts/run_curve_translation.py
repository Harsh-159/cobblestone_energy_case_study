#!/usr/bin/env python3
"""
Run prompt curve translation pipeline (Part 3).

Translates hourly DA price forecasts from Part 2 into tradable signals
for European power forward contracts (EEX German Power Base/Peak futures).

Requires:
    data/processed/oos_predictions.parquet  (from Part 2)
    data/processed/de_power_dataset.parquet (from Part 1)

Produces:
    outputs/curve_translation/signal_table.csv
    outputs/curve_translation/delivery_periods.csv
    outputs/curve_translation/fig_ct_01_signal_dashboard.png
    outputs/curve_translation/fig_ct_02_shape_premium.png
    outputs/curve_translation/fig_ct_03_signal_backtest.png
    outputs/curve_translation/fig_ct_04_invalidation_monitor.png
    outputs/curve_translation/fig_ct_05_confidence_bands.png
    outputs/curve_translation/curve_translation_report.txt
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.curve_translation import run_curve_translation_pipeline

if __name__ == "__main__":
    results = run_curve_translation_pipeline()
    print("\nCurve translation complete.")
    print(f"Signal table: {len(results['signal_table'])} rows")
    print(f"Invalidated days: {results['signal_table']['any_invalidation'].sum()}")
    print(f"Backtest hit rate: {results['backtest']['summary']['hit_rate']:.1%}")
