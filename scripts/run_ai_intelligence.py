#!/usr/bin/env python3
"""
Run AI intelligence pipeline (Task 4).

Generates a RAG-based, evidence-anchored daily pre-trade briefing and
(if invalidation flags are active) an anomaly investigation report.
Every factual claim is grounded in pipeline output data and validated
post-generation for hallucinations.

Requires:
    outputs/curve_translation/signal_table.csv     (from Part 3)
    outputs/curve_translation/delivery_periods.csv (from Part 3)
    outputs/model_performance_report.txt           (from Part 2)
    data/processed/de_power_dataset.parquet        (from Part 1)
    GEMINI_API_KEY in .env

Produces (per run date):
    outputs/ai_intelligence/{date}/briefing.txt
    outputs/ai_intelligence/{date}/briefing.html
    outputs/ai_intelligence/{date}/evidence_package.json
    outputs/ai_intelligence/{date}/citation_validation.json
    outputs/ai_intelligence/{date}/anomaly_report.txt  (if flags active)
    outputs/ai_intelligence/{date}/llm_call_log.jsonl

Usage:
    python scripts/run_ai_intelligence.py                  # latest date
    python scripts/run_ai_intelligence.py 2026-03-10       # specific date
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_intelligence import run_ai_intelligence_pipeline

if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_ai_intelligence_pipeline(target_date=target_date)

    print(f"\nAI Intelligence Pipeline Complete")
    print(f"Date:                 {results['target_date']}")
    print(f"Hallucination found:  {results['hallucination_detected']}")
    print(f"Anomaly investigated: {results['anomaly_investigated']}")
    print(f"Outputs saved to:     {results['output_dir']}")
    print(f"\nBriefing preview:")
    print("─" * 60)
    if results["briefing_text"]:
        # Show first 800 chars
        preview = results["briefing_text"][:800]
        if len(results["briefing_text"]) > 800:
            preview += "\n…"
        print(preview)
