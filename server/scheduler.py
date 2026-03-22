"""
Pipeline scheduler — orchestrates the full daily run.

Functions:
    run_full_pipeline()   — execute all pipeline steps in sequence
    load_config()         — read server_config.json (with defaults)
    update_config(patch)  — merge patch dict and persist
    cleanup_old_reports() — remove reports older than retention window
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("scheduler")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "server_config.json"

DEFAULT_CONFIG = {
    "recipient_email": "",
    "delivery_time": "06:30",
    "timezone": "Europe/Berlin",
    "retention_days": 30,
    "enabled": True,
}


# ── Config helpers ───────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config from disk, creating defaults if missing."""
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            # Ensure all default keys present
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt config — resetting to defaults")
    # Write defaults
    CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    return dict(DEFAULT_CONFIG)


def update_config(patch: dict) -> dict:
    """Merge patch into current config and save."""
    cfg = load_config()
    cfg.update(patch)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info("Config updated: %s", list(patch.keys()))
    return cfg


# ── Data freshness ───────────────────────────────────────────────────────

def get_last_ingestion_time() -> datetime | None:
    """Return mtime of the main dataset, or None if missing."""
    dataset = PROJECT_ROOT / "data" / "processed" / "de_power_dataset.parquet"
    if dataset.exists():
        return datetime.fromtimestamp(dataset.stat().st_mtime)
    return None


# ── Pipeline runner ──────────────────────────────────────────────────────

def _run_script(name: str, script_path: str, args: list[str] | None = None) -> bool:
    """Run a pipeline script in a subprocess. Returns True on success."""
    cmd = [sys.executable, str(PROJECT_ROOT / script_path)]
    if args:
        cmd.extend(args)
    logger.info("  [%s] Starting …", name)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max per step
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info("  [%s] COMPLETE (%.1fs)", name, elapsed)
            return True
        else:
            logger.error("  [%s] FAILED (rc=%d, %.1fs)\n%s",
                         name, result.returncode, elapsed,
                         (result.stderr or result.stdout)[-500:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("  [%s] TIMEOUT after 1800s", name)
        return False
    except Exception as exc:
        logger.error("  [%s] ERROR: %s", name, exc)
        return False


def run_full_pipeline(target_date: str | None = None, on_step=None) -> dict:
    """
    Execute the full daily pipeline:
      1. Check data freshness
      2. Run ingestion (if stale)
      3. Run forecasting
      4. Run curve translation
      5. Run AI intelligence
      6. Generate report (figures + Excel + PDF)
      7. Send email
      8. Cleanup old reports

    Each step is wrapped in try/except — partial success over total failure.

    Parameters
    ----------
    target_date : str or None
        YYYY-MM-DD date to run for. None = latest.
    on_step : callable or None
        Called with a human-readable step label for live UI updates.

    Returns a status dict.
    """
    def _step(label: str):
        if on_step:
            on_step(label)

    run_start = time.time()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info("PIPELINE RUN START  id=%s  target_date=%s", run_id, target_date or "auto")
    logger.info("=" * 60)

    status = {
        "run_id": run_id,
        "started": datetime.utcnow().isoformat(),
        "steps": {},
        "target_date": target_date,
        "success": False,
        "pdf_path": None,
        "excel_path": None,
    }

    config = load_config()

    # Step 1: Check data freshness
    _step("Checking data freshness…")
    last_ingest = get_last_ingestion_time()
    data_stale = True
    if last_ingest is not None:
        age_hours = (datetime.now() - last_ingest).total_seconds() / 3600
        data_stale = age_hours > 23
        logger.info("  Data age: %.1f hours (%s)",
                     age_hours, "stale" if data_stale else "fresh")
    else:
        logger.info("  No existing dataset found — will run ingestion")

    # Step 2: Ingestion (if stale)
    if data_stale:
        _step("Ingesting ENTSO-E data…")
        status["steps"]["ingestion"] = _run_script(
            "Ingestion", "scripts/run_ingestion.py")
    else:
        status["steps"]["ingestion"] = True
        logger.info("  [Ingestion] Skipped — data is fresh")

    # Step 3: Extend dataset to today (fetch latest ENTSO-E, predict new rows)
    try:
        _step("Extending data to today…")
        status["steps"]["extend_to_today"] = _run_script(
            "Extend to Today", "scripts/extend_to_today.py")
    except Exception as exc:
        logger.error("  [Extend to Today] ERROR: %s", exc)
        status["steps"]["extend_to_today"] = False

    # Step 4: Forecasting (full retrain — only if extend failed or fresh ingest)
    if not status["steps"].get("extend_to_today"):
        try:
            _step("Running forecast models…")
            status["steps"]["forecasting"] = _run_script(
                "Forecasting", "scripts/run_forecasting.py")
        except Exception as exc:
            logger.error("  [Forecasting] ERROR: %s", exc)
            status["steps"]["forecasting"] = False
    else:
        status["steps"]["forecasting"] = True
        logger.info("  [Forecasting] Skipped — extend_to_today handled it")

    # Step 5: Curve translation (skip if extend_to_today already ran it)
    if not status["steps"].get("extend_to_today"):
        try:
            _step("Translating signals…")
            status["steps"]["curve_translation"] = _run_script(
                "Curve Translation", "scripts/run_curve_translation.py")
        except Exception as exc:
            logger.error("  [Curve Translation] ERROR: %s", exc)
            status["steps"]["curve_translation"] = False
    else:
        status["steps"]["curve_translation"] = True
        logger.info("  [Curve translation] Skipped — extend_to_today handled it")

    # Step 5: AI intelligence
    args = [target_date] if target_date else []
    try:
        _step("Generating AI briefing…")
        status["steps"]["ai_intelligence"] = _run_script(
            "AI Intelligence", "scripts/run_ai_intelligence.py", args=args or None)
    except Exception as exc:
        logger.error("  [AI Intelligence] ERROR: %s", exc)
        status["steps"]["ai_intelligence"] = False

    # Step 6: Report generation
    try:
        _step("Building PDF & Excel report…")
        from src.report_generator import run_report_generation
        logger.info("  [Report Gen] Starting …")
        t0 = time.time()
        report = run_report_generation(target_date=target_date)
        status["steps"]["report_generation"] = True
        status["pdf_path"] = str(report["pdf"])
        status["excel_path"] = str(report["excel"])
        status["target_date"] = report["target_date"]
        logger.info("  [Report Gen] COMPLETE (%.1fs)", time.time() - t0)
    except Exception as exc:
        logger.error("  [Report Gen] FAILED: %s", exc, exc_info=True)
        status["steps"]["report_generation"] = False

    # Step 7: Email delivery
    if config.get("recipient_email") and status.get("pdf_path"):
        try:
            _step("Sending email…")
            from server.mailer import send_daily_report
            logger.info("  [Email] Sending to %s …", config["recipient_email"])
            send_daily_report(
                recipient_email=config["recipient_email"],
                report_paths={
                    "pdf": status["pdf_path"],
                    "excel": status["excel_path"],
                },
                target_date=status["target_date"],
            )
            status["steps"]["email"] = True
            logger.info("  [Email] COMPLETE")
        except Exception as exc:
            logger.error("  [Email] FAILED: %s", exc)
            status["steps"]["email"] = False
    else:
        status["steps"]["email"] = None  # skipped
        if not config.get("recipient_email"):
            logger.info("  [Email] Skipped — no recipient configured")
        else:
            logger.info("  [Email] Skipped — no report generated")

    # Step 8: Cleanup old reports
    try:
        _step("Cleaning up…")
        removed = cleanup_old_reports(config.get("retention_days", 30))
        status["steps"]["cleanup"] = True
        if removed:
            logger.info("  [Cleanup] Removed %d old report(s)", removed)
    except Exception as exc:
        logger.error("  [Cleanup] FAILED: %s", exc)
        status["steps"]["cleanup"] = False

    # Summary
    elapsed = time.time() - run_start
    n_ok = sum(1 for v in status["steps"].values() if v is True)
    n_total = sum(1 for v in status["steps"].values() if v is not None)
    status["success"] = n_ok == n_total and n_total > 0
    status["completed"] = datetime.utcnow().isoformat()
    status["duration_seconds"] = round(elapsed, 1)

    logger.info("=" * 60)
    logger.info("PIPELINE RUN %s  %d/%d steps OK  (%.1fs)",
                "COMPLETE" if status["success"] else "PARTIAL",
                n_ok, n_total, elapsed)
    logger.info("=" * 60)

    # Persist last run status
    status_path = PROJECT_ROOT / "outputs" / "last_run_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    return status


# ── Cleanup ──────────────────────────────────────────────────────────────

def cleanup_old_reports(retention_days: int = 30) -> int:
    """Remove report directories older than retention_days. Returns count removed."""
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    if not reports_dir.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for d in sorted(reports_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            dir_date = datetime.strptime(d.name, "%Y-%m-%d")
            if dir_date < cutoff:
                import shutil
                shutil.rmtree(d)
                removed += 1
                logger.info("  Cleaned up old report: %s", d.name)
        except ValueError:
            continue  # skip non-date directories
    return removed
