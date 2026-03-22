"""
Flask web server with APScheduler for automated daily pipeline execution.

Routes:
    GET  /             — Configuration UI
    POST /save_config  — Save settings
    POST /run_now      — Trigger immediate pipeline run
    GET  /status       — JSON status of last run
    GET  /reports       — JSON list of available reports
    GET  /download/<date>/<filetype>  — Download PDF or Excel
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

logger = logging.getLogger("server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "server" / "templates"))

# Global state for run-now tracking
_run_lock = threading.Lock()
_running = False
_current_step = ""  # live progress label


def _is_running() -> bool:
    return _running


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the configuration / dashboard UI."""
    from server.scheduler import load_config
    config = load_config()
    return render_template("index.html", config=config)


@app.route("/save_config", methods=["POST"])
def save_config():
    """Save configuration and reschedule the job."""
    from server.scheduler import update_config

    patch = {
        "recipient_email": request.form.get("recipient_email", "").strip(),
        "delivery_time": request.form.get("delivery_time", "06:30").strip(),
        "timezone": request.form.get("timezone", "Europe/Berlin").strip(),
    }
    update_config(patch)

    # Reschedule the APScheduler job
    reschedule_job()

    return jsonify({"status": "ok", "message": "Configuration saved and job rescheduled."})


@app.route("/run_now", methods=["POST"])
def run_now():
    """Trigger an immediate pipeline run in a background thread."""
    global _running

    if _running:
        return jsonify({"status": "busy", "message": "A pipeline run is already in progress."})

    target_date = request.form.get("target_date") or None

    def _bg_run():
        global _running, _current_step
        with _run_lock:
            _running = True
            _current_step = "Starting…"
            try:
                from server.scheduler import run_full_pipeline
                run_full_pipeline(target_date=target_date, on_step=_set_step)
            except Exception as exc:
                logger.error("Background pipeline run failed: %s", exc, exc_info=True)
            finally:
                _current_step = ""
                _running = False

    def _set_step(step_name):
        global _current_step
        _current_step = step_name

    thread = threading.Thread(target=_bg_run, daemon=True)
    thread.start()
    return jsonify({"status": "ok", "message": "Pipeline run started in background."})


@app.route("/status")
def status():
    """Return JSON status of last pipeline run."""
    status_path = OUTPUTS_DIR / "last_run_status.json"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            data["currently_running"] = _running
            data["current_step"] = _current_step
            return jsonify(data)
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify({
        "success": None,
        "message": "No pipeline run recorded yet.",
        "currently_running": _running,
        "current_step": _current_step,
    })


@app.route("/reports")
def reports():
    """Return JSON list of available reports with signal metadata."""
    if not REPORTS_DIR.exists():
        return jsonify([])

    # Load signal table once for metadata lookup
    signal_meta = {}
    sig_path = OUTPUTS_DIR / "curve_translation" / "signal_table.csv"
    if sig_path.exists():
        try:
            import pandas as pd
            sig_df = pd.read_csv(sig_path)
            sig_df["delivery_date"] = pd.to_datetime(sig_df["delivery_date"])
            for _, row in sig_df.iterrows():
                date_str = str(row["delivery_date"].date())
                signal_meta[date_str] = {
                    "signal_label": str(row.get("signal_label_month_base", "N/A")),
                    "premium": round(float(row["signal_month_base"]), 1) if pd.notna(row.get("signal_month_base")) else None,
                    "fair_value": round(float(row["fv_month_base"]), 1) if pd.notna(row.get("fv_month_base")) else None,
                    "confidence": round(float(row["month_confidence"]), 3) if pd.notna(row.get("month_confidence")) else None,
                    "invalidated": bool(row.get("any_invalidation", False)),
                }
        except Exception as exc:
            logger.warning("Could not load signal metadata: %s", exc)

    # Load delivery periods for base forecast
    del_meta = {}
    del_path = OUTPUTS_DIR / "curve_translation" / "delivery_periods.csv"
    if del_path.exists():
        try:
            import pandas as pd
            del_df = pd.read_csv(del_path)
            del_df["delivery_date"] = pd.to_datetime(del_df["delivery_date"])
            for _, row in del_df.iterrows():
                date_str = str(row["delivery_date"].date())
                del_meta[date_str] = {
                    "base_forecast": round(float(row["base_pred"]), 1) if pd.notna(row.get("base_pred")) else None,
                }
        except Exception:
            pass

    report_list = []
    for d in sorted(REPORTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        try:
            datetime.strptime(d.name, "%Y-%m-%d")
        except ValueError:
            continue

        pdf_path = d / "daily_report.pdf"
        xlsx_path = d / "raw_data.xlsx"

        # Get generated time from the PDF file's mtime
        generated_at = None
        for candidate in (pdf_path, xlsx_path):
            if candidate.exists():
                import time as _time
                mtime = candidate.stat().st_mtime
                generated_at = datetime.fromtimestamp(mtime).strftime("%H:%M")
                break

        entry = {
            "date": d.name,
            "pdf": pdf_path.exists(),
            "excel": xlsx_path.exists(),
            "generated_at": generated_at,
        }
        # Merge signal metadata
        meta = signal_meta.get(d.name, {})
        raw_label = meta.get("signal_label", "N/A")
        # Normalise NO_DATA / N/A to HOLD for display
        if raw_label not in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "INVALIDATED"):
            raw_label = "HOLD"
        entry["signal_label"] = raw_label
        entry["premium"] = meta.get("premium")
        entry["fair_value"] = meta.get("fair_value")
        entry["confidence"] = meta.get("confidence")
        entry["invalidated"] = meta.get("invalidated", False)
        entry["base_forecast"] = del_meta.get(d.name, {}).get("base_forecast")

        report_list.append(entry)
    return jsonify(report_list)


@app.route("/download/<date>/<filetype>")
def download(date: str, filetype: str):
    """Download a report file."""
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    file_map = {
        "pdf": "daily_report.pdf",
        "excel": "raw_data.xlsx",
        "fig1": "fig_signal_dashboard.png",
        "fig2": "fig_forecast_tomorrow.png",
    }
    filename = file_map.get(filetype)
    if not filename:
        return jsonify({"error": f"Unknown filetype: {filetype}"}), 400

    filepath = REPORTS_DIR / date / filename
    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(str(filepath), as_attachment=True, download_name=filename)


# ── APScheduler integration ──────────────────────────────────────────────

def init_scheduler(flask_app: Flask):
    """Initialize APScheduler with the Flask app."""
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from server.scheduler import load_config

    scheduler = BackgroundScheduler()
    flask_app.apscheduler = scheduler

    config = load_config()
    _add_job(scheduler, config)

    scheduler.start()
    logger.info("APScheduler started")
    return scheduler


def _add_job(scheduler, config: dict):
    """Add or replace the daily pipeline job."""
    import pytz
    from apscheduler.triggers.cron import CronTrigger

    time_str = config.get("delivery_time", "06:30")
    tz_str = config.get("timezone", "Europe/Berlin")

    parts = time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0

    try:
        tz = pytz.timezone(tz_str)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.warning("Unknown timezone '%s' — falling back to Europe/Berlin", tz_str)
        tz = pytz.timezone("Europe/Berlin")

    trigger = CronTrigger(hour=hour, minute=minute, timezone=tz)

    from server.scheduler import run_full_pipeline

    # Remove existing job if any
    try:
        scheduler.remove_job("daily_pipeline")
    except Exception:
        pass

    scheduler.add_job(
        run_full_pipeline,
        trigger=trigger,
        id="daily_pipeline",
        name="Daily Pipeline Run",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    logger.info("Scheduled daily pipeline at %s %s", time_str, tz_str)


def reschedule_job():
    """Reschedule the daily job after config change."""
    from server.scheduler import load_config
    config = load_config()
    if hasattr(app, "apscheduler"):
        _add_job(app.apscheduler, config)
