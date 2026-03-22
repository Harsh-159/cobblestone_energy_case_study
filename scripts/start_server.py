#!/usr/bin/env python3
"""
Start the DE_LU Power Intelligence server.

Usage:
    python scripts/start_server.py              # default port 5000
    python scripts/start_server.py --port 8080  # custom port
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging():
    """Configure logging to stdout and logs/server.log."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-16s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_dir / "server.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)
    root.addHandler(fh)

    # Quiet noisy loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="DE_LU Power Intelligence Server")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("server")

    # Ensure config exists
    from server.scheduler import load_config
    config = load_config()

    # Initialize Flask app and scheduler
    from server.app import app, init_scheduler
    scheduler = init_scheduler(app)

    # Banner
    logger.info("=" * 60)
    logger.info("  DE_LU POWER INTELLIGENCE SERVER")
    logger.info("=" * 60)
    logger.info("  Host:           %s", args.host)
    logger.info("  Port:           %d", args.port)
    logger.info("  Dashboard:      http://%s:%d/", args.host, args.port)
    logger.info("  Delivery time:  %s %s", config.get("delivery_time", "06:30"),
                config.get("timezone", "Europe/Berlin"))
    logger.info("  Recipient:      %s", config.get("recipient_email") or "(not configured)")
    logger.info("=" * 60)

    try:
        # use_reloader=False prevents APScheduler double-scheduling
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down …")
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped. Goodbye.")


if __name__ == "__main__":
    main()
