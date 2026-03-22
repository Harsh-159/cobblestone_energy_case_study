"""
Email delivery via Gmail SMTP (ssl).

Credentials are read from .env:
    SENDER_EMAIL=your.gmail@gmail.com
    SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx

Never raises on failure — logs the error and returns False.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("mailer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_signal_summary(target_date: str) -> tuple[str, str]:
    """Extract signal label and premium from signal_table for email subject."""
    try:
        import pandas as pd
        sig_path = PROJECT_ROOT / "outputs" / "curve_translation" / "signal_table.csv"
        if not sig_path.exists():
            return "N/A", "N/A"
        df = pd.read_csv(sig_path)
        df["delivery_date"] = pd.to_datetime(df["delivery_date"])
        td = pd.Timestamp(target_date)
        row = df[df["delivery_date"] == td]
        if len(row) == 0:
            row = df.tail(1)
        r = row.iloc[0]
        label = str(r.get("signal_label_month_base", "N/A"))
        premium = r.get("signal_month_base")
        prem_str = f"{premium:+.1f}" if pd.notna(premium) else "N/A"
        return label, prem_str
    except Exception:
        return "N/A", "N/A"


def _build_email_body(target_date: str) -> str:
    """Build inline-styled HTML email body."""
    template_path = PROJECT_ROOT / "server" / "templates" / "email_body.html"
    if not template_path.exists():
        return f"<p>DE_LU Power Intelligence report for {target_date} attached.</p>"

    try:
        import pandas as pd
        from jinja2 import Template

        sig_path = PROJECT_ROOT / "outputs" / "curve_translation" / "signal_table.csv"
        del_path = PROJECT_ROOT / "outputs" / "curve_translation" / "delivery_periods.csv"

        sig_df = pd.read_csv(sig_path)
        sig_df["delivery_date"] = pd.to_datetime(sig_df["delivery_date"])
        td = pd.Timestamp(target_date)
        row = sig_df[sig_df["delivery_date"] == td]
        if len(row) == 0:
            row = sig_df.tail(1)
        r = row.iloc[0]

        del_df = pd.read_csv(del_path)
        del_df["delivery_date"] = pd.to_datetime(del_df["delivery_date"])
        d_row = del_df[del_df["delivery_date"] == td]
        d = d_row.iloc[0] if len(d_row) > 0 else pd.Series()

        def _fv(val, fmt=".2f"):
            if pd.isna(val) or val is None:
                return "N/A"
            return f"{val:{fmt}}"

        sig_label = str(r.get("signal_label_month_base", "N/A"))

        briefing_path = PROJECT_ROOT / "outputs" / "ai_intelligence" / target_date / "briefing.txt"
        ai_snippet = ""
        if briefing_path.exists():
            lines = briefing_path.read_text(encoding="utf-8").strip().splitlines()
            body_lines = [l for l in lines if not l.startswith("===") and not l.startswith("────")]
            ai_snippet = "\n".join(body_lines[:8])

        template_data = {
            "target_date": target_date,
            "sig_label": sig_label,
            "fv_month_base": _fv(r.get("fv_month_base")),
            "curve_proxy_30d": _fv(r.get("curve_proxy_30d")),
            "base_pred": _fv(d.get("base_pred") if len(d) > 0 else None),
            "peak_pred": _fv(d.get("peak_pred") if len(d) > 0 else None),
            "gas_price": _fv(d.get("gas_price") if len(d) > 0 else None),
            "spark_spread_actual": _fv(r.get("spark_spread_actual")),
            "inv_count": sum(1 for f in ["inv_gas_spike", "inv_wind_revision",
                                          "inv_residual_load_swing",
                                          "inv_negative_price_regime",
                                          "inv_high_error_regime"]
                            if r.get(f, False)),
            "any_invalidation": bool(r.get("any_invalidation", False)),
            "ai_snippet": ai_snippet,
        }

        template = Template(template_path.read_text(encoding="utf-8"))
        return template.render(**template_data)

    except Exception as exc:
        logger.warning("Could not render email body: %s — using fallback", exc)
        return f"<p>DE_LU Power Intelligence report for {target_date} attached.</p>"


def send_daily_report(
    recipient_email: str,
    report_paths: dict,
    target_date: str,
) -> bool:
    """
    Send the daily report email with PDF and Excel attachments.

    Uses Gmail SMTP over SSL (port 465). The sender authenticates with a
    Gmail App Password, which allows sending to *any* recipient address.

    Parameters
    ----------
    recipient_email : str
        Recipient email address (any domain).
    report_paths : dict
        Keys: 'pdf', 'excel' — file paths.
    target_date : str
        Report date (YYYY-MM-DD).

    Returns
    -------
    bool
        True if sent successfully.
    """
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_APP_PASSWORD")

    if not sender_email or not sender_password:
        logger.error("Email credentials not configured in .env "
                     "(SENDER_EMAIL / SENDER_APP_PASSWORD)")
        return False

    if not recipient_email:
        logger.error("No recipient email configured")
        return False

    try:
        # Build subject
        label, premium = _get_signal_summary(target_date)
        subject = (f"DE_LU Power Intelligence — {target_date} | "
                   f"{label} M+1 Base {premium} EUR/MWh")

        # Compose message
        msg = MIMEMultipart("mixed")
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # HTML body
        html_body = _build_email_body(target_date)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # Attachments (PDF + Excel only, not PNGs)
        mime_map = {
            "pdf": ("application", "pdf", "daily_report.pdf"),
            "excel": ("application",
                      "vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                      "raw_data.xlsx"),
        }
        for key in ("pdf", "excel"):
            path = report_paths.get(key)
            if not path or not Path(path).exists():
                continue
            maintype, subtype, filename = mime_map[key]
            part = MIMEBase(maintype, subtype)
            part.set_payload(Path(path).read_bytes())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)

        # Send via Gmail SMTP-SSL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        logger.info("Email sent to %s (subject: %s)", recipient_email, subject[:60])
        return True

    except Exception as exc:
        logger.error("Failed to send email: %s", exc, exc_info=True)
        return False
