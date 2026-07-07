"""
alerts.py  — NIDS Alert Manager

This fires email notifications and maintains a persistent in-memory alert log
when the RL agent issues ORANGE or RED severity decisions.

Setup (one-time):
    Set these two environment variables before starting the server:
        ALERT_EMAIL_FROM   your Gmail address
        ALERT_EMAIL_PASS   your Gmail App Password (NOT your account password please)
        ALERT_EMAIL_TO     recipient email (can be the same address)

    Gmail App Password setup:
        1. Go to myaccount.google.com → Security
        2. Enable 2-Step Verification
        3. Search "App passwords" → create one for "Mail"
        4. Use the 16-char code as ALERT_EMAIL_PASS

    If env vars are missing, email is silently skipped — alerts still log locally.
"""

import os
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("nids.alerts")

# ─── Config ─── #

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# Only fire alerts for these levels
ALERT_LEVELS = {"ORANGE", "RED"}

# Keep last N alerts in memory (returned by /alerts endpoint)
MAX_LOG_SIZE = 200


# ─── Alert Record ─── #

@dataclass
class AlertRecord:
    timestamp: str
    alert_level: str
    threat_class: str
    confidence: float
    action: str
    anomaly_score: float
    shap_verdict: Optional[str]
    top_features: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Alert Manager ─── #

class AlertManager:
    """
    Singleton-style class — instantiate once in main.py, reuse across requests.
    Thread-safe for FastAPI's async context (deque ops are GIL-protected).
    """

    def __init__(self):
        self._log: deque[AlertRecord] = deque(maxlen=MAX_LOG_SIZE)
        self._from  = os.getenv("ALERT_EMAIL_FROM", "")
        self._pass  = os.getenv("ALERT_EMAIL_PASS", "")
        self._to    = os.getenv("ALERT_EMAIL_TO", self._from)
        self._email_ready = bool(self._from and self._pass and self._to)

        if self._email_ready:
            print(f"[AlertManager] ✓ Email alerts enabled: {self._to}")
        else:
            print("[AlertManager] ⚠ Email env vars not set. Alerts will only log locally.")

    # ── Public API ── #

    def process(
        self,
        alert_level: str,
        threat_class: str,
        confidence: float,
        action: str,
        anomaly_score: float,
        shap_verdict: Optional[str] = None,
        shap_top_features: Optional[list] = None,
    ) -> Optional[AlertRecord]:
        """
        Called from /analyze after every packet.
        Returns the AlertRecord if an alert was fired, else None.
        """
        if alert_level not in ALERT_LEVELS:
            return None

        record = AlertRecord(
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            alert_level=alert_level,
            threat_class=threat_class,
            confidence=round(confidence, 4),
            action=action,
            anomaly_score=round(anomaly_score, 4),
            shap_verdict=shap_verdict,
            top_features=[
                {"feature": f.feature, "shap_contribution": round(f.shap_contribution, 4)}
                for f in (shap_top_features or [])
            ],
        )

        self._log.appendleft(record) 
        logger.warning(f"[ALERT {alert_level}] {threat_class} | action={action} | confidence={confidence:.2%}")

        if self._email_ready:
            self._send_email(record)

        return record

    def get_recent(self, limit: int = 50) -> list:
        """Returns the most recent alerts as dicts (for the /alerts endpoint)."""
        return [r.to_dict() for r in list(self._log)[:limit]]

    def get_counts(self) -> dict:
        """Summary counts by alert level used by dashboard stats."""
        counts = {"RED": 0, "ORANGE": 0, "total": 0}
        for r in self._log:
            counts[r.alert_level] = counts.get(r.alert_level, 0) + 1
            counts["total"] += 1
        return counts

    # ── Email ── #

    def _send_email(self, record: AlertRecord) -> None:
        """Sends a formatted HTML alert email. Fails silently on error."""
        try:
            subject = f"[NIDS {record.alert_level}] {record.threat_class} detected — action: {record.action}"
            body = self._build_email_body(record)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = self._from
            msg["To"]      = self._to
            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self._from, self._pass)
                server.sendmail(self._from, self._to, msg.as_string())

            logger.info(f"[AlertManager] Email sent for {record.threat_class} alert.")

        except Exception as e:
            # This will never crash the API because of an email failure
            logger.error(f"[AlertManager] Email failed: {e}")

    def _build_email_body(self, r: AlertRecord) -> str:
        header_color = "#c0392b" if r.alert_level == "RED" else "#d35400"
        pos_color = "#ef4444"
        neg_color = "#22c55e"

        shap_rows = ""
        if r.top_features:
            max_val = max(abs(f["shap_contribution"]) for f in r.top_features) or 1
            for f in r.top_features:
                val = f["shap_contribution"]
                bar_pct = int(abs(val) / max_val * 90)
                bar_color = pos_color if val > 0 else neg_color
                sign = "+" if val > 0 else ""
                shap_rows += f"""
                <tr>
                <td style="padding:5px 0;color:#d1d5db;font-size:12px">{f['feature']}</td>
                <td style="padding:5px 0;width:90px">
                    <div style="width:80px;height:4px;background:#1f2937;border-radius:2px;overflow:hidden">
                    <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:2px"></div>
                    </div>
                </td>
                <td style="padding:5px 0;color:{bar_color};font-size:11px;text-align:right;min-width:52px">{sign}{val:.4f}</td>
                </tr>"""

        shap_section = ""
        if shap_rows:
            shap_section = f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:6px;padding:14px;margin-bottom:16px">
            <p style="margin:0 0 10px;color:#6b7280;font-size:10px;letter-spacing:1px">TOP SHAP FEATURES</p>
            <table style="width:100%;border-collapse:collapse">{shap_rows}</table>
            </div>"""

        verdict_section = ""
        if r.shap_verdict:
            verdict_section = f"""
            <div style="background:#111827;border-left:3px solid {header_color};padding:10px 14px;margin-bottom:16px;border-radius:0 6px 6px 0">
            <p style="margin:0;color:#9ca3af;font-size:12px">{r.shap_verdict}</p>
            </div>"""

        return f"""
        <div style="font-family:'Courier New',monospace;background:#0b1120;max-width:600px;border-radius:8px;overflow:hidden">

        <div style="background:{header_color};padding:18px 24px;display:flex;align-items:center;gap:12px">
            <div style="width:36px;height:36px;border-radius:50%;background:rgba(255,255,255,0.15);display:flex;align-items:center;justify-content:center;font-size:18px">&#128721;</div>
            <div>
            <p style="margin:0;color:#fff;font-weight:bold;font-size:15px;letter-spacing:0.5px">SECURITY ALERT — {r.alert_level}</p>
            <p style="margin:0;color:rgba(255,255,255,0.65);font-size:11px">Autonomous NIDS &nbsp;•&nbsp; {r.timestamp}</p>
            </div>
            <div style="margin-left:auto;background:rgba(255,255,255,0.15);border-radius:4px;padding:4px 10px">
            <p style="margin:0;color:#fff;font-size:11px;letter-spacing:1px">{r.threat_class}</p>
            </div>
        </div>

        <div style="padding:24px">

            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:20px">
            <div style="background:#111827;border:1px solid #1f2937;border-radius:6px;padding:12px">
                <p style="margin:0 0 4px;color:#6b7280;font-size:10px;letter-spacing:1px">TIER 1 — LGBM</p>
                <p style="margin:0;color:#f87171;font-size:13px;font-weight:bold">{r.threat_class}</p>
                <p style="margin:2px 0 0;color:#9ca3af;font-size:11px">conf: {r.confidence:.1%}</p>
            </div>
            <div style="background:#111827;border:1px solid #1f2937;border-radius:6px;padding:12px">
                <p style="margin:0 0 4px;color:#6b7280;font-size:10px;letter-spacing:1px">TIER 2 — IFOREST</p>
                <p style="margin:0;color:#fbbf24;font-size:13px;font-weight:bold">{'ANOMALOUS' if r.anomaly_score < 0 else 'NORMAL'}</p>
                <p style="margin:2px 0 0;color:#9ca3af;font-size:11px">score: {r.anomaly_score:.4f}</p>
            </div>
            <div style="background:#111827;border:1px solid #1f2937;border-radius:6px;padding:12px">
                <p style="margin:0 0 4px;color:#6b7280;font-size:10px;letter-spacing:1px">TIER 3 — PPO RL</p>
                <p style="margin:0;color:{header_color};font-size:13px;font-weight:bold">{r.action}</p>
                <p style="margin:2px 0 0;color:#9ca3af;font-size:11px">directive issued</p>
            </div>
            </div>

            {verdict_section}
            {shap_section}

            <div style="border-top:1px solid #1f2937;padding-top:12px;display:flex;justify-content:space-between">
            <p style="margin:0;color:#4b5563;font-size:10px">Three-Tier Autonomous NIDS &nbsp;|&nbsp; LightGBM + IForest + PPO</p>
            <p style="margin:0;color:#4b5563;font-size:10px">Do not reply</p>
            </div>

        </div>
        </div>"""