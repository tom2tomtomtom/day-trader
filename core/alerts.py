#!/usr/bin/env python3
"""
ALERT SYSTEM — Generates alerts for high-conviction trades, drawdown warnings,
regime changes, and ML model events.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass

from .db import get_db

logger = logging.getLogger(__name__)


class AlertType:
    HIGH_CONVICTION = "high_conviction"
    DRAWDOWN_WARNING = "drawdown_warning"
    REGIME_CHANGE = "regime_change"
    ML_RETRAIN = "ml_retrain"
    POSITION_EXIT = "position_exit"
    TRADE_EXECUTED = "trade_executed"


class AlertSeverity:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertEngine:
    """Generates and logs alerts to the database."""

    def __init__(self):
        self.db = get_db()
        self._last_regime: Optional[str] = None
        self._last_drawdown_alert: float = 0

    def check_high_conviction(self, symbol: str, confidence: float,
                               action: str, reasons: List[str]) -> Optional[Dict]:
        """Alert on high-conviction trade signals."""
        if confidence < 0.7 or action == "HOLD":
            return None

        severity = AlertSeverity.CRITICAL if confidence >= 0.85 else AlertSeverity.WARNING
        alert = {
            "alert_type": AlertType.HIGH_CONVICTION,
            "severity": severity,
            "title": f"High conviction {action} on {symbol}",
            "message": f"Confidence: {confidence:.0%}. {'; '.join(reasons[:3])}",
            "symbol": symbol,
            "data": {"confidence": confidence, "action": action, "reasons": reasons},
        }
        self.db.log_alert(alert)
        return alert

    def check_drawdown(self, current_drawdown_pct: float, max_allowed: float) -> Optional[Dict]:
        """Alert on drawdown thresholds."""
        # Alert at 50%, 75%, and 100% of max allowed drawdown
        thresholds = [max_allowed * 0.5, max_allowed * 0.75, max_allowed]

        for threshold in thresholds:
            if current_drawdown_pct >= threshold and self._last_drawdown_alert < threshold:
                self._last_drawdown_alert = threshold
                pct_of_max = current_drawdown_pct / max_allowed * 100

                if pct_of_max >= 100:
                    severity = AlertSeverity.CRITICAL
                    title = "DRAWDOWN HALT — Trading suspended"
                elif pct_of_max >= 75:
                    severity = AlertSeverity.CRITICAL
                    title = f"Drawdown at {current_drawdown_pct:.1f}% — approaching halt"
                else:
                    severity = AlertSeverity.WARNING
                    title = f"Drawdown warning: {current_drawdown_pct:.1f}%"

                alert = {
                    "alert_type": AlertType.DRAWDOWN_WARNING,
                    "severity": severity,
                    "title": title,
                    "message": f"Current drawdown: {current_drawdown_pct:.1f}% of {max_allowed:.0f}% max",
                    "data": {"drawdown_pct": current_drawdown_pct, "max_allowed": max_allowed},
                }
                self.db.log_alert(alert)
                return alert
        return None

    def check_regime_change(self, new_regime: str, confidence: float) -> Optional[Dict]:
        """Alert on market regime changes."""
        if self._last_regime is not None and new_regime != self._last_regime:
            severity = AlertSeverity.CRITICAL if new_regime in ("crisis", "CRISIS") else AlertSeverity.WARNING
            alert = {
                "alert_type": AlertType.REGIME_CHANGE,
                "severity": severity,
                "title": f"Regime change: {self._last_regime} -> {new_regime}",
                "message": f"Market regime shifted with {confidence:.0%} confidence",
                "data": {"old_regime": self._last_regime, "new_regime": new_regime, "confidence": confidence},
            }
            self._last_regime = new_regime
            self.db.log_alert(alert)
            return alert

        self._last_regime = new_regime
        return None

    def log_trade_exit(self, symbol: str, pnl_dollars: float, pnl_pct: float,
                        exit_reason: str) -> Dict:
        """Log an alert when a position is closed."""
        severity = AlertSeverity.INFO if pnl_dollars >= 0 else AlertSeverity.WARNING
        emoji = "+" if pnl_dollars >= 0 else ""
        alert = {
            "alert_type": AlertType.POSITION_EXIT,
            "severity": severity,
            "title": f"Closed {symbol}: {emoji}${pnl_dollars:.2f} ({emoji}{pnl_pct:.1f}%)",
            "message": f"Exit reason: {exit_reason}",
            "symbol": symbol,
            "data": {"pnl_dollars": pnl_dollars, "pnl_pct": pnl_pct, "exit_reason": exit_reason},
        }
        self.db.log_alert(alert)
        return alert

    def log_trade_entry(self, symbol: str, direction: str, shares: int,
                         entry_price: float) -> Dict:
        """Log an alert when a new position is opened."""
        alert = {
            "alert_type": AlertType.TRADE_EXECUTED,
            "severity": AlertSeverity.INFO,
            "title": f"Opened {direction.upper()} {symbol}: {shares} shares @ ${entry_price:.2f}",
            "message": f"Position value: ${shares * entry_price:,.2f}",
            "symbol": symbol,
            "data": {"direction": direction, "shares": shares, "entry_price": entry_price},
        }
        self.db.log_alert(alert)
        return alert

    def log_ml_retrain(self, accuracy: float, f1: float, samples: int) -> Dict:
        """Log an alert when ML models are retrained."""
        alert = {
            "alert_type": AlertType.ML_RETRAIN,
            "severity": AlertSeverity.INFO,
            "title": f"ML models retrained: accuracy={accuracy:.1%}, F1={f1:.3f}",
            "message": f"Trained on {samples} trades",
            "data": {"accuracy": accuracy, "f1": f1, "training_samples": samples},
        }
        self.db.log_alert(alert)
        return alert


# Singleton
_engine: Optional[AlertEngine] = None


def get_alert_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine()
    return _engine
