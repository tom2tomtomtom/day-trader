#!/usr/bin/env python3
"""
SIGNAL EVALUATOR — Measures signal accuracy against trade outcomes.

Computes per-symbol and per-regime accuracy by matching signals to trades.
Results stored in signal_evaluations table for dashboard visibility.

Required Supabase table: signal_evaluations
Columns: id (uuid auto), symbol (text), regime (text), action (text),
         total_signals (int4), profitable_signals (int4), accuracy (float8),
         avg_pnl_pct (float8), evaluation_period_start (timestamptz),
         evaluation_period_end (timestamptz), created_at (timestamptz default now())
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from .db import get_db

logger = logging.getLogger(__name__)


class SignalEvaluator:
    """Evaluates historical signal accuracy against trade outcomes."""

    def __init__(self):
        self.db = get_db()

    def evaluate(self, days: int = 30) -> List[Dict]:
        """
        Evaluate signal accuracy over the given period.

        Matches signals to trades by symbol and time window (signal must
        precede trade entry by < 1 hour). Computes accuracy per symbol,
        per regime, and per action.
        """
        if not self.db.connected:
            logger.warning("DB not connected — skipping signal evaluation")
            return []

        # Fetch signals and trades
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            signals_resp = (self.db._client.table("signals")
                           .select("*")
                           .gte("created_at", cutoff)
                           .order("created_at", desc=False)
                           .limit(5000)
                           .execute())
            signals = signals_resp.data or []

            trades_resp = (self.db._client.table("trades")
                          .select("*")
                          .gte("entry_date", cutoff)
                          .order("entry_date", desc=False)
                          .limit(2000)
                          .execute())
            trades = trades_resp.data or []
        except Exception as e:
            logger.error(f"Failed to fetch data for evaluation: {e}")
            return []

        if not signals or not trades:
            logger.info(f"Not enough data for evaluation: {len(signals)} signals, {len(trades)} trades")
            return []

        # Build trade lookup: symbol -> list of trades
        trade_lookup: Dict[str, List[Dict]] = {}
        for t in trades:
            sym = t.get("symbol", "")
            trade_lookup.setdefault(sym, []).append(t)

        # Match signals to trade outcomes
        evaluations: Dict[str, Dict] = {}  # key = "symbol|regime|action"

        for sig in signals:
            symbol = sig.get("symbol", "")
            action = sig.get("action", "HOLD")
            regime = sig.get("regime", "unknown") or "unknown"
            sig_time = sig.get("created_at", "")

            if action in ("HOLD", "BLOCKED_BY_ML"):
                continue

            # Find matching trade (same symbol, entry within 1 hour of signal)
            matched_trade = self._find_matching_trade(
                symbol, sig_time, trade_lookup.get(symbol, [])
            )

            # Group key
            key = f"{symbol}|{regime}|{action}"
            if key not in evaluations:
                evaluations[key] = {
                    "symbol": symbol,
                    "regime": regime,
                    "action": action,
                    "total_signals": 0,
                    "profitable_signals": 0,
                    "total_pnl_pct": 0.0,
                }

            evaluations[key]["total_signals"] += 1
            if matched_trade:
                pnl = float(matched_trade.get("pnl_pct", 0))
                if pnl > 0:
                    evaluations[key]["profitable_signals"] += 1
                evaluations[key]["total_pnl_pct"] += pnl

        # Compute accuracy and store
        results = []
        now = datetime.now(timezone.utc).isoformat()
        period_start = cutoff
        period_end = now

        for key, ev in evaluations.items():
            total = ev["total_signals"]
            if total == 0:
                continue

            accuracy = ev["profitable_signals"] / total
            avg_pnl = ev["total_pnl_pct"] / total

            result = {
                "symbol": ev["symbol"],
                "regime": ev["regime"],
                "action": ev["action"],
                "total_signals": total,
                "profitable_signals": ev["profitable_signals"],
                "accuracy": round(accuracy, 4),
                "avg_pnl_pct": round(avg_pnl, 4),
                "evaluation_period_start": period_start,
                "evaluation_period_end": period_end,
            }
            results.append(result)

            # Store to DB
            self._store_evaluation(result)

        logger.info(
            f"Signal evaluation complete: {len(results)} groups evaluated "
            f"from {len(signals)} signals and {len(trades)} trades"
        )
        return results

    def _find_matching_trade(self, symbol: str, signal_time: str,
                              trades: List[Dict]) -> Optional[Dict]:
        """Find a trade that matches this signal (same symbol, entry within 1 hour)."""
        try:
            sig_dt = datetime.fromisoformat(signal_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

        for trade in trades:
            entry_time = trade.get("entry_date", "")
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            # Signal should precede trade entry by 0-60 minutes
            diff = (entry_dt - sig_dt).total_seconds()
            if 0 <= diff <= 3600:
                return trade

        return None

    def _store_evaluation(self, evaluation: Dict) -> bool:
        """Store evaluation result in signal_evaluations table."""
        if not self.db.connected:
            return False
        try:
            self.db._client.table("signal_evaluations").insert(evaluation).execute()
            return True
        except Exception as e:
            logger.warning(f"Failed to store signal evaluation: {e}")
            return False

    def get_evaluations(self, days: int = 30) -> List[Dict]:
        """Retrieve recent signal evaluations."""
        if not self.db.connected:
            return []
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            resp = (self.db._client.table("signal_evaluations")
                    .select("*")
                    .gte("created_at", cutoff)
                    .order("created_at", desc=True)
                    .limit(500)
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.error(f"get_signal_evaluations: {e}")
            return []
