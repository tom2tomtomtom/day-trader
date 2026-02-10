#!/usr/bin/env python3
"""
DATABASE CLIENT — Typed Supabase wrapper for all trading data.

Every module uses this for persistence instead of JSON files.
Falls back gracefully when Supabase is not configured.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .config import get_config

logger = logging.getLogger(__name__)

# Lazy import — supabase may not be installed yet
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    cfg = get_config()
    if not cfg.has_supabase:
        logger.warning("Supabase not configured — DB operations will be no-ops")
        return None

    try:
        from supabase import create_client
        _client = create_client(cfg.supabase.url, cfg.supabase.service_role_key)
        return _client
    except ImportError:
        logger.warning("supabase package not installed — run: pip install supabase")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return None


class TradingDB:
    """Typed wrapper around Supabase for all trading data operations."""

    def __init__(self):
        self._client = _get_client()

    @property
    def connected(self) -> bool:
        return self._client is not None

    # ── Portfolio State ──────────────────────────────────────────────

    def save_portfolio_state(self, state: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "cash": state.get("cash", 0),
                "portfolio_value": state.get("portfolio_value", 0),
                "total_return_pct": state.get("total_return_pct", 0),
                "max_drawdown_pct": state.get("max_drawdown_pct", 0),
                "total_trades": state.get("total_trades", 0),
                "winning_trades": state.get("winning_trades", 0),
                "losing_trades": state.get("losing_trades", 0),
                "win_rate": state.get("win_rate", 0),
                "profit_factor": state.get("profit_factor", 0),
                "portfolio_heat": state.get("portfolio_heat", 0),
                "open_positions": state.get("open_positions", 0),
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("portfolio_state").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"save_portfolio_state: {e}")
            return False

    def get_latest_portfolio(self) -> Optional[Dict]:
        if not self.connected:
            return None
        try:
            resp = (self._client.table("portfolio_state")
                    .select("*")
                    .order("snapshot_at", desc=True)
                    .limit(1)
                    .execute())
            return resp.data[0] if resp.data else None
        except Exception as e:
            logger.error(f"get_latest_portfolio: {e}")
            return None

    # ── Positions ────────────────────────────────────────────────────

    def upsert_position(self, position: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "symbol": position["symbol"],
                "direction": position.get("direction", "long"),
                "entry_price": position["entry_price"],
                "current_price": position.get("current_price", position["entry_price"]),
                "shares": position.get("shares", 0),
                "stop_loss": position.get("stop_loss", 0),
                "take_profit": position.get("take_profit", 0),
                "entry_date": position.get("entry_date"),
                "entry_score": position.get("entry_score", 0),
                "entry_features": json.dumps(position.get("entry_features", {})),
                "unrealized_pnl": position.get("unrealized_pnl", 0),
                "unrealized_pnl_pct": position.get("unrealized_pnl_pct", 0),
                "max_favorable_excursion": position.get("max_favorable_excursion", 0),
                "max_adverse_excursion": position.get("max_adverse_excursion", 0),
                "status": "open",
            }
            self._client.table("positions").upsert(
                row, on_conflict="symbol,status"
            ).execute()
            return True
        except Exception as e:
            logger.error(f"upsert_position: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        if not self.connected:
            return False
        try:
            self._client.table("positions").update(
                {"status": "closed"}
            ).eq("symbol", symbol).eq("status", "open").execute()
            return True
        except Exception as e:
            logger.error(f"close_position: {e}")
            return False

    def get_open_positions(self) -> List[Dict]:
        if not self.connected:
            return []
        try:
            resp = (self._client.table("positions")
                    .select("*")
                    .eq("status", "open")
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.error(f"get_open_positions: {e}")
            return []

    # ── Trades ───────────────────────────────────────────────────────

    def log_trade(self, trade: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "symbol": trade["symbol"],
                "direction": trade.get("direction", "long"),
                "entry_date": trade.get("entry_date"),
                "entry_price": trade["entry_price"],
                "exit_date": trade.get("exit_date"),
                "exit_price": trade.get("exit_price", 0),
                "shares": trade.get("shares", 0),
                "pnl_dollars": trade.get("pnl_dollars", 0),
                "pnl_pct": trade.get("pnl_pct", 0),
                "exit_reason": trade.get("exit_reason", ""),
                "entry_score": trade.get("entry_score", 0),
                "entry_features": json.dumps(trade.get("entry_features", {})),
                "max_favorable_excursion": trade.get("max_favorable_excursion", 0),
                "max_adverse_excursion": trade.get("max_adverse_excursion", 0),
                "regime_at_entry": trade.get("regime_at_entry", ""),
                "is_backtest": trade.get("is_backtest", False),
            }
            self._client.table("trades").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"log_trade: {e}")
            return False

    def get_trades(self, limit: int = 200, include_backtest: bool = False) -> List[Dict]:
        if not self.connected:
            return []
        try:
            q = self._client.table("trades").select("*")
            if not include_backtest:
                q = q.eq("is_backtest", False)
            resp = q.order("exit_date", desc=True).limit(limit).execute()
            return resp.data or []
        except Exception as e:
            logger.error(f"get_trades: {e}")
            return []

    def get_trades_with_features(self, limit: int = 500) -> List[Dict]:
        """Get trades that have entry_features — used for ML training."""
        if not self.connected:
            return []
        try:
            resp = (self._client.table("trades")
                    .select("*")
                    .neq("entry_features", "{}")
                    .order("exit_date", desc=True)
                    .limit(limit)
                    .execute())
            return resp.data or []
        except Exception as e:
            logger.error(f"get_trades_with_features: {e}")
            return []

    # ── Signals ──────────────────────────────────────────────────────

    def log_signal(self, signal: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "symbol": signal["symbol"],
                "action": signal.get("action", "HOLD"),
                "score": signal.get("score", 0),
                "confidence": signal.get("confidence", 0),
                "reasons": json.dumps(signal.get("reasons", [])),
                "regime": signal.get("regime", ""),
                "ml_quality_score": signal.get("ml_quality_score"),
                "ml_size_multiplier": signal.get("ml_size_multiplier"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("signals").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"log_signal: {e}")
            return False

    # ── Market Snapshots ─────────────────────────────────────────────

    def log_market_snapshot(self, snapshot: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "regime": snapshot.get("regime", ""),
                "fear_greed": snapshot.get("fear_greed", 50),
                "vix": snapshot.get("vix", 20),
                "spy_change_pct": snapshot.get("spy_change_pct"),
                "portfolio_value": snapshot.get("portfolio_value"),
                "extra": json.dumps(snapshot.get("extra", {})),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("market_snapshots").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"log_market_snapshot: {e}")
            return False

    # ── ML Models ────────────────────────────────────────────────────

    def save_ml_model(self, model_info: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "model_name": model_info["model_name"],
                "model_type": model_info.get("model_type", "gradient_boosting"),
                "version": model_info.get("version", 1),
                "accuracy": model_info.get("accuracy"),
                "precision_score": model_info.get("precision"),
                "recall": model_info.get("recall"),
                "f1": model_info.get("f1"),
                "feature_importance": json.dumps(model_info.get("feature_importance", {})),
                "training_samples": model_info.get("training_samples", 0),
                "is_active": model_info.get("is_active", True),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            # Deactivate previous versions
            if model_info.get("is_active"):
                self._client.table("ml_models").update(
                    {"is_active": False}
                ).eq("model_name", model_info["model_name"]).execute()
            self._client.table("ml_models").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"save_ml_model: {e}")
            return False

    def get_active_model(self, model_name: str) -> Optional[Dict]:
        if not self.connected:
            return None
        try:
            resp = (self._client.table("ml_models")
                    .select("*")
                    .eq("model_name", model_name)
                    .eq("is_active", True)
                    .limit(1)
                    .execute())
            return resp.data[0] if resp.data else None
        except Exception as e:
            logger.error(f"get_active_model: {e}")
            return None

    # ── Equity Curve ─────────────────────────────────────────────────

    def log_equity_point(self, value: float, cash: float, positions_value: float) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "portfolio_value": value,
                "cash": cash,
                "positions_value": positions_value,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("equity_curve").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"log_equity_point: {e}")
            return False

    def get_equity_curve(self, limit: int = 500) -> List[Dict]:
        if not self.connected:
            return []
        try:
            resp = (self._client.table("equity_curve")
                    .select("*")
                    .order("recorded_at", desc=True)
                    .limit(limit)
                    .execute())
            return list(reversed(resp.data)) if resp.data else []
        except Exception as e:
            logger.error(f"get_equity_curve: {e}")
            return []

    # ── Alerts ───────────────────────────────────────────────────────

    def log_alert(self, alert: Dict) -> bool:
        if not self.connected:
            return False
        try:
            row = {
                "alert_type": alert["alert_type"],
                "severity": alert.get("severity", "info"),
                "title": alert["title"],
                "message": alert.get("message", ""),
                "symbol": alert.get("symbol"),
                "data": json.dumps(alert.get("data", {})),
                "acknowledged": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._client.table("alerts").insert(row).execute()
            return True
        except Exception as e:
            logger.error(f"log_alert: {e}")
            return False

    def get_alerts(self, unacknowledged_only: bool = True, limit: int = 50) -> List[Dict]:
        if not self.connected:
            return []
        try:
            q = self._client.table("alerts").select("*")
            if unacknowledged_only:
                q = q.eq("acknowledged", False)
            resp = q.order("created_at", desc=True).limit(limit).execute()
            return resp.data or []
        except Exception as e:
            logger.error(f"get_alerts: {e}")
            return []

    # ── Watchlist ────────────────────────────────────────────────────

    def get_watchlist(self) -> List[str]:
        if not self.connected:
            return []
        try:
            resp = (self._client.table("watchlist")
                    .select("symbol")
                    .eq("active", True)
                    .execute())
            return [r["symbol"] for r in resp.data] if resp.data else []
        except Exception as e:
            logger.error(f"get_watchlist: {e}")
            return []

    def set_watchlist(self, symbols: List[str]) -> bool:
        if not self.connected:
            return False
        try:
            # Deactivate all
            self._client.table("watchlist").update(
                {"active": False}
            ).eq("active", True).execute()
            # Upsert new list
            rows = [{"symbol": s, "active": True} for s in symbols]
            self._client.table("watchlist").upsert(
                rows, on_conflict="symbol"
            ).execute()
            return True
        except Exception as e:
            logger.error(f"set_watchlist: {e}")
            return False


# Singleton
_db: Optional[TradingDB] = None


def get_db() -> TradingDB:
    global _db
    if _db is None:
        _db = TradingDB()
    return _db
