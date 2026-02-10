#!/usr/bin/env python3
"""
PAPER TRADER - Forward-Test Strategies in Real-Time

Simulates real trading without risking capital. Connects the TradingModel
to live data and tracks a virtual portfolio with full P&L tracking.

Features:
- Virtual portfolio with cash and positions
- Real-time signal generation from TradingModel
- Automatic stop-loss and take-profit execution
- Partial exits (50% at first target, trail remainder)
- MFE/MAE tracking for ML learning
- Full trade log with performance metrics
- DB persistence via Supabase (falls back to JSON)
- Feature vector logging at entry for ML training

Usage:
    python3 -m core.paper_trader                    # Scan default universe
    python3 -m core.paper_trader --symbol AAPL      # Single symbol
    python3 -m core.paper_trader --portfolio 50000   # Custom starting capital
"""

import json
import logging
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

from .trading_model import TradingModel, TradeSignal, MEME_ASSETS
from .feature_engine import FeatureEngine, FeatureVector
from .execution_engine import ExecutionEngine, ManagedPosition, ExitReason
from .db import get_db
from .alerts import get_alert_engine

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
PAPER_STATE = BASE_DIR / "paper_portfolio.json"
PAPER_LOG = BASE_DIR / "paper_trades.json"


@dataclass
class PaperPosition:
    """An open paper trading position"""
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    shares: int
    stop_loss: float
    take_profit: float
    entry_date: str
    entry_score: int
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0
    # ML features
    entry_features: Dict = field(default_factory=dict)
    # MFE / MAE tracking
    max_favorable_excursion: float = 0  # Best unrealized % gain
    max_adverse_excursion: float = 0    # Worst unrealized % loss
    # Partial exit tracking
    partial_exited: bool = False
    original_shares: int = 0
    # ATR trailing stop
    atr_at_entry: float = 0
    trailing_stop: float = 0  # Dynamic trailing stop level

    def update(self, current_price: float):
        self.current_price = current_price
        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.shares
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.shares
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

        # Track MFE/MAE
        if self.unrealized_pnl_pct > self.max_favorable_excursion:
            self.max_favorable_excursion = self.unrealized_pnl_pct
        if self.unrealized_pnl_pct < -self.max_adverse_excursion:
            self.max_adverse_excursion = abs(self.unrealized_pnl_pct)

        # Update ATR trailing stop (only ratchets in favorable direction)
        if self.atr_at_entry > 0 and self.trailing_stop > 0:
            if self.direction == "long":
                new_trail = current_price - self.atr_at_entry * self._trail_multiplier()
                if new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail
            else:
                new_trail = current_price + self.atr_at_entry * self._trail_multiplier()
                if new_trail < self.trailing_stop:
                    self.trailing_stop = new_trail

    def _trail_multiplier(self) -> float:
        """ATR multiplier: tighter after partial exit (riding with house money)."""
        return 1.5 if self.partial_exited else 2.5


@dataclass
class PaperTrade:
    """A completed paper trade"""
    symbol: str
    direction: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    entry_score: int
    entry_features: Dict = field(default_factory=dict)
    max_favorable_excursion: float = 0
    max_adverse_excursion: float = 0


@dataclass
class PaperPortfolio:
    """Complete paper trading portfolio state"""
    timestamp: str
    initial_capital: float
    cash: float
    portfolio_value: float
    total_return_pct: float
    positions: List[PaperPosition]
    closed_trades: List[PaperTrade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    daily_pnl: float
    equity_history: List[Dict]


class PaperTrader:
    """
    Paper trading engine that simulates real trading with the TradingModel.

    Uses the unified ExecutionEngine for position management so that
    exit logic (trailing stops, partial exits) matches the backtester.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperTrade] = []
        self.equity_history: List[Dict] = []
        self.peak_value = initial_capital

        self.model = TradingModel(portfolio_value=initial_capital)
        self.feature_engine = FeatureEngine()
        self.engine = ExecutionEngine(slippage_pct=0.0, commission=0.0)
        self.db = get_db()
        self.alerts = get_alert_engine()

        # Try to load existing state
        self._load_state()

    def analyze_and_trade(self, symbols: List[str],
                          price_data: Dict[str, Dict],
                          regime_state=None,
                          fear_greed_data: Optional[Dict] = None) -> PaperPortfolio:
        """
        Main loop: analyze all symbols and execute paper trades.

        Args:
            symbols: List of symbols to analyze
            price_data: Dict mapping symbol -> {prices, highs, lows, volumes}
            regime_state: Optional regime detection result
            fear_greed_data: Optional fear & greed index data
        """
        now = datetime.now(timezone.utc).isoformat()

        for symbol in symbols:
            data = price_data.get(symbol, {})
            prices = data.get("prices", [])
            highs = data.get("highs")
            lows = data.get("lows")
            volumes = data.get("volumes")

            if not prices or len(prices) < 20:
                continue

            current_price = prices[-1]

            # Generate signal
            signal = self.model.generate_signal(
                symbol, prices, highs, lows, volumes
            )

            if signal is None:
                # Still update existing position prices
                if symbol in self.positions:
                    self.positions[symbol].update(current_price)
                continue

            # Check existing position
            if symbol in self.positions:
                self._manage_position(symbol, signal, current_price, now)
            else:
                self._check_entry(symbol, signal, now,
                                  prices=prices, highs=highs, lows=lows,
                                  volumes=volumes, regime_state=regime_state,
                                  fear_greed_data=fear_greed_data)

            # Update position price if still open
            if symbol in self.positions:
                self.positions[symbol].update(current_price)
                # Sync position to DB
                self._sync_position_to_db(symbol)

        # Update portfolio value and equity curve
        portfolio = self._build_portfolio_state(now)

        # Save state (JSON fallback + DB)
        self._save_state()
        self._sync_portfolio_to_db(portfolio)

        return portfolio

    def _manage_position(self, symbol: str, signal: TradeSignal,
                         current_price: float, timestamp: str):
        """Check if position should be closed or partially exited.

        Delegates to ExecutionEngine for consistent exit logic
        matching the backtester.
        """
        pos = self.positions[symbol]

        # Build a ManagedPosition for the engine
        managed = ManagedPosition(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            shares=pos.shares,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            entry_date=pos.entry_date,
            entry_score=pos.entry_score,
            atr_at_entry=pos.atr_at_entry,
            trailing_stop=pos.trailing_stop,
            partial_exited=pos.partial_exited,
            original_shares=pos.original_shares,
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
        )

        exit_signal = self.engine.check_exit(managed, current_price, signal.score)
        if exit_signal is None:
            return

        if exit_signal.is_partial:
            self._partial_exit(symbol, exit_signal.exit_price,
                               exit_signal.exit_shares, timestamp)
        else:
            self._close_position(symbol, exit_signal.exit_price,
                                 exit_signal.reason.value, signal.score, timestamp)

    def _partial_exit(self, symbol: str, exit_price: float,
                      exit_shares: int, timestamp: str):
        """Exit portion of position and move stop to breakeven."""
        pos = self.positions[symbol]

        if exit_shares <= 0:
            return

        # Calculate P&L on exited portion
        pnl, pnl_pct = self.engine.calculate_pnl(
            ManagedPosition(
                symbol=pos.symbol, direction=pos.direction,
                entry_price=pos.entry_price, current_price=exit_price,
                shares=exit_shares, stop_loss=0, take_profit=0,
                entry_date=pos.entry_date, entry_score=pos.entry_score,
            ),
            exit_price, exit_shares,
        )

        self.cash += exit_shares * exit_price

        # Log partial exit as a trade
        trade = PaperTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=timestamp,
            exit_price=exit_price,
            shares=exit_shares,
            pnl_dollars=pnl,
            pnl_pct=round(pnl_pct, 2),
            exit_reason="partial_target",
            entry_score=pos.entry_score,
            entry_features=pos.entry_features,
            max_favorable_excursion=round(pos.max_favorable_excursion, 2),
            max_adverse_excursion=round(pos.max_adverse_excursion, 2),
        )
        self.closed_trades.append(trade)
        self._log_trade_to_db(trade)

        # Update position: fewer shares, move stop to breakeven
        remain_shares = pos.shares - exit_shares
        pos.shares = remain_shares
        pos.stop_loss = pos.entry_price  # Breakeven stop
        pos.partial_exited = True

        logger.info(f"Partial exit {symbol}: {exit_shares} shares @ ${exit_price:.2f} "
                     f"(+{pnl_pct:.1f}%), trailing {remain_shares} shares")

    def _check_entry(self, symbol: str, signal: TradeSignal, timestamp: str,
                     prices=None, highs=None, lows=None, volumes=None,
                     regime_state=None, fear_greed_data=None):
        """Check if we should enter a new position"""
        if signal.direction == "flat":
            return

        # Check cash available
        position_cost = signal.position_shares * signal.entry_price
        if position_cost > self.cash * 0.95:  # Keep 5% cash buffer
            return
        if signal.position_shares <= 0:
            return

        # Compute feature vector for ML training
        features = self.feature_engine.compute(
            symbol=symbol,
            prices=prices or [],
            highs=highs,
            lows=lows,
            volumes=volumes,
            indicators=signal.indicators if signal else None,
            regime_state=regime_state,
            fear_greed_data=fear_greed_data,
        )

        # ML quality gate — block low-quality entries
        from .ml_pipeline import get_ml_pipeline
        from .config import get_feature_flags
        ml_prediction = get_ml_pipeline().predict(features)
        flags = get_feature_flags()

        if flags.ml_gate_enabled and ml_prediction.quality_score < 0.35:
            # Log blocked trade as signal for analysis
            self.db.log_signal({
                "symbol": symbol,
                "action": "BLOCKED_BY_ML",
                "score": signal.score,
                "confidence": ml_prediction.quality_score,
                "reasons": [f"ML quality {ml_prediction.quality_score:.3f} < 0.35 threshold"],
                "regime": "",
                "ml_quality_score": ml_prediction.quality_score,
                "ml_size_multiplier": ml_prediction.size_multiplier,
            })
            logger.info(
                f"BLOCKED: {symbol} entry blocked by ML gate "
                f"(quality={ml_prediction.quality_score:.3f} < 0.35)"
            )
            return

        # Pre-trade risk check
        if flags.pre_trade_risk_enabled:
            from .risk_engine import PreTradeRiskCheck, RiskEngine
            risk_check = PreTradeRiskCheck(RiskEngine())
            daily_pnl_pct = 0.0
            if self.equity_history:
                prev_val = self.equity_history[-1].get("value", self.initial_capital)
                portfolio_val = self.cash + sum(
                    p.shares * p.current_price for p in self.positions.values()
                )
                daily_pnl_pct = (portfolio_val - prev_val) / prev_val * 100 if prev_val else 0.0

            result = risk_check.validate(
                symbol=symbol,
                direction=signal.direction,
                position_value=position_cost,
                portfolio_value=self.cash + sum(
                    p.shares * p.current_price for p in self.positions.values()
                ),
                open_positions=self.positions,
                daily_pnl_pct=daily_pnl_pct,
                regime="",
            )
            if not result.can_trade:
                self.db.log_signal({
                    "symbol": symbol,
                    "action": "BLOCKED_BY_RISK",
                    "score": signal.score,
                    "confidence": 0,
                    "reasons": [result.reason],
                    "regime": "",
                })
                logger.info(f"BLOCKED: {symbol} blocked by risk check: {result.reason}")
                return

        # Enter position
        self.cash -= position_cost
        shares = signal.position_shares

        # ATR trailing stop: 2.5x ATR from entry, adapts to regime
        atr = signal.indicators.atr_14 if signal.indicators else 0
        if atr > 0 and signal.direction == "long":
            trailing = signal.entry_price - atr * 2.5
        elif atr > 0 and signal.direction == "short":
            trailing = signal.entry_price + atr * 2.5
        else:
            trailing = 0

        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            shares=shares,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_date=timestamp,
            entry_score=signal.score,
            entry_features=features.to_dict(),
            original_shares=shares,
            atr_at_entry=atr,
            trailing_stop=trailing,
        )

        # Log entry alert
        self.alerts.log_trade_entry(symbol, signal.direction, shares, signal.entry_price)

        # Sync to DB
        self._sync_position_to_db(symbol)

    def _close_position(self, symbol: str, exit_price: float,
                        reason: str, exit_score: int, timestamp: str):
        """Close a position and log the trade"""
        pos = self.positions.pop(symbol)

        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.shares
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl = (pos.entry_price - exit_price) * pos.shares
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        # Return position value to cash
        self.cash += pos.shares * exit_price

        trade = PaperTrade(
            symbol=symbol,
            direction=pos.direction,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=timestamp,
            exit_price=exit_price,
            shares=pos.shares,
            pnl_dollars=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=reason,
            entry_score=pos.entry_score,
            entry_features=pos.entry_features,
            max_favorable_excursion=round(pos.max_favorable_excursion, 2),
            max_adverse_excursion=round(pos.max_adverse_excursion, 2),
        )
        self.closed_trades.append(trade)

        # Log to DB
        self._log_trade_to_db(trade)

        # Close position in DB
        self.db.close_position(symbol)

        # Alert
        self.alerts.log_trade_exit(symbol, pnl, pnl_pct, reason)

    def _log_trade_to_db(self, trade: PaperTrade):
        """Log a completed trade to Supabase."""
        self.db.log_trade({
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_date": trade.entry_date,
            "entry_price": trade.entry_price,
            "exit_date": trade.exit_date,
            "exit_price": trade.exit_price,
            "shares": trade.shares,
            "pnl_dollars": trade.pnl_dollars,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason,
            "entry_score": trade.entry_score,
            "entry_features": trade.entry_features,
            "max_favorable_excursion": trade.max_favorable_excursion,
            "max_adverse_excursion": trade.max_adverse_excursion,
        })

    def _sync_position_to_db(self, symbol: str):
        """Sync an open position to Supabase."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        self.db.upsert_position({
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "shares": pos.shares,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "entry_date": pos.entry_date,
            "entry_score": pos.entry_score,
            "entry_features": pos.entry_features,
            "unrealized_pnl": pos.unrealized_pnl,
            "unrealized_pnl_pct": pos.unrealized_pnl_pct,
            "max_favorable_excursion": pos.max_favorable_excursion,
            "max_adverse_excursion": pos.max_adverse_excursion,
        })

    def _sync_portfolio_to_db(self, portfolio: PaperPortfolio):
        """Sync portfolio state to Supabase."""
        self.db.save_portfolio_state({
            "cash": portfolio.cash,
            "portfolio_value": portfolio.portfolio_value,
            "total_return_pct": portfolio.total_return_pct,
            "max_drawdown_pct": portfolio.max_drawdown_pct,
            "total_trades": portfolio.total_trades,
            "winning_trades": portfolio.winning_trades,
            "losing_trades": portfolio.losing_trades,
            "win_rate": portfolio.win_rate,
            "profit_factor": portfolio.profit_factor,
            "portfolio_heat": 0,  # Calculated elsewhere
            "open_positions": len(portfolio.positions),
        })
        self.db.log_equity_point(
            value=portfolio.portfolio_value,
            cash=portfolio.cash,
            positions_value=portfolio.portfolio_value - portfolio.cash,
        )

    def _build_portfolio_state(self, timestamp: str) -> PaperPortfolio:
        """Build current portfolio state"""
        positions_value = sum(
            p.shares * p.current_price for p in self.positions.values()
        )
        portfolio_value = self.cash + positions_value
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100

        # Track peak for drawdown
        self.peak_value = max(self.peak_value, portfolio_value)
        max_dd = (self.peak_value - portfolio_value) / self.peak_value * 100

        # Check drawdown alerts
        from .config import get_config
        cfg = get_config()
        self.alerts.check_drawdown(max_dd, cfg.trading.max_drawdown_halt * 100)

        # Trade stats
        winners = [t for t in self.closed_trades if t.pnl_dollars > 0]
        losers = [t for t in self.closed_trades if t.pnl_dollars <= 0]
        total_trades = len(self.closed_trades)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        avg_win = float(np.mean([t.pnl_pct for t in winners])) if winners else 0
        avg_loss = float(np.mean([t.pnl_pct for t in losers])) if losers else 0
        gross_profit = sum(t.pnl_dollars for t in winners)
        gross_loss = abs(sum(t.pnl_dollars for t in losers)) if losers else 0.0001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Daily P&L
        prev_value = self.equity_history[-1]["value"] if self.equity_history else self.initial_capital
        daily_pnl = portfolio_value - prev_value

        # Record equity
        self.equity_history.append({
            "date": timestamp,
            "value": round(portfolio_value, 2),
            "cash": round(self.cash, 2),
            "positions": round(positions_value, 2),
        })
        # Keep last 500 points
        self.equity_history = self.equity_history[-500:]

        return PaperPortfolio(
            timestamp=timestamp,
            initial_capital=self.initial_capital,
            cash=round(self.cash, 2),
            portfolio_value=round(portfolio_value, 2),
            total_return_pct=round(total_return, 2),
            positions=list(self.positions.values()),
            closed_trades=self.closed_trades[-50:],  # Last 50 trades
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            profit_factor=round(pf, 2),
            max_drawdown_pct=round(max_dd, 2),
            daily_pnl=round(daily_pnl, 2),
            equity_history=self.equity_history[-100:],
        )

    def _save_state(self):
        """Save state to disk for persistence across runs"""
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "peak_value": self.peak_value,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades[-200:]],
            "equity_history": self.equity_history[-500:],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        PAPER_STATE.write_text(json.dumps(state, indent=2, default=str))

    def _load_state(self):
        """Load state from Supabase DB first, then fall back to JSON file."""
        # Try Supabase first (works on Railway where there's no filesystem)
        if self.db.connected:
            try:
                portfolio = self.db.get_latest_portfolio()
                if portfolio:
                    self.cash = portfolio.get("cash", self.initial_capital)
                    self.peak_value = portfolio.get("portfolio_value", self.initial_capital)
                    logger.info(f"Loaded portfolio from DB: ${self.cash:,.2f} cash")

                # Restore open positions from DB
                db_positions = self.db.get_open_positions()
                for pos in db_positions:
                    features = pos.get("entry_features", "{}")
                    if isinstance(features, str):
                        features = json.loads(features) if features else {}
                    self.positions[pos["symbol"]] = PaperPosition(
                        symbol=pos["symbol"],
                        direction=pos.get("direction", "long"),
                        entry_price=pos.get("entry_price", 0),
                        current_price=pos.get("current_price", pos.get("entry_price", 0)),
                        shares=pos.get("shares", 0),
                        stop_loss=pos.get("stop_loss", 0),
                        take_profit=pos.get("take_profit", 0),
                        entry_date=pos.get("entry_date", ""),
                        entry_score=pos.get("entry_score", 0),
                        entry_features=features,
                        unrealized_pnl=pos.get("unrealized_pnl", 0),
                        unrealized_pnl_pct=pos.get("unrealized_pnl_pct", 0),
                        max_favorable_excursion=pos.get("max_favorable_excursion", 0),
                        max_adverse_excursion=pos.get("max_adverse_excursion", 0),
                        trailing_stop=pos.get("stop_loss", 0),
                        atr_at_entry=0,
                        partial_exited=False,
                    )
                if db_positions:
                    logger.info(f"Loaded {len(db_positions)} open positions from DB")

                # Restore recent trades from DB
                db_trades = self.db.get_trades(limit=200)
                for t in db_trades:
                    features = t.get("entry_features", "{}")
                    if isinstance(features, str):
                        features = json.loads(features) if features else {}
                    self.closed_trades.append(PaperTrade(
                        symbol=t["symbol"],
                        direction=t.get("direction", "long"),
                        entry_date=t.get("entry_date", ""),
                        entry_price=t.get("entry_price", 0),
                        exit_date=t.get("exit_date", ""),
                        exit_price=t.get("exit_price", 0),
                        shares=t.get("shares", 0),
                        pnl_dollars=t.get("pnl_dollars", 0),
                        pnl_pct=t.get("pnl_pct", 0),
                        exit_reason=t.get("exit_reason", ""),
                        entry_score=t.get("entry_score", 0),
                        entry_features=features,
                        max_favorable_excursion=t.get("max_favorable_excursion", 0),
                        max_adverse_excursion=t.get("max_adverse_excursion", 0),
                    ))
                if db_trades:
                    logger.info(f"Loaded {len(db_trades)} historical trades from DB")
                return
            except Exception as e:
                logger.warning(f"Could not load state from DB: {e}")

        # Fall back to JSON file (local dev)
        if not PAPER_STATE.exists():
            return

        try:
            state = json.loads(PAPER_STATE.read_text())
            self.cash = state.get("cash", self.initial_capital)
            self.peak_value = state.get("peak_value", self.initial_capital)
            self.equity_history = state.get("equity_history", [])

            # Restore positions
            for sym, pos_data in state.get("positions", {}).items():
                known = {f.name for f in PaperPosition.__dataclass_fields__.values()}
                filtered = {k: v for k, v in pos_data.items() if k in known}
                self.positions[sym] = PaperPosition(**filtered)

            # Restore trades
            for trade_data in state.get("closed_trades", []):
                known = {f.name for f in PaperTrade.__dataclass_fields__.values()}
                filtered = {k: v for k, v in trade_data.items() if k in known}
                self.closed_trades.append(PaperTrade(**filtered))

        except Exception as e:
            logger.warning(f"Could not load paper state: {e} — starting fresh")

    def reset(self):
        """Reset paper portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        self.peak_value = self.initial_capital
        if PAPER_STATE.exists():
            PAPER_STATE.unlink()


def run_paper_trading(symbols: List[str] = None,
                      portfolio_value: float = 100000) -> PaperPortfolio:
    """Convenience function: fetch data and run paper trader"""
    import yfinance as yf

    if symbols is None:
        symbols = [
            "SPY", "QQQ", "IWM",
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
            "BTC-USD", "ETH-USD", "SOL-USD",
            "DOGE-USD", "MSTR", "COIN",
        ]

    trader = PaperTrader(initial_capital=portfolio_value)

    # Fetch price data for all symbols
    price_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
            if not hist.empty and len(hist) >= 20:
                price_data[symbol] = {
                    "prices": hist["Close"].tolist(),
                    "highs": hist["High"].tolist(),
                    "lows": hist["Low"].tolist(),
                    "volumes": hist["Volume"].tolist(),
                }
        except Exception:
            continue

    # Run analysis
    portfolio = trader.analyze_and_trade(symbols, price_data)

    # Save full portfolio state for dashboard (JSON fallback)
    dashboard_data = {
        "timestamp": portfolio.timestamp,
        "portfolio": {
            "initial_capital": portfolio.initial_capital,
            "current_value": portfolio.portfolio_value,
            "cash": portfolio.cash,
            "total_return_pct": portfolio.total_return_pct,
            "daily_pnl": portfolio.daily_pnl,
            "max_drawdown_pct": portfolio.max_drawdown_pct,
        },
        "stats": {
            "total_trades": portfolio.total_trades,
            "win_rate": portfolio.win_rate,
            "avg_win_pct": portfolio.avg_win_pct,
            "avg_loss_pct": portfolio.avg_loss_pct,
            "profit_factor": portfolio.profit_factor,
        },
        "positions": [asdict(p) for p in portfolio.positions],
        "recent_trades": [asdict(t) for t in portfolio.closed_trades[-20:]],
        "equity_curve": portfolio.equity_history,
    }
    PAPER_LOG.write_text(json.dumps(dashboard_data, indent=2, default=str))

    return portfolio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper Trader")
    parser.add_argument("--symbol", "-s", help="Single symbol to trade")
    parser.add_argument("--portfolio", "-p", type=float, default=100000)
    parser.add_argument("--reset", action="store_true", help="Reset paper portfolio")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.reset:
        trader = PaperTrader(args.portfolio)
        trader.reset()
        print("Paper portfolio reset.")
    else:
        symbols = [args.symbol] if args.symbol else None
        portfolio = run_paper_trading(symbols, args.portfolio)

        if args.json:
            print(json.dumps({
                "value": portfolio.portfolio_value,
                "return": portfolio.total_return_pct,
                "positions": len(portfolio.positions),
                "trades": portfolio.total_trades,
                "win_rate": portfolio.win_rate,
            }, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"PAPER TRADING REPORT")
            print(f"{'='*60}")
            print(f"  Portfolio Value: ${portfolio.portfolio_value:,.2f}")
            print(f"  Cash:            ${portfolio.cash:,.2f}")
            print(f"  Total Return:    {portfolio.total_return_pct:+.2f}%")
            print(f"  Max Drawdown:    {portfolio.max_drawdown_pct:.2f}%")
            print(f"\n  Trades:     {portfolio.total_trades}")
            print(f"  Win Rate:   {portfolio.win_rate:.1%}")
            print(f"  Profit Factor: {portfolio.profit_factor:.2f}")
            if portfolio.positions:
                print(f"\n  Open Positions ({len(portfolio.positions)}):")
                for p in portfolio.positions:
                    pnl_emoji = "+" if p.unrealized_pnl >= 0 else ""
                    print(f"    {p.symbol} {p.direction.upper()} "
                          f"@ ${p.entry_price:.2f} -> ${p.current_price:.2f} "
                          f"({pnl_emoji}{p.unrealized_pnl_pct:.1f}%)")
            if portfolio.closed_trades:
                print(f"\n  Recent Trades:")
                for t in portfolio.closed_trades[-5:]:
                    emoji = "W" if t.pnl_dollars > 0 else "L"
                    print(f"    [{emoji}] {t.symbol} {t.direction} "
                          f"${t.entry_price:.2f}->${t.exit_price:.2f} "
                          f"({t.exit_reason}) ${t.pnl_dollars:+.2f}")
