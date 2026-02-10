#!/usr/bin/env python3
"""
PAPER TRADER - Forward-Test Strategies in Real-Time

Simulates real trading without risking capital. Connects the TradingModel
to live data and tracks a virtual portfolio with full P&L tracking.

Features:
- Virtual portfolio with cash and positions
- Real-time signal generation from TradingModel
- Automatic stop-loss and take-profit execution
- Full trade log with performance metrics
- JSON output for dashboard consumption

Usage:
    python3 -m core.paper_trader                    # Scan default universe
    python3 -m core.paper_trader --symbol AAPL      # Single symbol
    python3 -m core.paper_trader --portfolio 50000   # Custom starting capital
"""

import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

from .trading_model import TradingModel, TradeSignal, MEME_ASSETS

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

    def update(self, current_price: float):
        self.current_price = current_price
        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.shares
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.shares
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100


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
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperTrade] = []
        self.equity_history: List[Dict] = []
        self.peak_value = initial_capital

        self.model = TradingModel(portfolio_value=initial_capital)

        # Try to load existing state
        self._load_state()

    def analyze_and_trade(self, symbols: List[str],
                          price_data: Dict[str, Dict]) -> PaperPortfolio:
        """
        Main loop: analyze all symbols and execute paper trades.

        Args:
            symbols: List of symbols to analyze
            price_data: Dict mapping symbol -> {prices, highs, lows, volumes}
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
                self._check_entry(symbol, signal, now)

            # Update position price if still open
            if symbol in self.positions:
                self.positions[symbol].update(current_price)

        # Update portfolio value and equity curve
        portfolio = self._build_portfolio_state(now)
        self._save_state()

        return portfolio

    def _manage_position(self, symbol: str, signal: TradeSignal,
                         current_price: float, timestamp: str):
        """Check if position should be closed"""
        pos = self.positions[symbol]
        exit_reason = None

        # Check stop loss
        if pos.direction == "long" and current_price <= pos.stop_loss:
            exit_reason = "stop_loss"
        elif pos.direction == "short" and current_price >= pos.stop_loss:
            exit_reason = "stop_loss"
        # Check take profit
        elif pos.direction == "long" and current_price >= pos.take_profit:
            exit_reason = "take_profit"
        elif pos.direction == "short" and current_price <= pos.take_profit:
            exit_reason = "take_profit"
        # Check signal reversal
        elif pos.direction == "long" and signal.score <= -25:
            exit_reason = "signal_reversal"
        elif pos.direction == "short" and signal.score >= 25:
            exit_reason = "signal_reversal"

        if exit_reason:
            self._close_position(symbol, current_price, exit_reason,
                                 signal.score, timestamp)

    def _check_entry(self, symbol: str, signal: TradeSignal, timestamp: str):
        """Check if we should enter a new position"""
        if signal.direction == "flat":
            return

        # Check cash available
        position_cost = signal.position_shares * signal.entry_price
        if position_cost > self.cash * 0.95:  # Keep 5% cash buffer
            return
        if signal.position_shares <= 0:
            return

        # Enter position
        self.cash -= position_cost
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            shares=signal.position_shares,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_date=timestamp,
            entry_score=signal.score,
        )

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

        self.closed_trades.append(PaperTrade(
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
        ))

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
        PAPER_STATE.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        """Load state from disk"""
        if not PAPER_STATE.exists():
            return

        try:
            state = json.loads(PAPER_STATE.read_text())
            self.cash = state.get("cash", self.initial_capital)
            self.peak_value = state.get("peak_value", self.initial_capital)
            self.equity_history = state.get("equity_history", [])

            # Restore positions
            for sym, pos_data in state.get("positions", {}).items():
                self.positions[sym] = PaperPosition(**pos_data)

            # Restore trades
            for trade_data in state.get("closed_trades", []):
                self.closed_trades.append(PaperTrade(**trade_data))

        except Exception:
            pass  # Start fresh if state is corrupt

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

    # Save full portfolio state for dashboard
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
    PAPER_LOG.write_text(json.dumps(dashboard_data, indent=2))

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
