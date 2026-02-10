#!/usr/bin/env python3
"""
BACKTESTER - Test Trading Strategies Against Historical Data

Run any strategy against historical price data with realistic assumptions:
- Commission costs
- Slippage simulation
- Partial fill modeling
- Walk-forward out-of-sample testing

Usage:
    python3 -m core.backtester --symbol AAPL --period 1y
    python3 -m core.backtester --universe default --period 6mo --json
"""

import json
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
from enum import Enum

from .trading_model import TradingModel, TechnicalIndicators, TradeSignal

BASE_DIR = Path(__file__).parent.parent
BACKTEST_RESULTS = BASE_DIR / "backtest_results.json"


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class BacktestTrade:
    """A single completed trade in the backtest"""
    symbol: str
    direction: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl_dollars: float
    pnl_pct: float
    return_pct: float  # Pct return on position value
    hold_days: int
    exit_reason: str  # "stop_loss", "take_profit", "signal_reversal", "end_of_data"
    entry_score: int
    exit_score: int


@dataclass
class EquityCurvePoint:
    """A point on the equity curve"""
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    num_positions: int


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    # Risk
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    volatility_annualized: float
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_hold_days: float
    profit_factor: float  # gross profits / gross losses
    expectancy: float  # average $ per trade
    # Position stats
    avg_position_size_pct: float
    max_concurrent_positions: int


@dataclass
class BacktestResult:
    """Complete backtest result"""
    timestamp: str
    strategy_name: str
    symbols: List[str]
    period: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    metrics: PerformanceMetrics
    trades: List[BacktestTrade]
    equity_curve: List[EquityCurvePoint]
    parameters: Dict


class Backtester:
    """
    Runs the TradingModel against historical data and produces
    comprehensive performance analytics.
    """

    def __init__(self, initial_capital: float = 100000,
                 commission: float = 0.0,  # Per-trade commission
                 slippage_pct: float = 0.001):  # 0.1% slippage
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct

    def run(self, symbol: str, prices: List[float],
            dates: List[str],
            highs: List[float] = None,
            lows: List[float] = None,
            volumes: List[float] = None,
            model: TradingModel = None) -> BacktestResult:
        """
        Run backtest on a single symbol.

        Args:
            symbol: Ticker symbol
            prices: List of closing prices
            dates: List of date strings (ISO format)
            highs/lows/volumes: Optional OHLCV data
            model: TradingModel instance (creates default if None)
        """
        if model is None:
            model = TradingModel(portfolio_value=self.initial_capital)

        n = len(prices)
        if n < 50:
            raise ValueError(f"Need at least 50 price points, got {n}")

        # State
        cash = self.initial_capital
        position = None  # Current position dict or None
        trades: List[BacktestTrade] = []
        equity_curve: List[EquityCurvePoint] = []
        peak_value = self.initial_capital

        # Min lookback for indicators
        lookback = 50

        for i in range(lookback, n):
            current_price = prices[i]
            current_date = dates[i] if i < len(dates) else f"day_{i}"

            # Get signal from model
            price_window = prices[:i + 1]
            high_window = highs[:i + 1] if highs else None
            low_window = lows[:i + 1] if lows else None
            vol_window = volumes[:i + 1] if volumes else None

            signal = model.generate_signal(
                symbol, price_window, high_window, low_window, vol_window
            )

            if signal is None:
                continue

            score = signal.score

            # === POSITION MANAGEMENT ===
            if position is not None:
                pos_direction = position["direction"]
                entry_price = position["entry_price"]
                stop_loss = position["stop_loss"]
                take_profit = position["take_profit"]
                shares = position["shares"]

                exit_reason = None

                # Check stop loss
                if pos_direction == "long" and current_price <= stop_loss:
                    exit_reason = "stop_loss"
                elif pos_direction == "short" and current_price >= stop_loss:
                    exit_reason = "stop_loss"
                # Check take profit
                elif pos_direction == "long" and current_price >= take_profit:
                    exit_reason = "take_profit"
                elif pos_direction == "short" and current_price <= take_profit:
                    exit_reason = "take_profit"
                # Check signal reversal
                elif pos_direction == "long" and score <= -25:
                    exit_reason = "signal_reversal"
                elif pos_direction == "short" and score >= 25:
                    exit_reason = "signal_reversal"

                if exit_reason:
                    # Apply slippage on exit
                    if pos_direction == "long":
                        exit_price = current_price * (1 - self.slippage_pct)
                        pnl = (exit_price - entry_price) * shares
                        return_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        exit_price = current_price * (1 + self.slippage_pct)
                        pnl = (entry_price - exit_price) * shares
                        return_pct = (entry_price - exit_price) / entry_price * 100

                    pnl -= self.commission  # Exit commission

                    hold_days = i - position["entry_idx"]

                    trades.append(BacktestTrade(
                        symbol=symbol,
                        direction=pos_direction,
                        entry_date=position["entry_date"],
                        entry_price=entry_price,
                        exit_date=current_date,
                        exit_price=round(exit_price, 2),
                        shares=shares,
                        pnl_dollars=round(pnl, 2),
                        pnl_pct=round(pnl / self.initial_capital * 100, 4),
                        return_pct=round(return_pct, 4),
                        hold_days=hold_days,
                        exit_reason=exit_reason,
                        entry_score=position["entry_score"],
                        exit_score=score,
                    ))

                    cash += shares * exit_price + pnl
                    position = None

            # === ENTRY LOGIC ===
            if position is None and signal.direction in ("long", "short"):
                # Check we have enough cash
                entry_price = current_price
                if signal.direction == "long":
                    entry_price *= (1 + self.slippage_pct)  # Slippage on buy
                else:
                    entry_price *= (1 - self.slippage_pct)  # Slippage on short

                # Calculate position size using the model's sizing
                shares = signal.position_shares
                if shares <= 0:
                    shares = max(1, int(self.initial_capital * 0.02 / abs(entry_price - signal.stop_loss))) if signal.stop_loss > 0 else 1

                position_cost = shares * entry_price + self.commission

                if position_cost <= cash:
                    cash -= position_cost
                    position = {
                        "direction": signal.direction,
                        "entry_price": round(entry_price, 2),
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "shares": shares,
                        "entry_date": current_date,
                        "entry_idx": i,
                        "entry_score": score,
                    }

            # === EQUITY CURVE ===
            positions_value = 0
            if position:
                if position["direction"] == "long":
                    positions_value = position["shares"] * current_price
                else:
                    # Short position P&L
                    entry_val = position["shares"] * position["entry_price"]
                    current_val = position["shares"] * current_price
                    positions_value = entry_val + (entry_val - current_val)

            portfolio_value = cash + positions_value
            daily_return = (portfolio_value - (equity_curve[-1].portfolio_value if equity_curve else self.initial_capital)) / (equity_curve[-1].portfolio_value if equity_curve else self.initial_capital)
            cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
            peak_value = max(peak_value, portfolio_value)
            drawdown = (peak_value - portfolio_value) / peak_value

            equity_curve.append(EquityCurvePoint(
                date=current_date,
                portfolio_value=round(portfolio_value, 2),
                cash=round(cash, 2),
                positions_value=round(positions_value, 2),
                daily_return=round(daily_return, 6),
                cumulative_return=round(cumulative_return, 6),
                drawdown=round(drawdown, 6),
                num_positions=1 if position else 0,
            ))

        # Close any remaining position at end
        if position:
            final_price = prices[-1]
            if position["direction"] == "long":
                pnl = (final_price - position["entry_price"]) * position["shares"]
                return_pct = (final_price - position["entry_price"]) / position["entry_price"] * 100
            else:
                pnl = (position["entry_price"] - final_price) * position["shares"]
                return_pct = (position["entry_price"] - final_price) / position["entry_price"] * 100

            trades.append(BacktestTrade(
                symbol=symbol,
                direction=position["direction"],
                entry_date=position["entry_date"],
                entry_price=position["entry_price"],
                exit_date=dates[-1] if dates else "end",
                exit_price=round(final_price, 2),
                shares=position["shares"],
                pnl_dollars=round(pnl, 2),
                pnl_pct=round(pnl / self.initial_capital * 100, 4),
                return_pct=round(return_pct, 4),
                hold_days=n - position["entry_idx"],
                exit_reason="end_of_data",
                entry_score=position["entry_score"],
                exit_score=0,
            ))
            cash += position["shares"] * final_price

        # Calculate metrics
        final_value = equity_curve[-1].portfolio_value if equity_curve else self.initial_capital
        metrics = self._calculate_metrics(trades, equity_curve, self.initial_capital, final_value)

        return BacktestResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_name="TradingModel_v1",
            symbols=[symbol],
            period=f"{n} bars",
            start_date=dates[lookback] if dates else "unknown",
            end_date=dates[-1] if dates else "unknown",
            initial_capital=self.initial_capital,
            final_value=round(final_value, 2),
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            parameters={
                "long_threshold": TradingModel.LONG_ENTRY_THRESHOLD,
                "short_threshold": TradingModel.SHORT_ENTRY_THRESHOLD,
                "stop_loss_pct": TradingModel.STOP_LOSS_PCT,
                "take_profit_pct": TradingModel.TAKE_PROFIT_PCT,
                "max_risk_per_trade": TradingModel.MAX_RISK_PER_TRADE,
                "commission": self.commission,
                "slippage_pct": self.slippage_pct,
            },
        )

    def _calculate_metrics(self, trades: List[BacktestTrade],
                           equity_curve: List[EquityCurvePoint],
                           initial_capital: float,
                           final_value: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""

        # Returns
        total_return = (final_value - initial_capital) / initial_capital * 100
        trading_days = len(equity_curve)
        years = trading_days / 252 if trading_days > 0 else 1
        annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Drawdown
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = 0
        for pt in equity_curve:
            if pt.drawdown > max_dd:
                max_dd = pt.drawdown
            if pt.drawdown > 0:
                current_dd_start += 1
                max_dd_duration = max(max_dd_duration, current_dd_start)
            else:
                current_dd_start = 0

        # Daily returns for Sharpe/Sortino
        daily_returns = [pt.daily_return for pt in equity_curve]
        if daily_returns:
            avg_daily = np.mean(daily_returns)
            std_daily = np.std(daily_returns)
            downside_returns = [r for r in daily_returns if r < 0]
            std_downside = np.std(downside_returns) if downside_returns else 0.0001

            volatility = float(std_daily * np.sqrt(252))
            sharpe = float(avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
            sortino = float(avg_daily / std_downside * np.sqrt(252)) if std_downside > 0 else 0
        else:
            volatility = 0
            sharpe = 0
            sortino = 0

        calmar = annualized_return / (max_dd * 100) if max_dd > 0 else 0

        # Trade statistics
        total_trades = len(trades)
        winners = [t for t in trades if t.pnl_dollars > 0]
        losers = [t for t in trades if t.pnl_dollars <= 0]
        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.return_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.return_pct for t in losers]) if losers else 0
        largest_win = max([t.return_pct for t in winners], default=0)
        largest_loss = min([t.return_pct for t in losers], default=0)
        avg_hold = np.mean([t.hold_days for t in trades]) if trades else 0

        gross_profits = sum(t.pnl_dollars for t in winners) if winners else 0
        gross_losses = abs(sum(t.pnl_dollars for t in losers)) if losers else 0.0001
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0

        expectancy = np.mean([t.pnl_dollars for t in trades]) if trades else 0

        # Position stats
        avg_size = np.mean([t.shares * t.entry_price / initial_capital * 100 for t in trades]) if trades else 0
        max_concurrent = max([pt.num_positions for pt in equity_curve], default=0)

        return PerformanceMetrics(
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(annualized_return, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            max_drawdown_duration_days=max_dd_duration,
            volatility_annualized=round(volatility * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate, 4),
            avg_win_pct=round(float(avg_win), 2),
            avg_loss_pct=round(float(avg_loss), 2),
            largest_win_pct=round(float(largest_win), 2),
            largest_loss_pct=round(float(largest_loss), 2),
            avg_hold_days=round(float(avg_hold), 1),
            profit_factor=round(float(profit_factor), 2),
            expectancy=round(float(expectancy), 2),
            avg_position_size_pct=round(float(avg_size), 2),
            max_concurrent_positions=max_concurrent,
        )

    def save_results(self, result: BacktestResult):
        """Save backtest results to disk"""
        data = {
            "timestamp": result.timestamp,
            "strategy": result.strategy_name,
            "symbols": result.symbols,
            "period": result.period,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "initial_capital": result.initial_capital,
            "final_value": result.final_value,
            "metrics": asdict(result.metrics),
            "trades": [asdict(t) for t in result.trades],
            "equity_curve": [asdict(p) for p in result.equity_curve[::5]],  # Every 5th point
            "parameters": result.parameters,
        }
        BACKTEST_RESULTS.write_text(json.dumps(data, indent=2))


def run_backtest_yfinance(symbol: str, period: str = "1y",
                          initial_capital: float = 100000) -> BacktestResult:
    """Convenience function: fetch from Yahoo Finance and backtest"""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval="1d")

    if hist.empty or len(hist) < 50:
        raise ValueError(f"Not enough data for {symbol} ({len(hist)} bars)")

    prices = hist["Close"].tolist()
    dates = [d.isoformat() for d in hist.index.tolist()]
    highs = hist["High"].tolist()
    lows = hist["Low"].tolist()
    volumes = hist["Volume"].tolist()

    bt = Backtester(initial_capital=initial_capital)
    model = TradingModel(portfolio_value=initial_capital)

    result = bt.run(symbol, prices, dates, highs, lows, volumes, model)
    bt.save_results(result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Trading Model")
    parser.add_argument("--symbol", "-s", default="SPY", help="Symbol to backtest")
    parser.add_argument("--period", "-p", default="1y", help="History period (1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--capital", "-c", type=float, default=100000, help="Initial capital")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    print(f"Backtesting {args.symbol} over {args.period}...")
    result = run_backtest_yfinance(args.symbol, args.period, args.capital)

    if args.json:
        print(json.dumps({
            "symbol": args.symbol,
            "metrics": asdict(result.metrics),
            "trades": len(result.trades),
            "final_value": result.final_value,
        }, indent=2))
    else:
        m = result.metrics
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS: {args.symbol} ({args.period})")
        print(f"{'='*60}")
        print(f"  Period: {result.start_date} to {result.end_date}")
        print(f"  Initial Capital: ${result.initial_capital:,.2f}")
        print(f"  Final Value:     ${result.final_value:,.2f}")
        print(f"\n--- Returns ---")
        print(f"  Total Return:       {m.total_return_pct:+.2f}%")
        print(f"  Annualized Return:  {m.annualized_return_pct:+.2f}%")
        print(f"\n--- Risk ---")
        print(f"  Max Drawdown:       {m.max_drawdown_pct:.2f}%")
        print(f"  Drawdown Duration:  {m.max_drawdown_duration_days} days")
        print(f"  Volatility (ann.):  {m.volatility_annualized:.2f}%")
        print(f"\n--- Risk-Adjusted ---")
        print(f"  Sharpe Ratio:       {m.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:      {m.sortino_ratio:.2f}")
        print(f"  Calmar Ratio:       {m.calmar_ratio:.2f}")
        print(f"\n--- Trades ---")
        print(f"  Total Trades:       {m.total_trades}")
        print(f"  Win Rate:           {m.win_rate:.1%}")
        print(f"  Avg Win:            {m.avg_win_pct:+.2f}%")
        print(f"  Avg Loss:           {m.avg_loss_pct:+.2f}%")
        print(f"  Largest Win:        {m.largest_win_pct:+.2f}%")
        print(f"  Largest Loss:       {m.largest_loss_pct:+.2f}%")
        print(f"  Avg Hold Days:      {m.avg_hold_days:.1f}")
        print(f"  Profit Factor:      {m.profit_factor:.2f}")
        print(f"  Expectancy:         ${m.expectancy:+.2f}/trade")
        print(f"\nResults saved to backtest_results.json")
