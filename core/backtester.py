#!/usr/bin/env python3
"""
BACKTESTER - Test Trading Strategies Against Historical Data

Run any strategy against historical price data with realistic assumptions:
- Commission costs
- Slippage simulation
- Partial fill modeling
- Walk-forward out-of-sample testing
- Feature vector computation at each trade entry (Phase 2.4: Cold Start)
- Optional bulk upload of backtested trades to Supabase for ML training

Usage:
    python3 -m core.backtester --symbol AAPL --period 1y
    python3 -m core.backtester --universe default --period 6mo --json
    python3 -m core.backtester --symbol SPY --period 2y --upload  # Upload trades to DB
"""

import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
from enum import Enum

from .trading_model import TradingModel, TechnicalIndicators, TradeSignal
from .feature_engine import FeatureEngine, FeatureVector
from .execution_engine import ExecutionEngine, ManagedPosition, ExitReason

logger = logging.getLogger(__name__)

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
    entry_features: Dict = field(default_factory=dict)  # FeatureVector at entry for ML


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
                 slippage_pct: float = 0.001,  # 0.1% slippage
                 latency_bars: int = 0):  # Bars of latency (0=same bar, 1=next bar open)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.latency_bars = latency_bars

    def run(self, symbol: str, prices: List[float],
            dates: List[str],
            highs: List[float] = None,
            lows: List[float] = None,
            volumes: List[float] = None,
            model: TradingModel = None) -> BacktestResult:
        """
        Run backtest on a single symbol using the unified ExecutionEngine.

        This shares the same exit logic (trailing stops, partial exits,
        ATR stops, signal reversal) as the paper trader.
        """
        if model is None:
            model = TradingModel(portfolio_value=self.initial_capital)

        n = len(prices)
        if n < 50:
            raise ValueError(f"Need at least 50 price points, got {n}")

        engine = ExecutionEngine(
            slippage_pct=self.slippage_pct,
            commission=self.commission,
        )
        feature_engine = FeatureEngine()

        # State
        cash = self.initial_capital
        position: Optional[ManagedPosition] = None
        trades: List[BacktestTrade] = []
        equity_curve: List[EquityCurvePoint] = []
        peak_value = self.initial_capital

        lookback = 50
        latency = self.latency_bars

        # Pending entry: (bar_idx_to_execute, signal, feature_data)
        pending_entry = None

        def _record_trade(pos: ManagedPosition, exit_price: float,
                          exit_shares: int, exit_reason: str,
                          exit_date: str, exit_score: int, bar_idx: int):
            pnl, return_pct = engine.calculate_pnl(pos, exit_price, exit_shares)
            pnl_pct = round(pnl / self.initial_capital * 100, 4)
            hold_days = bar_idx - pos.entry_idx
            trades.append(BacktestTrade(
                symbol=symbol,
                direction=pos.direction,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                shares=exit_shares,
                pnl_dollars=pnl,
                pnl_pct=pnl_pct,
                return_pct=return_pct,
                hold_days=hold_days,
                exit_reason=exit_reason,
                entry_score=pos.entry_score,
                exit_score=exit_score,
                entry_features=pos.entry_features,
            ))
            return pnl

        for i in range(lookback, n):
            current_price = prices[i]
            current_date = dates[i] if i < len(dates) else f"day_{i}"

            price_window = prices[:i + 1]
            high_window = highs[:i + 1] if highs else None
            low_window = lows[:i + 1] if lows else None
            vol_window = volumes[:i + 1] if volumes else None

            # === EXECUTE PENDING ENTRY (latency-aware) ===
            if pending_entry is not None and i >= pending_entry[0]:
                p_signal, p_features = pending_entry[1], pending_entry[2]
                pending_entry = None

                if position is None:
                    shares = p_signal.position_shares
                    if shares <= 0:
                        if p_signal.stop_loss > 0:
                            risk = abs(current_price - p_signal.stop_loss)
                            shares = max(1, int(self.initial_capital * 0.02 / risk)) if risk > 0 else 1
                        else:
                            shares = 1

                    atr = p_signal.indicators.atr_14 if p_signal.indicators else 0

                    # Latency > 0: execute at this bar's open (approximated by price)
                    pos = engine.create_position(
                        symbol=symbol,
                        direction=p_signal.direction,
                        entry_price=current_price,
                        shares=shares,
                        stop_loss=p_signal.stop_loss,
                        take_profit=p_signal.take_profit,
                        entry_date=current_date,
                        entry_score=p_signal.score,
                        atr=atr,
                        entry_idx=i,
                        apply_slippage=True,
                    )

                    position_cost = pos.shares * pos.entry_price + self.commission
                    if position_cost <= cash:
                        pos.entry_features = p_features
                        cash -= position_cost
                        position = pos

            signal = model.generate_signal(
                symbol, price_window, high_window, low_window, vol_window
            )

            if signal is None:
                # Still update position price for trailing stop tracking
                if position is not None:
                    position.update_price(current_price)
                # Record equity even with no signal
                positions_value = self._position_value(position, current_price)
                portfolio_value = cash + positions_value
                prev_val = equity_curve[-1].portfolio_value if equity_curve else self.initial_capital
                daily_return = (portfolio_value - prev_val) / prev_val
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
                continue

            score = signal.score

            # === POSITION MANAGEMENT (via ExecutionEngine) ===
            if position is not None:
                position.update_price(current_price)
                exit_signal = engine.check_exit(position, current_price, score)

                if exit_signal is not None:
                    if exit_signal.is_partial:
                        _record_trade(
                            position, exit_signal.exit_price, exit_signal.exit_shares,
                            exit_signal.reason.value, current_date, score, i,
                        )
                        cash += exit_signal.exit_shares * exit_signal.exit_price
                        engine.apply_partial_exit(position, exit_signal.exit_shares)
                    else:
                        pnl = _record_trade(
                            position, exit_signal.exit_price, exit_signal.exit_shares,
                            exit_signal.reason.value, current_date, score, i,
                        )
                        cash += exit_signal.exit_shares * exit_signal.exit_price
                        position = None

            # === ENTRY LOGIC ===
            if position is None and signal.direction in ("long", "short"):
                # Compute features now (from data available at signal bar)
                entry_fv = feature_engine.compute(
                    symbol=symbol,
                    prices=price_window,
                    highs=high_window,
                    lows=low_window,
                    volumes=vol_window,
                    indicators=signal.indicators,
                )
                entry_features = entry_fv.to_dict()

                if latency > 0:
                    # Queue for execution on a future bar
                    pending_entry = (i + latency, signal, entry_features)
                else:
                    # Immediate execution (latency=0)
                    shares = signal.position_shares
                    if shares <= 0:
                        if signal.stop_loss > 0:
                            risk = abs(current_price - signal.stop_loss)
                            shares = max(1, int(self.initial_capital * 0.02 / risk)) if risk > 0 else 1
                        else:
                            shares = 1

                    atr = signal.indicators.atr_14 if signal.indicators else 0

                    pos = engine.create_position(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_price=current_price,
                        shares=shares,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        entry_date=current_date,
                        entry_score=score,
                        atr=atr,
                        entry_idx=i,
                        apply_slippage=True,
                    )

                    position_cost = pos.shares * pos.entry_price + self.commission
                    if position_cost <= cash:
                        pos.entry_features = entry_features
                        cash -= position_cost
                        position = pos

            # === EQUITY CURVE ===
            positions_value = self._position_value(position, current_price)
            portfolio_value = cash + positions_value
            prev_val = equity_curve[-1].portfolio_value if equity_curve else self.initial_capital
            daily_return = (portfolio_value - prev_val) / prev_val
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
            exit_signal = engine.close_at_end(position, prices[-1])
            _record_trade(
                position, exit_signal.exit_price, exit_signal.exit_shares,
                exit_signal.reason.value,
                dates[-1] if dates else "end", 0, n,
            )
            cash += exit_signal.exit_shares * exit_signal.exit_price

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
                "latency_bars": self.latency_bars,
            },
        )

    @staticmethod
    def _position_value(position: Optional[ManagedPosition], current_price: float) -> float:
        """Calculate current value of an open position."""
        if position is None:
            return 0.0
        if position.direction == "long":
            return position.shares * current_price
        else:
            entry_val = position.shares * position.entry_price
            current_val = position.shares * current_price
            return entry_val + (entry_val - current_val)

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

    def upload_trades_to_db(self, result: BacktestResult) -> int:
        """
        Bulk upload backtested trades to Supabase for ML training (Phase 2.4).

        Each trade is marked with is_backtest=True and includes the full
        entry_features dict computed at the time of the trade entry.

        Returns the number of successfully uploaded trades.
        """
        try:
            from .db import get_db
        except ImportError:
            logger.warning("Could not import db module — skipping DB upload")
            return 0

        db = get_db()
        if not db.connected:
            logger.warning("Supabase not connected — skipping DB upload. "
                           "Trades are still saved locally in backtest_results.json")
            return 0

        uploaded = 0
        failed = 0
        for trade in result.trades:
            # Only upload trades that have feature data (meaningful for ML)
            if not trade.entry_features:
                logger.debug(f"Skipping trade {trade.symbol} {trade.entry_date} — no entry features")
                continue

            success = db.log_trade({
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
                "regime_at_entry": trade.entry_features.get("regime", ""),
                "is_backtest": True,
            })

            if success:
                uploaded += 1
            else:
                failed += 1

        logger.info(f"Uploaded {uploaded}/{len(result.trades)} backtested trades to DB "
                     f"({failed} failed)")
        return uploaded


def run_backtest_yfinance(symbol: str, period: str = "1y",
                          initial_capital: float = 100000,
                          upload_to_db: bool = False) -> BacktestResult:
    """
    Convenience function: fetch from Yahoo Finance and backtest.

    Args:
        symbol: Ticker symbol
        period: yfinance period string (1mo, 3mo, 6mo, 1y, 2y, 5y)
        initial_capital: Starting portfolio value
        upload_to_db: If True, upload backtested trades to Supabase
                      for ML training data (Phase 2.4 cold start)
    """
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

    if upload_to_db:
        uploaded = bt.upload_trades_to_db(result)
        trades_with_features = sum(1 for t in result.trades if t.entry_features)
        print(f"  DB Upload: {uploaded}/{trades_with_features} trades with features uploaded")

    return result


def run_cold_start_backtest(symbols: List[str] = None,
                            period: str = "2y",
                            initial_capital: float = 100000,
                            upload: bool = True) -> Dict:
    """
    Phase 2.4: Cold Start via Backtesting.

    Run backtests across multiple symbols to generate 100-300+ trades
    with full feature context for initial ML model training.

    Args:
        symbols: List of symbols to backtest (defaults to diversified universe)
        period: History period per symbol
        initial_capital: Starting capital per backtest
        upload: Whether to upload trades to Supabase

    Returns:
        Summary dict with total trades, uploads, and per-symbol results
    """
    import yfinance as yf

    if symbols is None:
        # Diversified universe designed to generate plenty of trades
        symbols = [
            # Major indices / ETFs
            "SPY", "QQQ", "IWM", "DIA",
            # Mega-cap tech (high liquidity, frequent signals)
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
            # Volatile / momentum stocks
            "COIN", "MSTR", "PLTR", "SQ", "SHOP", "ROKU", "SNAP",
            # Crypto (high volatility = more trades)
            "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
        ]

    total_trades = 0
    total_uploaded = 0
    total_with_features = 0
    results_summary = []

    for symbol in symbols:
        try:
            print(f"\nBacktesting {symbol} ({period})...")
            result = run_backtest_yfinance(
                symbol, period, initial_capital,
                upload_to_db=upload,
            )
            n_trades = len(result.trades)
            n_features = sum(1 for t in result.trades if t.entry_features)
            total_trades += n_trades
            total_with_features += n_features

            results_summary.append({
                "symbol": symbol,
                "trades": n_trades,
                "trades_with_features": n_features,
                "total_return_pct": result.metrics.total_return_pct,
                "win_rate": result.metrics.win_rate,
            })

            print(f"  {symbol}: {n_trades} trades ({n_features} with features), "
                  f"return {result.metrics.total_return_pct:+.2f}%, "
                  f"win rate {result.metrics.win_rate:.1%}")

        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")
            results_summary.append({
                "symbol": symbol,
                "trades": 0,
                "error": str(e),
            })

    print(f"\n{'='*60}")
    print(f"COLD START BACKTEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Symbols tested:      {len(symbols)}")
    print(f"  Total trades:        {total_trades}")
    print(f"  Trades with features: {total_with_features}")
    if upload:
        print(f"  (Trades uploaded to Supabase with is_backtest=True)")
    print(f"  Target range:        100-300 trades")
    if total_with_features < 100:
        print(f"  WARNING: Below target. Consider adding more symbols or longer period.")
    elif total_with_features > 300:
        print(f"  Excellent: Above target range for ML training.")
    else:
        print(f"  On target for initial ML model training.")

    return {
        "total_trades": total_trades,
        "total_with_features": total_with_features,
        "symbols_tested": len(symbols),
        "results": results_summary,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Trading Model")
    parser.add_argument("--symbol", "-s", default="SPY", help="Symbol to backtest")
    parser.add_argument("--period", "-p", default="1y", help="History period (1mo, 3mo, 6mo, 1y, 2y, 5y)")
    parser.add_argument("--capital", "-c", type=float, default=100000, help="Initial capital")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--upload", action="store_true",
                        help="Upload backtested trades to Supabase for ML training")
    parser.add_argument("--cold-start", action="store_true",
                        help="Run Phase 2.4 cold start: backtest multiple symbols and upload trades")
    args = parser.parse_args()

    if args.cold_start:
        # Phase 2.4: Cold Start — backtest diversified universe for ML training data
        summary = run_cold_start_backtest(
            period=args.period,
            initial_capital=args.capital,
            upload=args.upload,
        )
        if args.json:
            print(json.dumps(summary, indent=2))
    else:
        print(f"Backtesting {args.symbol} over {args.period}...")
        result = run_backtest_yfinance(args.symbol, args.period, args.capital,
                                       upload_to_db=args.upload)

        if args.json:
            print(json.dumps({
                "symbol": args.symbol,
                "metrics": asdict(result.metrics),
                "trades": len(result.trades),
                "trades_with_features": sum(1 for t in result.trades if t.entry_features),
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
            trades_with_ft = sum(1 for t in result.trades if t.entry_features)
            print(f"  With Features:      {trades_with_ft}/{m.total_trades}")
            print(f"  Win Rate:           {m.win_rate:.1%}")
            print(f"  Avg Win:            {m.avg_win_pct:+.2f}%")
            print(f"  Avg Loss:           {m.avg_loss_pct:+.2f}%")
            print(f"  Largest Win:        {m.largest_win_pct:+.2f}%")
            print(f"  Largest Loss:       {m.largest_loss_pct:+.2f}%")
            print(f"  Avg Hold Days:      {m.avg_hold_days:.1f}")
            print(f"  Profit Factor:      {m.profit_factor:.2f}")
            print(f"  Expectancy:         ${m.expectancy:+.2f}/trade")
            print(f"\nResults saved to backtest_results.json")
            if args.upload:
                print(f"Trades uploaded to Supabase (is_backtest=True)")
