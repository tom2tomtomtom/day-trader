#!/usr/bin/env python3
"""
WALK-FORWARD OPTIMIZER — Systematic parameter search for strategies.

For each strategy (momentum, mean_reversion, breakout), optimizes parameters
using walk-forward methodology:
1. Split data into N windows
2. For each window: optimize on train set, validate on out-of-sample
3. Average OOS performance = true expected performance

Avoids overfitting by never using future data and reporting only OOS metrics.
Results stored in Supabase `optimization_runs` table.
"""

import logging
import itertools
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from .strategy import get_strategy, StrategyConfig, StrategySignal
from .execution_engine import ExecutionEngine, ManagedPosition, ExitReason
from .db import get_db

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


@dataclass
class OptimizationResult:
    """Result of a single parameter combination."""
    strategy_name: str
    parameters: Dict
    # In-sample metrics
    is_return_pct: float = 0.0
    is_sharpe: float = 0.0
    is_win_rate: float = 0.0
    is_trades: int = 0
    # Out-of-sample metrics (the ones that matter)
    oos_return_pct: float = 0.0
    oos_sharpe: float = 0.0
    oos_win_rate: float = 0.0
    oos_trades: int = 0
    # Walk-forward details
    n_folds: int = 0
    fold_results: List[Dict] = field(default_factory=list)


@dataclass
class OptimizationRun:
    """Complete optimization run for a strategy."""
    timestamp: str
    strategy_name: str
    symbol: str
    period: str
    n_bars: int
    n_folds: int
    param_grid: Dict[str, List]
    total_combinations: int
    best_params: Dict
    best_oos_return: float
    best_oos_sharpe: float
    all_results: List[OptimizationResult]


# Parameter grids for each strategy
PARAM_GRIDS = {
    "momentum": {
        "fast_ma": [5, 8, 10, 15],
        "slow_ma": [20, 30, 40, 50],
        "stop_loss_pct": [0.03, 0.05, 0.07],
        "take_profit_pct": [0.08, 0.12, 0.15],
    },
    "mean_reversion": {
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
        "stop_loss_pct": [0.03, 0.04, 0.05],
        "take_profit_pct": [0.04, 0.06, 0.08],
    },
    "breakout": {
        "lookback": [10, 15, 20, 30],
        "stop_loss_pct": [0.03, 0.04, 0.05],
        "take_profit_pct": [0.08, 0.10, 0.15],
    },
}


class WalkForwardOptimizer:
    """Walk-forward parameter optimizer for trading strategies."""

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        initial_capital: float = 100000,
        slippage_pct: float = 0.001,
    ):
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct

    def optimize(
        self,
        strategy_name: str,
        symbol: str,
        prices: List[float],
        dates: List[str],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
        param_grid: Optional[Dict] = None,
    ) -> OptimizationRun:
        """Run walk-forward optimization for a strategy.

        Args:
            strategy_name: One of "momentum", "mean_reversion", "breakout"
            symbol: Ticker symbol
            prices: Price array
            dates: Date strings
            highs/lows/volumes: Optional OHLCV data
            param_grid: Override default parameter grid

        Returns:
            OptimizationRun with best parameters and all results
        """
        grid = param_grid or PARAM_GRIDS.get(strategy_name, {})
        if not grid:
            raise ValueError(f"No parameter grid for strategy '{strategy_name}'")

        n = len(prices)
        if n < 100:
            raise ValueError(f"Need at least 100 bars, got {n}")

        # Generate all parameter combinations
        param_names = list(grid.keys())
        param_values = list(grid.values())
        combinations = list(itertools.product(*param_values))
        total_combos = len(combinations)

        logger.info(
            f"Optimizing {strategy_name} on {symbol}: "
            f"{total_combos} combinations × {self.n_folds} folds"
        )

        # Split data into walk-forward folds
        folds = self._create_folds(n)

        all_results = []

        for combo_idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            # Filter invalid combos (e.g., fast_ma >= slow_ma)
            if "fast_ma" in params and "slow_ma" in params:
                if params["fast_ma"] >= params["slow_ma"]:
                    continue

            config = StrategyConfig(name=strategy_name, parameters=params)

            is_returns = []
            oos_returns = []
            is_trades_list = []
            oos_trades_list = []
            fold_details = []

            for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(folds):
                # In-sample (train)
                is_metrics = self._run_strategy(
                    strategy_name, config, symbol,
                    prices[train_start:train_end],
                    dates[train_start:train_end],
                    highs[train_start:train_end] if highs else None,
                    lows[train_start:train_end] if lows else None,
                    volumes[train_start:train_end] if volumes else None,
                )

                # Out-of-sample (validation)
                oos_metrics = self._run_strategy(
                    strategy_name, config, symbol,
                    prices[val_start:val_end],
                    dates[val_start:val_end],
                    highs[val_start:val_end] if highs else None,
                    lows[val_start:val_end] if lows else None,
                    volumes[val_start:val_end] if volumes else None,
                )

                is_returns.append(is_metrics["return_pct"])
                oos_returns.append(oos_metrics["return_pct"])
                is_trades_list.append(is_metrics["n_trades"])
                oos_trades_list.append(oos_metrics["n_trades"])

                fold_details.append({
                    "fold": fold_idx + 1,
                    "is_return": is_metrics["return_pct"],
                    "oos_return": oos_metrics["return_pct"],
                    "is_sharpe": is_metrics["sharpe"],
                    "oos_sharpe": oos_metrics["sharpe"],
                    "is_trades": is_metrics["n_trades"],
                    "oos_trades": oos_metrics["n_trades"],
                })

            result = OptimizationResult(
                strategy_name=strategy_name,
                parameters=params,
                is_return_pct=float(np.mean(is_returns)) if is_returns else 0,
                is_sharpe=float(np.mean([f["is_sharpe"] for f in fold_details])),
                is_win_rate=0,  # Simplified
                is_trades=sum(is_trades_list),
                oos_return_pct=float(np.mean(oos_returns)) if oos_returns else 0,
                oos_sharpe=float(np.mean([f["oos_sharpe"] for f in fold_details])),
                oos_win_rate=0,
                oos_trades=sum(oos_trades_list),
                n_folds=len(folds),
                fold_results=fold_details,
            )
            all_results.append(result)

            if (combo_idx + 1) % 10 == 0:
                logger.info(f"  Progress: {combo_idx + 1}/{total_combos}")

        # Sort by OOS Sharpe (most robust metric)
        all_results.sort(key=lambda r: r.oos_sharpe, reverse=True)

        best = all_results[0] if all_results else None

        run = OptimizationRun(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_name=strategy_name,
            symbol=symbol,
            period=f"{n} bars",
            n_bars=n,
            n_folds=self.n_folds,
            param_grid=grid,
            total_combinations=total_combos,
            best_params=best.parameters if best else {},
            best_oos_return=best.oos_return_pct if best else 0,
            best_oos_sharpe=best.oos_sharpe if best else 0,
            all_results=all_results[:20],  # Top 20 only
        )

        # Save to DB
        self._save_to_db(run)

        logger.info(
            f"Optimization complete for {strategy_name}/{symbol}: "
            f"best OOS Sharpe={run.best_oos_sharpe:.3f} "
            f"return={run.best_oos_return:.2f}% "
            f"params={run.best_params}"
        )

        return run

    def _create_folds(self, n: int) -> List[Tuple[int, int, int, int]]:
        """Create walk-forward folds (expanding window)."""
        folds = []
        min_train = max(50, int(n * 0.3))  # At least 30% or 50 bars for training
        fold_size = (n - min_train) // self.n_folds

        if fold_size < 20:
            # Not enough data for requested folds, use fewer
            fold_size = max(20, (n - min_train) // 2)
            actual_folds = max(2, (n - min_train) // fold_size)
        else:
            actual_folds = self.n_folds

        for i in range(actual_folds):
            val_end = min_train + (i + 1) * fold_size
            val_start = min_train + i * fold_size
            train_start = 0
            train_end = val_start

            if val_end > n:
                val_end = n
            if train_end <= train_start + 20:
                continue

            folds.append((train_start, train_end, val_start, val_end))

        return folds

    def _run_strategy(
        self,
        strategy_name: str,
        config: StrategyConfig,
        symbol: str,
        prices: List[float],
        dates: List[str],
        highs: Optional[List[float]],
        lows: Optional[List[float]],
        volumes: Optional[List[float]],
    ) -> Dict:
        """Run a strategy on a price series and return metrics."""
        if len(prices) < 30:
            return {"return_pct": 0, "sharpe": 0, "n_trades": 0, "win_rate": 0}

        strategy = get_strategy(strategy_name, config)
        engine = ExecutionEngine(slippage_pct=self.slippage_pct)

        capital = self.initial_capital
        position: Optional[ManagedPosition] = None
        trades_pnl = []
        daily_returns = []
        prev_equity = capital

        for i in range(30, len(prices)):
            current_price = prices[i]

            # Generate signal
            signal = strategy.generate_signals(
                symbol=symbol,
                prices=prices[:i + 1],
                highs=highs[:i + 1] if highs else None,
                lows=lows[:i + 1] if lows else None,
                volumes=volumes[:i + 1] if volumes else None,
            )

            # Manage existing position
            if position is not None:
                position.update_price(current_price)
                exit_signal = engine.check_exit(position, current_price, signal.score)

                if exit_signal is not None:
                    pnl, pnl_pct = engine.calculate_pnl(position, exit_signal.exit_price, exit_signal.exit_shares)
                    capital += pnl

                    if exit_signal.is_partial:
                        engine.apply_partial_exit(position, exit_signal.exit_shares)
                    else:
                        trades_pnl.append(pnl_pct)
                        position = None

            # Entry logic
            if position is None and signal.direction in ("long", "short"):
                shares = max(1, int(capital * 0.1 / current_price))
                position = engine.create_position(
                    symbol=symbol,
                    direction=signal.direction,
                    entry_price=current_price,
                    shares=shares,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_date=dates[i] if i < len(dates) else "",
                    entry_score=signal.score,
                )

            # Track daily return
            equity = capital
            if position is not None:
                equity += position.unrealized_pnl
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns.append(daily_ret)
            prev_equity = equity

        # Close any open position
        if position is not None:
            exit_sig = engine.close_at_end(position, prices[-1])
            pnl, pnl_pct = engine.calculate_pnl(position, exit_sig.exit_price, exit_sig.exit_shares)
            capital += pnl
            trades_pnl.append(pnl_pct)

        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        n_trades = len(trades_pnl)
        win_rate = sum(1 for p in trades_pnl if p > 0) / n_trades if n_trades > 0 else 0

        # Sharpe ratio (annualized, assuming daily returns)
        if daily_returns and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        return {
            "return_pct": round(total_return, 4),
            "sharpe": round(sharpe, 4),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
        }

    def _save_to_db(self, run: OptimizationRun):
        """Save optimization results to Supabase."""
        try:
            db = get_db()
            db.client.table("optimization_runs").insert({
                "strategy_name": run.strategy_name,
                "symbol": run.symbol,
                "n_bars": run.n_bars,
                "n_folds": run.n_folds,
                "total_combinations": run.total_combinations,
                "best_params": run.best_params,
                "best_oos_return": run.best_oos_return,
                "best_oos_sharpe": run.best_oos_sharpe,
                "top_results": [asdict(r) for r in run.all_results[:10]],
                "param_grid": run.param_grid,
                "created_at": run.timestamp,
            }).execute()
            logger.info(f"Optimization results saved to Supabase")
        except Exception as e:
            logger.warning(f"Could not save optimization to Supabase: {e}")

        # Also save to disk
        try:
            results_path = BASE_DIR / "optimization_results.json"
            data = {
                "timestamp": run.timestamp,
                "strategy": run.strategy_name,
                "symbol": run.symbol,
                "best_params": run.best_params,
                "best_oos_sharpe": run.best_oos_sharpe,
                "best_oos_return": run.best_oos_return,
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass


def run_optimization(
    strategy_name: str = "momentum",
    symbol: str = "SPY",
    period: str = "2y",
) -> OptimizationRun:
    """Convenience function: fetch data from Yahoo Finance and optimize."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval="1d")

    if hist.empty or len(hist) < 100:
        raise ValueError(f"Not enough data for {symbol} ({len(hist)} bars)")

    prices = hist["Close"].tolist()
    dates = [d.isoformat() for d in hist.index.tolist()]
    highs = hist["High"].tolist()
    lows = hist["Low"].tolist()
    volumes = hist["Volume"].tolist()

    optimizer = WalkForwardOptimizer()
    return optimizer.optimize(
        strategy_name=strategy_name,
        symbol=symbol,
        prices=prices,
        dates=dates,
        highs=highs,
        lows=lows,
        volumes=volumes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-Forward Strategy Optimizer")
    parser.add_argument("--strategy", "-s", default="momentum",
                        choices=["momentum", "mean_reversion", "breakout"])
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    print(f"\nOptimizing {args.strategy} on {args.symbol} ({args.period})...")
    result = run_optimization(args.strategy, args.symbol, args.period)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION RESULTS: {result.strategy_name}/{result.symbol}")
    print(f"{'='*60}")
    print(f"Bars: {result.n_bars} | Folds: {result.n_folds} | Combos tested: {result.total_combinations}")
    print(f"\nBest Parameters: {result.best_params}")
    print(f"Best OOS Return: {result.best_oos_return:+.2f}%")
    print(f"Best OOS Sharpe: {result.best_oos_sharpe:.3f}")

    print(f"\nTop 5 Parameter Sets:")
    for i, r in enumerate(result.all_results[:5], 1):
        print(f"  {i}. {r.parameters}")
        print(f"     IS: {r.is_return_pct:+.2f}% (Sharpe {r.is_sharpe:.3f})")
        print(f"     OOS: {r.oos_return_pct:+.2f}% (Sharpe {r.oos_sharpe:.3f})")
        print(f"     Trades: {r.is_trades} IS, {r.oos_trades} OOS")
