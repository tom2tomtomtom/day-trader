"""
Backtesting Framework.

Replays historical match data through the full pipeline
(state engine → pricing model → signal generator → execution)
to validate strategy performance before live deployment.

Supports both directional signals and market-making strategies.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cricket.config import EngineConfig, RiskConfig, SignalConfig
from cricket.data.ball_event import BallEvent, MatchInfo
from cricket.data.cricsheet_loader import load_match_from_csv, load_matches_from_directory
from cricket.data.exchange_feed import SimulatedExchangeFeed
from cricket.execution.engine import ExecutionEngine, PositionStatus
from cricket.execution.market_maker import MarketMaker, MarketMakerConfig
from cricket.models.ensemble import EnsemblePricingModel
from cricket.signals.signals import SignalDirection, SignalGenerator
from cricket.state.match_state import MatchStateEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single match backtest."""

    match_id: str
    team_a: str
    team_b: str
    winner: str
    format: str

    # Directional trading results
    signals_generated: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    directional_pnl: float = 0.0

    # Market-making results
    mm_trades: int = 0
    mm_spread_captured: float = 0.0
    mm_pnl: float = 0.0

    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return self.trades_won / total if total > 0 else 0.0


@dataclass
class BacktestSummary:
    """Aggregate results across all backtested matches."""

    matches: list[BacktestResult] = field(default_factory=list)

    @property
    def total_matches(self) -> int:
        return len(self.matches)

    @property
    def total_pnl(self) -> float:
        return sum(m.total_pnl for m in self.matches)

    @property
    def total_directional_pnl(self) -> float:
        return sum(m.directional_pnl for m in self.matches)

    @property
    def total_mm_pnl(self) -> float:
        return sum(m.mm_pnl for m in self.matches)

    @property
    def total_signals(self) -> int:
        return sum(m.signals_generated for m in self.matches)

    @property
    def total_trades(self) -> int:
        return sum(m.trades_executed for m in self.matches)

    @property
    def total_wins(self) -> int:
        return sum(m.trades_won for m in self.matches)

    @property
    def win_rate(self) -> float:
        total = sum(m.trades_won + m.trades_lost for m in self.matches)
        return self.total_wins / total if total > 0 else 0.0

    @property
    def profitable_matches(self) -> int:
        return sum(1 for m in self.matches if m.total_pnl > 0)

    @property
    def avg_pnl_per_match(self) -> float:
        return self.total_pnl / self.total_matches if self.total_matches > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio based on match-level returns."""
        if len(self.matches) < 2:
            return 0.0
        pnls = [m.total_pnl for m in self.matches]
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std = variance ** 0.5
        return mean_pnl / std if std > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from cumulative P&L curve."""
        if not self.matches:
            return 0.0
        cumulative = []
        total = 0.0
        for m in self.matches:
            total += m.total_pnl
            cumulative.append(total)
        peak = cumulative[0]
        max_dd = 0.0
        for val in cumulative:
            peak = max(peak, val)
            dd = peak - val
            max_dd = max(max_dd, dd)
        return max_dd

    def summary_str(self) -> str:
        """Human-readable summary."""
        return (
            f"=== BACKTEST SUMMARY ===\n"
            f"Matches: {self.total_matches}\n"
            f"Profitable: {self.profitable_matches}/{self.total_matches} "
            f"({self.profitable_matches/self.total_matches*100:.0f}%)\n"
            f"Signals: {self.total_signals} | Trades: {self.total_trades}\n"
            f"Win Rate: {self.win_rate*100:.1f}%\n"
            f"--- P&L ---\n"
            f"Directional: £{self.total_directional_pnl:.2f}\n"
            f"Market-Making: £{self.total_mm_pnl:.2f}\n"
            f"Total P&L: £{self.total_pnl:.2f}\n"
            f"Avg per match: £{self.avg_pnl_per_match:.2f}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: £{self.max_drawdown:.2f}\n"
        )


class Backtester:
    """Runs historical match data through the trading pipeline.

    Usage:
        bt = Backtester(config)
        summary = bt.run_directory("data/cricsheet/t20s/")
        print(summary.summary_str())
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        enable_directional: bool = True,
        enable_market_making: bool = True,
    ):
        self._config = config or EngineConfig()
        self._enable_directional = enable_directional
        self._enable_mm = enable_market_making

    def run_match(
        self,
        match_info: MatchInfo,
        events: list[BallEvent],
        winner: Optional[str] = None,
    ) -> BacktestResult:
        """Backtest a single match.

        Args:
            match_info: Match metadata
            events: Ball-by-ball events
            winner: Actual match winner (for P&L settlement)

        Returns:
            BacktestResult with performance metrics
        """
        # Determine winner from data if not provided
        if winner is None:
            winner = self._infer_winner(match_info, events)

        result = BacktestResult(
            match_id=match_info.match_id,
            team_a=match_info.team_a,
            team_b=match_info.team_b,
            winner=winner,
            format=match_info.format,
        )

        # Initialize pipeline components
        state_engine = MatchStateEngine(match_info)
        pricing_model = EnsemblePricingModel(
            config=self._config.model,
            match_format=match_info.format,
        )
        signal_gen = SignalGenerator(self._config.signal)
        execution = ExecutionEngine(
            risk_config=self._config.risk,
            bankroll=self._config.bankroll,
            paper_mode=True,
        )
        exchange = SimulatedExchangeFeed()
        exchange.connect()

        # Market maker (if enabled)
        mm = MarketMaker(bankroll=self._config.bankroll) if self._enable_mm else None

        market_id = f"mkt_{match_info.match_id}"
        exchange.subscribe_market(market_id)

        # Process each ball
        for event in events:
            # Update match state
            match_state = state_engine.process_ball(event)
            features = state_engine.get_features()

            # Determine who is batting relative to team_a
            batting_is_team_a = event.batting_team == match_info.team_a

            # Get model prediction
            prediction = pricing_model.predict(features, batting_is_team_a)

            # Inject simulated market prices (model + noise to simulate market)
            noise = random.gauss(0, 0.02)
            market_prob_a = max(0.02, min(0.98, prediction.team_a_win_prob + noise))
            market_prob_b = 1.0 - market_prob_a
            exchange.inject_prices(
                market_id, match_info.match_id,
                {match_info.team_a: market_prob_a, match_info.team_b: market_prob_b},
            )
            market_state = exchange.get_market_state(market_id)

            # Directional signals
            if self._enable_directional:
                signals = signal_gen.generate_signals(
                    match_state, market_state, prediction
                )
                result.signals_generated += len(signals)

                for sig in signals:
                    position = execution.execute_signal(sig)
                    if position:
                        result.trades_executed += 1

                # Check stop losses
                current_odds = {
                    match_info.team_a: 1.0 / market_prob_a if market_prob_a > 0 else 999,
                    match_info.team_b: 1.0 / market_prob_b if market_prob_b > 0 else 999,
                }
                execution.check_stop_losses(match_info.match_id, current_odds)

                # Try to green-book profitable positions
                for pos in execution.get_match_positions(match_info.match_id):
                    if pos.status == PositionStatus.OPEN:
                        odds = current_odds.get(pos.selection_name, 0)
                        if odds > 0:
                            execution.green_book(pos.position_id, odds)

            # Market-making quotes
            if mm and self._enable_mm:
                quotes = mm.generate_quotes(
                    match_info.match_id,
                    prediction,
                    match_info.team_a,
                    match_info.team_b,
                )
                # Simulate fills based on market noise
                for quote in quotes:
                    fill_chance = abs(noise) * 10  # Higher noise = more fills
                    if random.random() < min(0.3, fill_chance):
                        # Random side fill
                        if random.random() < 0.5:
                            mm.on_fill(
                                match_info.match_id,
                                quote.selection_name,
                                SignalDirection.BACK,
                                quote.back_price,
                                quote.back_stake * random.uniform(0.3, 1.0),
                            )
                        else:
                            mm.on_fill(
                                match_info.match_id,
                                quote.selection_name,
                                SignalDirection.LAY,
                                quote.lay_price,
                                quote.lay_stake * random.uniform(0.3, 1.0),
                            )

        # Settlement: close all open positions based on match result
        if self._enable_directional:
            self._settle_match(execution, match_info, winner)
            perf = execution.get_performance_summary()
            result.directional_pnl = perf["total_pnl"]
            result.trades_won = perf.get("total_trades", 0)  # Approximate
            # Count from trade log
            for trade in execution.get_trade_log():
                if trade["pnl"] > 0:
                    result.trades_won += 1
                else:
                    result.trades_lost += 1

        if mm and self._enable_mm:
            mm_perf = mm.get_performance()
            result.mm_trades = mm_perf["total_trades"]
            result.mm_spread_captured = mm_perf["total_spread_captured"]
            result.mm_pnl = mm_perf["total_spread_captured"]

        result.total_pnl = result.directional_pnl + result.mm_pnl

        exchange.disconnect()
        return result

    def run_directory(
        self,
        directory: str | Path,
        match_format: Optional[str] = None,
        max_matches: Optional[int] = None,
    ) -> BacktestSummary:
        """Backtest all matches in a directory.

        Args:
            directory: Path to Cricsheet CSV files
            match_format: Filter by format (t20, odi, test)
            max_matches: Maximum matches to process

        Returns:
            BacktestSummary with aggregate metrics
        """
        directory = Path(directory)
        matches = load_matches_from_directory(
            directory, match_format=match_format, max_matches=max_matches
        )

        summary = BacktestSummary()

        for i, (info, events) in enumerate(matches):
            logger.info(
                "Backtesting match %d/%d: %s vs %s",
                i + 1, len(matches), info.team_a, info.team_b,
            )
            try:
                result = self.run_match(info, events)
                summary.matches.append(result)
                logger.info(
                    "  -> P&L: £%.2f (dir: £%.2f, mm: £%.2f)",
                    result.total_pnl, result.directional_pnl, result.mm_pnl,
                )
            except Exception as e:
                logger.error("Failed to backtest %s: %s", info.match_id, e)
                continue

        return summary

    def _settle_match(
        self,
        execution: ExecutionEngine,
        match_info: MatchInfo,
        winner: str,
    ) -> None:
        """Settle all open positions based on match result."""
        for pos in execution.get_match_positions(match_info.match_id):
            if pos.status != PositionStatus.OPEN:
                continue

            # Determine final settlement odds
            if winner == pos.selection_name:
                # This selection won → odds = 1.0
                final_odds = 1.0
            else:
                # This selection lost → odds very high (loser)
                final_odds = 100.0

            execution.close_position(
                pos.position_id, final_odds, reason="settlement"
            )

    def _infer_winner(
        self, match_info: MatchInfo, events: list[BallEvent]
    ) -> str:
        """Infer match winner from ball-by-ball data."""
        innings_scores: dict[str, int] = {}
        for event in events:
            team = event.batting_team
            innings_scores[team] = max(
                innings_scores.get(team, 0), event.cumulative_score
            )

        if not innings_scores:
            return match_info.team_a

        # In limited overs, higher score wins
        # (simplification - doesn't handle DLS, ties, etc.)
        return max(innings_scores, key=innings_scores.get)  # type: ignore
