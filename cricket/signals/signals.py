"""
Signal Generator - Layer 4.

Continuously compares ensemble model price to live market price
and emits trading signals when edge exceeds defined thresholds.

Implements four signal types:
1. Overreaction Trade - wicket-driven price spikes
2. Divergence Trade - model vs market probability gap
3. Pattern Trade - support/resistance band exploitation
4. Powerplay Lay - systematic first-6-overs strategy
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from cricket.config import SignalConfig
from cricket.data.exchange_feed import MarketSnapshot, MarketState
from cricket.models.ensemble import EnsemblePrediction
from cricket.state.match_state import MatchState

logger = logging.getLogger(__name__)


class SignalType(Enum):
    OVERREACTION = "overreaction"
    DIVERGENCE = "divergence"
    PATTERN = "pattern"
    POWERPLAY_LAY = "powerplay_lay"


class SignalDirection(Enum):
    BACK = "back"  # Bet on this outcome (buy)
    LAY = "lay"  # Bet against this outcome (sell)


@dataclass
class TradeSignal:
    """A trading signal emitted by the signal generator."""

    signal_id: str
    signal_type: SignalType
    direction: SignalDirection
    selection_name: str  # Team name

    # Pricing
    model_probability: float
    market_probability: float
    model_fair_odds: float
    market_odds: float
    edge_probability: float  # model_prob - market_prob
    edge_ticks: int  # Edge in number of price ticks

    # Context
    match_id: str
    innings: int
    over: float
    score: int
    wickets: int
    confidence: str

    # Sizing guidance
    suggested_stake_pct: float = 0.01  # % of bankroll
    max_odds: float = 0.0  # Maximum acceptable entry price
    stop_loss_odds: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def edge_pct(self) -> float:
        """Edge as a percentage."""
        return abs(self.edge_probability) * 100


class OverreactionDetector:
    """Detects overreaction opportunities after wicket events.

    When a wicket falls, the market often overreacts, moving the price
    further than justified. This detector identifies when the market
    price exceeds the model's expected adjustment.
    """

    # Expected price movement per wicket (in probability points)
    # Varies by match phase and batting position
    EXPECTED_WICKET_IMPACT = {
        # (phase, batting_position_range) -> expected prob change
        ("powerplay", "top_order"): 0.06,
        ("powerplay", "middle_order"): 0.04,
        ("middle", "top_order"): 0.05,
        ("middle", "middle_order"): 0.04,
        ("middle", "lower_order"): 0.03,
        ("death", "top_order"): 0.04,
        ("death", "middle_order"): 0.03,
        ("death", "lower_order"): 0.02,
    }

    def __init__(self, config: SignalConfig):
        self._config = config
        self._pre_wicket_prices: dict[str, float] = {}  # match_id -> pre-wicket prob
        self._wicket_timestamps: dict[str, float] = {}

    def record_pre_wicket_state(
        self, match_id: str, market_probability: float
    ) -> None:
        """Record market state before a wicket (called every ball)."""
        self._pre_wicket_prices[match_id] = market_probability

    def check_overreaction(
        self,
        match_id: str,
        wickets: int,
        phase: str,
        market_prob_after: float,
        model_prob_after: float,
    ) -> Optional[tuple[float, float]]:
        """Check if market has overreacted to a wicket.

        Returns (market_move, expected_move) if overreaction detected,
        None otherwise.
        """
        pre_wicket_prob = self._pre_wicket_prices.get(match_id)
        if pre_wicket_prob is None:
            return None

        market_move = abs(market_prob_after - pre_wicket_prob)

        # Get expected impact based on phase
        batting_pos = "top_order" if wickets <= 3 else (
            "middle_order" if wickets <= 6 else "lower_order"
        )
        key = (phase, batting_pos)
        expected_move = self.EXPECTED_WICKET_IMPACT.get(key, 0.04)

        # Overreaction if market moved more than expected
        if market_move > expected_move * 1.5:
            return (market_move, expected_move)

        return None


class PriceBandTracker:
    """Tracks support/resistance bands in exchange prices.

    During stable match phases, prices oscillate between predictable
    floors and ceilings. This tracker identifies these bands.
    """

    def __init__(self, window_size: int = 60, min_touches: int = 3):
        self._prices: deque[float] = deque(maxlen=window_size)
        self._min_touches = min_touches

    def add_price(self, price: float) -> None:
        self._prices.append(price)

    def get_bands(self) -> Optional[tuple[float, float]]:
        """Identify current support/resistance band.

        Returns (support, resistance) if a band is detected,
        None otherwise.
        """
        if len(self._prices) < 20:
            return None

        prices = list(self._prices)
        # Simple approach: use recent min/max with touch count
        recent = prices[-30:] if len(prices) >= 30 else prices

        price_min = min(recent)
        price_max = max(recent)
        band_width = price_max - price_min

        if band_width < 0.04:  # Band too narrow
            return None

        # Count touches near support/resistance
        tolerance = band_width * 0.15
        support_touches = sum(1 for p in recent if abs(p - price_min) < tolerance)
        resistance_touches = sum(1 for p in recent if abs(p - price_max) < tolerance)

        if (
            support_touches >= self._min_touches
            and resistance_touches >= self._min_touches
        ):
            return (price_min, price_max)

        return None


class SignalGenerator:
    """Main signal generation engine.

    Consumes match state + market state + model predictions
    and emits trading signals when edge thresholds are met.
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self._config = config or SignalConfig()
        self._overreaction = OverreactionDetector(self._config)
        self._price_bands: dict[str, PriceBandTracker] = {}  # selection -> tracker
        self._signal_counter = 0
        self._last_wicket_count: dict[str, int] = {}  # match_id -> wickets
        self._powerplay_positions: dict[str, bool] = {}  # match_id -> has_position

    def generate_signals(
        self,
        match_state: MatchState,
        market_state: Optional[MarketState],
        prediction: EnsemblePrediction,
    ) -> list[TradeSignal]:
        """Generate all applicable trading signals.

        This is the main entry point called after each ball event.

        Args:
            match_state: Current match state from Layer 2
            market_state: Current exchange prices from Layer 1
            prediction: Model prediction from Layer 3

        Returns:
            List of trade signals (may be empty)
        """
        signals: list[TradeSignal] = []
        match_id = match_state.match_info.match_id
        inn = match_state.current_innings_state

        # Get market probabilities
        if market_state:
            team_a_snap = market_state.get_selection(match_state.match_info.team_a)
            team_b_snap = market_state.get_selection(match_state.match_info.team_b)
        else:
            team_a_snap = None
            team_b_snap = None

        market_prob_a = (
            team_a_snap.implied_probability if team_a_snap else prediction.team_a_win_prob
        )
        market_prob_b = (
            team_b_snap.implied_probability if team_b_snap else prediction.team_b_win_prob
        )

        market_odds_a = 1.0 / market_prob_a if market_prob_a > 0 else 999.0
        market_odds_b = 1.0 / market_prob_b if market_prob_b > 0 else 999.0

        # Determine phase
        overs = inn.legal_balls / 6.0
        if overs <= 6:
            phase = "powerplay"
        elif overs <= 15:
            phase = "middle"
        else:
            phase = "death"

        # Signal 1: Overreaction
        prev_wickets = self._last_wicket_count.get(match_id, 0)
        if inn.wickets > prev_wickets:
            # Wicket just fell
            overreaction = self._overreaction.check_overreaction(
                match_id, inn.wickets, phase,
                market_prob_a, prediction.team_a_win_prob,
            )
            if overreaction:
                market_move, expected_move = overreaction
                sig = self._create_overreaction_signal(
                    match_state, prediction, market_prob_a, market_odds_a,
                    market_move, expected_move,
                )
                if sig:
                    signals.append(sig)

        self._last_wicket_count[match_id] = inn.wickets
        self._overreaction.record_pre_wicket_state(match_id, market_prob_a)

        # Signal 2: Divergence
        div_signal = self._check_divergence(
            match_state, prediction,
            market_prob_a, market_prob_b,
            market_odds_a, market_odds_b,
        )
        if div_signal:
            signals.append(div_signal)

        # Signal 3: Pattern (support/resistance)
        if team_a_snap and team_a_snap.mid_price:
            key = f"{match_id}_a"
            if key not in self._price_bands:
                self._price_bands[key] = PriceBandTracker(
                    min_touches=self._config.pattern_min_touches
                )
            self._price_bands[key].add_price(team_a_snap.mid_price)
            pattern_sig = self._check_pattern(
                match_state, prediction, key,
                team_a_snap.mid_price, match_state.match_info.team_a,
            )
            if pattern_sig:
                signals.append(pattern_sig)

        # Signal 4: Powerplay Lay
        if (
            overs < 6
            and match_state.current_innings <= 2
            and not self._powerplay_positions.get(
                f"{match_id}_{match_state.current_innings}"
            )
        ):
            pp_signal = self._check_powerplay_lay(
                match_state, prediction,
                market_prob_a, market_odds_a,
            )
            if pp_signal:
                signals.append(pp_signal)
                self._powerplay_positions[
                    f"{match_id}_{match_state.current_innings}"
                ] = True

        return signals

    def _create_overreaction_signal(
        self,
        match_state: MatchState,
        prediction: EnsemblePrediction,
        market_prob: float,
        market_odds: float,
        market_move: float,
        expected_move: float,
    ) -> Optional[TradeSignal]:
        """Create an overreaction trade signal."""
        inn = match_state.current_innings_state

        # Direction: if market moved against batting team more than expected,
        # back the batting team (mean reversion expected)
        edge = prediction.team_a_win_prob - market_prob
        if abs(edge) < 0.03:
            return None

        direction = SignalDirection.BACK if edge > 0 else SignalDirection.LAY
        selection = (
            match_state.match_info.team_a
            if edge > 0
            else match_state.match_info.team_b
        )

        edge_ticks = int(abs(edge) / 0.01)
        if edge_ticks < self._config.overreaction_min_edge_ticks:
            return None

        self._signal_counter += 1

        return TradeSignal(
            signal_id=f"OR-{self._signal_counter:05d}",
            signal_type=SignalType.OVERREACTION,
            direction=direction,
            selection_name=selection,
            model_probability=prediction.team_a_win_prob if edge > 0 else prediction.team_b_win_prob,
            market_probability=market_prob if edge > 0 else 1.0 - market_prob,
            model_fair_odds=prediction.team_a_fair_odds if edge > 0 else prediction.team_b_fair_odds,
            market_odds=market_odds,
            edge_probability=abs(edge),
            edge_ticks=edge_ticks,
            match_id=match_state.match_info.match_id,
            innings=match_state.current_innings,
            over=inn.legal_balls / 6.0,
            score=inn.score,
            wickets=inn.wickets,
            confidence=prediction.confidence,
            suggested_stake_pct=0.015 if prediction.confidence == "HIGH" else 0.01,
            reason=f"Wicket overreaction: market moved {market_move:.3f} vs expected {expected_move:.3f}",
            metadata={
                "market_move": market_move,
                "expected_move": expected_move,
                "overreaction_ratio": market_move / expected_move if expected_move > 0 else 0,
            },
        )

    def _check_divergence(
        self,
        match_state: MatchState,
        prediction: EnsemblePrediction,
        market_prob_a: float,
        market_prob_b: float,
        market_odds_a: float,
        market_odds_b: float,
    ) -> Optional[TradeSignal]:
        """Check for model-market divergence signal."""
        inn = match_state.current_innings_state

        edge_a = prediction.team_a_win_prob - market_prob_a
        edge_b = prediction.team_b_win_prob - market_prob_b

        # Take the larger edge
        if abs(edge_a) >= abs(edge_b):
            edge = edge_a
            selection = match_state.match_info.team_a
            model_prob = prediction.team_a_win_prob
            market_prob = market_prob_a
            market_odds = market_odds_a
            fair_odds = prediction.team_a_fair_odds
        else:
            edge = edge_b
            selection = match_state.match_info.team_b
            model_prob = prediction.team_b_win_prob
            market_prob = market_prob_b
            market_odds = market_odds_b
            fair_odds = prediction.team_b_fair_odds

        if abs(edge) < self._config.divergence_min_probability_gap:
            return None

        if prediction.confidence == "LOW":
            return None

        direction = SignalDirection.BACK if edge > 0 else SignalDirection.LAY
        edge_ticks = int(abs(edge) / 0.01)

        self._signal_counter += 1

        return TradeSignal(
            signal_id=f"DV-{self._signal_counter:05d}",
            signal_type=SignalType.DIVERGENCE,
            direction=direction,
            selection_name=selection,
            model_probability=model_prob,
            market_probability=market_prob,
            model_fair_odds=fair_odds,
            market_odds=market_odds,
            edge_probability=abs(edge),
            edge_ticks=edge_ticks,
            match_id=match_state.match_info.match_id,
            innings=match_state.current_innings,
            over=inn.legal_balls / 6.0,
            score=inn.score,
            wickets=inn.wickets,
            confidence=prediction.confidence,
            suggested_stake_pct=0.02 if prediction.confidence == "HIGH" else 0.01,
            reason=f"Model-market divergence: {abs(edge)*100:.1f}% gap ({selection})",
        )

    def _check_pattern(
        self,
        match_state: MatchState,
        prediction: EnsemblePrediction,
        band_key: str,
        current_price: float,
        selection_name: str,
    ) -> Optional[TradeSignal]:
        """Check for support/resistance pattern signal."""
        tracker = self._price_bands.get(band_key)
        if not tracker:
            return None

        bands = tracker.get_bands()
        if not bands:
            return None

        support, resistance = bands
        band_width = resistance - support
        inn = match_state.current_innings_state

        # Check if price is near support (back) or resistance (lay)
        near_support = current_price < support + band_width * 0.2
        near_resistance = current_price > resistance - band_width * 0.2

        if not near_support and not near_resistance:
            return None

        if near_support:
            direction = SignalDirection.BACK
            edge_ticks = int((current_price - support) / 0.02)
        else:
            direction = SignalDirection.LAY
            edge_ticks = int((resistance - current_price) / 0.02)

        if edge_ticks < self._config.pattern_min_edge_ticks:
            return None

        implied_prob = 1.0 / current_price if current_price > 0 else 0.5

        self._signal_counter += 1

        return TradeSignal(
            signal_id=f"PT-{self._signal_counter:05d}",
            signal_type=SignalType.PATTERN,
            direction=direction,
            selection_name=selection_name,
            model_probability=prediction.team_a_win_prob,
            market_probability=implied_prob,
            model_fair_odds=prediction.team_a_fair_odds,
            market_odds=current_price,
            edge_probability=abs(prediction.team_a_win_prob - implied_prob),
            edge_ticks=edge_ticks,
            match_id=match_state.match_info.match_id,
            innings=match_state.current_innings,
            over=inn.legal_balls / 6.0,
            score=inn.score,
            wickets=inn.wickets,
            confidence="MEDIUM",
            suggested_stake_pct=0.008,
            reason=f"Price band: support={support:.2f}, resistance={resistance:.2f}",
            metadata={"support": support, "resistance": resistance},
        )

    def _check_powerplay_lay(
        self,
        match_state: MatchState,
        prediction: EnsemblePrediction,
        market_prob: float,
        market_odds: float,
    ) -> Optional[TradeSignal]:
        """Check for powerplay lay signal.

        Systematic strategy: lay the batting team at start of powerplay.
        ~78% of T20 matches see at least one wicket in overs 1-6.
        """
        inn = match_state.current_innings_state
        overs = inn.legal_balls / 6.0

        # Only trigger in first 2 overs before market has adjusted
        if overs > 2.0:
            return None

        # Don't lay if team is already significant underdog
        if market_prob < 0.35:
            return None

        self._signal_counter += 1

        return TradeSignal(
            signal_id=f"PP-{self._signal_counter:05d}",
            signal_type=SignalType.POWERPLAY_LAY,
            direction=SignalDirection.LAY,
            selection_name=inn.batting_team,
            model_probability=prediction.team_a_win_prob,
            market_probability=market_prob,
            model_fair_odds=prediction.team_a_fair_odds,
            market_odds=market_odds,
            edge_probability=self._config.powerplay_wicket_probability - 0.5,
            edge_ticks=3,
            match_id=match_state.match_info.match_id,
            innings=match_state.current_innings,
            over=overs,
            score=inn.score,
            wickets=inn.wickets,
            confidence="MEDIUM",
            suggested_stake_pct=0.01,
            reason="Powerplay lay: 78% wicket probability in overs 1-6",
            metadata={"wicket_probability": self._config.powerplay_wicket_probability},
        )
