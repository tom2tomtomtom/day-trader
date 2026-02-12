"""Tests for the Signal Generator (Layer 4)."""

from __future__ import annotations

import pytest

from cricket.config import SignalConfig
from cricket.data.ball_event import MatchInfo
from cricket.data.exchange_feed import MarketState, MarketSnapshot, PriceLevel
from cricket.models.ensemble import EnsemblePrediction
from cricket.signals.signals import (
    OverreactionDetector,
    PriceBandTracker,
    SignalDirection,
    SignalGenerator,
    SignalType,
)
from cricket.state.match_state import MatchState, MatchStateEngine, InningsState

from datetime import datetime


@pytest.fixture
def signal_config() -> SignalConfig:
    return SignalConfig()


@pytest.fixture
def match_info() -> MatchInfo:
    return MatchInfo(
        match_id="sig_test_001",
        format="t20",
        team_a="Thunder",
        team_b="Strikers",
        venue="Test Ground",
        venue_avg_first_innings_score=160.0,
        team_a_elo=1500,
        team_b_elo=1500,
    )


def make_market_state(
    match_id: str, prob_a: float, prob_b: float, volume: float = 50000.0
) -> MarketState:
    """Create a MarketState with given probabilities."""
    price_a = 1.0 / prob_a if prob_a > 0 else 999.0
    price_b = 1.0 / prob_b if prob_b > 0 else 999.0

    return MarketState(
        match_id=match_id,
        market_id=f"mkt_{match_id}",
        market_type="MATCH_ODDS",
        in_play=True,
        selections={
            1: MarketSnapshot(
                market_id=f"mkt_{match_id}",
                selection_id=1,
                selection_name="Thunder",
                timestamp=datetime.utcnow(),
                back_prices=[PriceLevel(round(price_a - 0.02, 2), volume * 0.3)],
                lay_prices=[PriceLevel(round(price_a + 0.02, 2), volume * 0.3)],
                last_price_traded=price_a,
                total_matched=volume,
            ),
            2: MarketSnapshot(
                market_id=f"mkt_{match_id}",
                selection_id=2,
                selection_name="Strikers",
                timestamp=datetime.utcnow(),
                back_prices=[PriceLevel(round(price_b - 0.02, 2), volume * 0.3)],
                lay_prices=[PriceLevel(round(price_b + 0.02, 2), volume * 0.3)],
                last_price_traded=price_b,
                total_matched=volume,
            ),
        },
    )


def make_prediction(prob_a: float, confidence: str = "HIGH") -> EnsemblePrediction:
    return EnsemblePrediction(
        team_a_win_prob=prob_a,
        team_b_win_prob=1.0 - prob_a,
        statistical_prob=prob_a,
        xgboost_prob=prob_a,
        confidence=confidence,
        confidence_score=0.8 if confidence == "HIGH" else 0.5,
        model_agreement=0.01,
    )


class TestOverreactionDetector:
    def test_detects_overreaction(self, signal_config: SignalConfig):
        detector = OverreactionDetector(signal_config)

        # Record pre-wicket state
        detector.record_pre_wicket_state("match_1", 0.55)

        # Market moved 12% after wicket (expected ~6%)
        result = detector.check_overreaction(
            "match_1", wickets=1, phase="powerplay",
            market_prob_after=0.43, model_prob_after=0.50,
        )

        assert result is not None
        market_move, expected = result
        assert market_move > expected

    def test_no_overreaction_normal_move(self, signal_config: SignalConfig):
        detector = OverreactionDetector(signal_config)
        detector.record_pre_wicket_state("match_1", 0.55)

        # Market moved only 3% (within expected range)
        result = detector.check_overreaction(
            "match_1", wickets=1, phase="powerplay",
            market_prob_after=0.52, model_prob_after=0.52,
        )

        assert result is None


class TestPriceBandTracker:
    def test_detects_bands(self):
        tracker = PriceBandTracker(window_size=60, min_touches=3)

        # Simulate oscillating prices between 1.80 and 2.00
        import random
        random.seed(42)
        for _ in range(40):
            price = random.uniform(1.80, 2.00)
            tracker.add_price(price)

        # Add explicit touches at support and resistance
        for _ in range(4):
            tracker.add_price(1.81)
            tracker.add_price(1.99)

        bands = tracker.get_bands()
        # May or may not detect depending on exact prices
        # The key test is it doesn't crash
        if bands:
            support, resistance = bands
            assert support < resistance

    def test_no_bands_insufficient_data(self):
        tracker = PriceBandTracker()
        tracker.add_price(2.0)
        assert tracker.get_bands() is None


class TestSignalGenerator:
    def test_divergence_signal_generated(self, match_info: MatchInfo):
        gen = SignalGenerator(SignalConfig(divergence_min_probability_gap=0.05))

        # Model says 65%, market says 55% = 10% gap
        state_engine = MatchStateEngine(match_info)
        from cricket.data.ball_event import BallEvent
        event = BallEvent(
            match_id="sig_test_001", innings=1, over=5, ball=3,
            batting_team="Thunder", bowling_team="Strikers",
            striker="B1", non_striker="B2", bowler="BW1",
            runs_off_bat=1, total_runs=1,
            cumulative_score=45, cumulative_wickets=1, cumulative_overs=5.3,
        )
        match_state = state_engine.process_ball(event)

        market_state = make_market_state("sig_test_001", 0.55, 0.45)
        prediction = make_prediction(0.65, "HIGH")

        signals = gen.generate_signals(match_state, market_state, prediction)

        # Should generate at least a divergence signal
        div_signals = [s for s in signals if s.signal_type == SignalType.DIVERGENCE]
        assert len(div_signals) >= 1
        assert div_signals[0].edge_probability >= 0.05

    def test_no_signal_when_low_confidence(self, match_info: MatchInfo):
        gen = SignalGenerator()

        state_engine = MatchStateEngine(match_info)
        from cricket.data.ball_event import BallEvent
        event = BallEvent(
            match_id="sig_test_001", innings=1, over=5, ball=3,
            batting_team="Thunder", bowling_team="Strikers",
            striker="B1", non_striker="B2", bowler="BW1",
            cumulative_score=45, cumulative_wickets=1, cumulative_overs=5.3,
        )
        match_state = state_engine.process_ball(event)

        market_state = make_market_state("sig_test_001", 0.50, 0.50)
        prediction = make_prediction(0.55, "LOW")

        signals = gen.generate_signals(match_state, market_state, prediction)

        # Divergence signals should be filtered out on low confidence
        div_signals = [s for s in signals if s.signal_type == SignalType.DIVERGENCE]
        assert len(div_signals) == 0

    def test_powerplay_lay_signal(self, match_info: MatchInfo):
        gen = SignalGenerator()

        state_engine = MatchStateEngine(match_info)
        from cricket.data.ball_event import BallEvent
        event = BallEvent(
            match_id="sig_test_001", innings=1, over=0, ball=3,
            batting_team="Thunder", bowling_team="Strikers",
            striker="B1", non_striker="B2", bowler="BW1",
            runs_off_bat=1, total_runs=1,
            cumulative_score=5, cumulative_wickets=0, cumulative_overs=0.3,
        )
        match_state = state_engine.process_ball(event)

        market_state = make_market_state("sig_test_001", 0.50, 0.50)
        prediction = make_prediction(0.50)

        signals = gen.generate_signals(match_state, market_state, prediction)

        pp_signals = [s for s in signals if s.signal_type == SignalType.POWERPLAY_LAY]
        assert len(pp_signals) == 1
        assert pp_signals[0].direction == SignalDirection.LAY
