"""Tests for the Market-Making Engine."""

from __future__ import annotations

import pytest

from cricket.execution.market_maker import MarketMaker, MarketMakerConfig
from cricket.models.ensemble import EnsemblePrediction
from cricket.signals.signals import SignalDirection


def make_prediction(prob_a: float = 0.5, confidence: str = "HIGH") -> EnsemblePrediction:
    return EnsemblePrediction(
        team_a_win_prob=prob_a,
        team_b_win_prob=1.0 - prob_a,
        statistical_prob=prob_a,
        xgboost_prob=prob_a,
        confidence=confidence,
        confidence_score=0.8,
        model_agreement=0.01,
    )


class TestMarketMaker:
    def test_generates_two_sided_quotes(self):
        mm = MarketMaker(bankroll=10000.0)
        prediction = make_prediction(0.55)

        quotes = mm.generate_quotes(
            "match_1", prediction, "Thunder", "Strikers", market_volume=50000.0
        )

        assert len(quotes) == 2

        for quote in quotes:
            assert quote.back_price > 0
            assert quote.lay_price > quote.back_price
            assert quote.back_stake > 0
            assert quote.lay_stake > 0
            assert quote.spread > 0

    def test_no_quotes_low_liquidity(self):
        mm = MarketMaker(config=MarketMakerConfig(min_liquidity=10000.0))
        prediction = make_prediction()

        quotes = mm.generate_quotes(
            "match_1", prediction, "Thunder", "Strikers", market_volume=1000.0
        )

        assert len(quotes) == 0

    def test_wider_spread_low_confidence(self):
        mm = MarketMaker(bankroll=10000.0)
        pred_high = make_prediction(0.55, "HIGH")
        pred_low = make_prediction(0.55, "LOW")

        quotes_high = mm.generate_quotes(
            "match_1", pred_high, "Thunder", "Strikers", market_volume=50000.0
        )
        quotes_low = mm.generate_quotes(
            "match_2", pred_low, "A", "B", market_volume=50000.0
        )

        # Low confidence should have wider spread
        spread_high = quotes_high[0].spread
        spread_low = quotes_low[0].spread
        assert spread_low > spread_high

    def test_inventory_tracking(self):
        mm = MarketMaker(bankroll=10000.0)
        prediction = make_prediction()
        mm.generate_quotes("m1", prediction, "Thunder", "Strikers", 50000.0)

        # Simulate back fill
        mm.on_fill("m1", "Thunder", SignalDirection.BACK, 1.80, 50.0)
        inv = mm._get_inventory("Thunder")
        assert inv.net_exposure == 50.0
        assert inv.is_long

        # Simulate lay fill
        mm.on_fill("m1", "Thunder", SignalDirection.LAY, 1.85, 60.0)
        assert inv.net_exposure == -10.0
        assert inv.is_short

    def test_inventory_skew(self):
        mm = MarketMaker(bankroll=10000.0)
        prediction = make_prediction()

        # Create large long position
        mm.generate_quotes("m1", prediction, "Thunder", "Strikers", 50000.0)
        for _ in range(10):
            mm.on_fill("m1", "Thunder", SignalDirection.BACK, 2.0, 30.0)

        # Skew should be negative (discourage more buying)
        skew = mm._calculate_inventory_skew("Thunder")
        assert skew < 0

    def test_should_flatten(self):
        config = MarketMakerConfig(max_inventory=200.0)
        mm = MarketMaker(config=config, bankroll=10000.0)
        prediction = make_prediction()
        mm.generate_quotes("m1", prediction, "Thunder", "Strikers", 50000.0)

        for _ in range(10):
            mm.on_fill("m1", "Thunder", SignalDirection.BACK, 2.0, 30.0)

        assert mm.should_flatten("Thunder")

    def test_performance_tracking(self):
        mm = MarketMaker(bankroll=10000.0)
        perf = mm.get_performance()
        assert perf["total_trades"] == 0
        assert perf["total_volume"] == 0.0

        prediction = make_prediction()
        mm.generate_quotes("m1", prediction, "Thunder", "Strikers", 50000.0)
        mm.on_fill("m1", "Thunder", SignalDirection.BACK, 1.80, 50.0)

        perf = mm.get_performance()
        assert perf["total_trades"] == 1
        assert perf["total_volume"] > 0

    def test_match_loss_limit(self):
        config = MarketMakerConfig(max_loss_per_match=100.0)
        mm = MarketMaker(config=config, bankroll=10000.0)
        mm._match_pnl["m1"] = -150.0  # Exceeded loss limit

        prediction = make_prediction()
        quotes = mm.generate_quotes("m1", prediction, "Thunder", "Strikers", 50000.0)
        assert len(quotes) == 0  # Should refuse to quote
