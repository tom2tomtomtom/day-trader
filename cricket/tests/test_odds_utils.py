"""Tests for odds utility functions."""

from __future__ import annotations

import pytest

from cricket.utils.odds import (
    calculate_overround,
    kelly_stake,
    move_price,
    odds_to_probability,
    probability_to_odds,
    remove_overround,
    snap_to_betfair_price,
    ticks_between,
)


class TestOddsConversion:
    def test_probability_to_odds(self):
        assert probability_to_odds(0.5) == 2.0
        assert probability_to_odds(0.25) == 4.0
        assert probability_to_odds(1.0) == 1.01  # Capped

    def test_odds_to_probability(self):
        assert odds_to_probability(2.0) == pytest.approx(0.5)
        assert odds_to_probability(4.0) == pytest.approx(0.25)

    def test_roundtrip(self):
        for prob in [0.1, 0.25, 0.5, 0.75, 0.9]:
            odds = probability_to_odds(prob)
            back = odds_to_probability(odds)
            assert back == pytest.approx(prob, abs=0.01)


class TestBetfairPriceLadder:
    def test_snap_to_tick(self):
        assert snap_to_betfair_price(1.55) == 1.55
        assert snap_to_betfair_price(2.03) == 2.02  # Rounds to nearest 0.02 increment
        assert snap_to_betfair_price(3.07) == 3.05  # Rounds to 0.05 increment

    def test_snap_round_up(self):
        snapped = snap_to_betfair_price(1.995, round_up=True)
        assert snapped >= 1.995

    def test_ticks_between(self):
        ticks = ticks_between(1.50, 1.55)
        assert ticks == 5  # 0.01 increments

    def test_move_price_up(self):
        price = move_price(1.50, 3)
        assert price == 1.53

    def test_move_price_down(self):
        price = move_price(1.50, -3)
        assert price == 1.47


class TestOverround:
    def test_fair_market(self):
        assert calculate_overround([0.5, 0.5]) == pytest.approx(0.0)

    def test_market_with_margin(self):
        overround = calculate_overround([0.52, 0.52])
        assert overround > 0

    def test_remove_overround(self):
        probs = remove_overround([0.52, 0.52])
        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.5)


class TestKellyStake:
    def test_positive_edge(self):
        stake = kelly_stake(
            probability=0.55,
            odds=2.0,
            bankroll=10000.0,
            fraction=0.25,
        )
        assert stake > 0

    def test_no_edge(self):
        stake = kelly_stake(
            probability=0.50,
            odds=2.0,
            bankroll=10000.0,
        )
        assert stake == pytest.approx(0.0, abs=1.0)

    def test_negative_edge(self):
        stake = kelly_stake(
            probability=0.40,
            odds=2.0,
            bankroll=10000.0,
        )
        assert stake == 0.0

    def test_fraction_reduces_stake(self):
        full = kelly_stake(0.60, 2.0, 10000.0, fraction=1.0)
        quarter = kelly_stake(0.60, 2.0, 10000.0, fraction=0.25)
        assert quarter < full
        assert quarter == pytest.approx(full * 0.25, abs=1.0)
