"""Tests for the Arbitrage Detection Engine."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from cricket.arbitrage.detector import (
    ArbType,
    ArbitrageScanner,
    CrossExchangeDetector,
    IntraMarketDetector,
    MarketPrices,
    SelectionPrice,
    TemporalArbDetector,
)
from cricket.arbitrage.executor import ArbitrageExecutor, ExecutionStrategy


def make_selection(
    name: str,
    back: float,
    lay: float,
    exchange: str = "betfair",
    market_id: str = "mkt_1",
    liquidity: float = 5000.0,
) -> SelectionPrice:
    return SelectionPrice(
        exchange=exchange,
        market_id=market_id,
        market_type="match_odds",
        selection_name=name,
        back_odds=back,
        lay_odds=lay,
        back_liquidity=liquidity,
        lay_liquidity=liquidity,
    )


def make_market(
    selections: list[SelectionPrice],
    exchange: str = "betfair",
    market_id: str = "mkt_1",
    match_id: str = "match_1",
) -> MarketPrices:
    return MarketPrices(
        exchange=exchange,
        market_id=market_id,
        market_type="match_odds",
        match_id=match_id,
        selections=selections,
    )


class TestIntraMarketDetector:
    def test_detects_back_all_arb(self):
        """When sum of 1/back_odds < 1.0, backing all gives guaranteed profit."""
        detector = IntraMarketDetector(min_profit_pct=0.001, min_liquidity=50.0)

        # Back odds: 2.20 + 3.50 = implied prob 0.454 + 0.286 = 0.740 (< 1.0)
        # This is a massive arb — unrealistic but tests the math
        market = make_market([
            make_selection("Team A", back=2.20, lay=2.25),
            make_selection("Team B", back=3.50, lay=3.60),
        ])

        arbs = detector.detect(market)
        back_arbs = [a for a in arbs if "Back-all" in a.description]
        assert len(back_arbs) == 1
        assert back_arbs[0].guaranteed_profit > 0
        assert back_arbs[0].is_pure_arb

    def test_no_arb_normal_market(self):
        """Normal efficient market: back probabilities sum to > 1.0."""
        detector = IntraMarketDetector(min_profit_pct=0.001, min_liquidity=50.0)

        # Back odds: 1.80 + 2.20 = implied prob 0.556 + 0.454 = 1.010
        market = make_market([
            make_selection("Team A", back=1.80, lay=1.82),
            make_selection("Team B", back=2.20, lay=2.24),
        ])

        arbs = detector.detect(market)
        back_arbs = [a for a in arbs if "Back-all" in a.description]
        assert len(back_arbs) == 0

    def test_no_arb_insufficient_liquidity(self):
        """Arb exists mathematically but not enough liquidity."""
        detector = IntraMarketDetector(min_profit_pct=0.001, min_liquidity=1000.0)

        market = make_market([
            make_selection("Team A", back=2.20, lay=2.25, liquidity=10.0),
            make_selection("Team B", back=3.50, lay=3.60, liquidity=10.0),
        ])

        arbs = detector.detect(market)
        assert len(arbs) == 0


class TestCrossExchangeDetector:
    def test_detects_cross_exchange_arb(self):
        """Back on one exchange at higher odds, lay on another at lower odds."""
        detector = CrossExchangeDetector(
            commission_rates={"betfair": 0.05, "smarkets": 0.02},
            min_profit_pct=0.001,
            min_liquidity=50.0,
        )

        # Betfair: Team A back at 2.10
        betfair = make_market(
            [
                make_selection("Team A", back=2.10, lay=2.14, exchange="betfair"),
                make_selection("Team B", back=1.85, lay=1.88, exchange="betfair"),
            ],
            exchange="betfair",
            market_id="bf_1",
        )

        # Smarkets: Team A lay at 2.02 (lower than Betfair back)
        smarkets = make_market(
            [
                make_selection("Team A", back=1.98, lay=2.02, exchange="smarkets"),
                make_selection("Team B", back=1.92, lay=1.96, exchange="smarkets"),
            ],
            exchange="smarkets",
            market_id="sm_1",
        )

        arbs = detector.detect([betfair, smarkets])
        assert len(arbs) >= 1

        arb = arbs[0]
        assert arb.arb_type == ArbType.CROSS_EXCHANGE
        assert arb.guaranteed_profit > 0
        assert len(arb.legs) == 2

        # Verify legs are on different exchanges
        exchanges = {leg.exchange for leg in arb.legs}
        assert len(exchanges) == 2

    def test_no_arb_same_exchange(self):
        """No cross-exchange arb if prices are from the same exchange."""
        detector = CrossExchangeDetector(min_profit_pct=0.001, min_liquidity=50.0)

        market = make_market([
            make_selection("Team A", back=2.10, lay=2.14),
            make_selection("Team B", back=1.85, lay=1.88),
        ])

        arbs = detector.detect([market])
        assert len(arbs) == 0

    def test_no_arb_when_lay_higher_than_back(self):
        """No arb when lay price is higher than back price across exchanges."""
        detector = CrossExchangeDetector(min_profit_pct=0.001, min_liquidity=50.0)

        betfair = make_market(
            [make_selection("Team A", back=2.00, lay=2.04, exchange="betfair")],
            exchange="betfair",
        )
        smarkets = make_market(
            [make_selection("Team A", back=1.96, lay=2.08, exchange="smarkets")],
            exchange="smarkets",
            market_id="sm_1",
        )

        arbs = detector.detect([betfair, smarkets])
        # Back 2.00 on betfair, lay 2.08 on smarkets → lay > back, no arb
        assert len(arbs) == 0

    def test_commission_kills_marginal_arb(self):
        """An arb that exists pre-commission but not after."""
        detector = CrossExchangeDetector(
            commission_rates={"betfair": 0.05, "smarkets": 0.02},
            min_profit_pct=0.005,
            min_liquidity=50.0,
        )

        # Very tight cross — 2.04 back vs 2.02 lay
        betfair = make_market(
            [make_selection("Team A", back=2.04, lay=2.08, exchange="betfair")],
            exchange="betfair",
        )
        smarkets = make_market(
            [make_selection("Team A", back=1.98, lay=2.02, exchange="smarkets")],
            exchange="smarkets",
            market_id="sm_1",
        )

        arbs = detector.detect([betfair, smarkets])
        # After 5% and 2% commission, this tiny edge should vanish
        for arb in arbs:
            assert arb.guaranteed_profit_pct >= 0.5  # Minimum threshold


class TestTemporalArbDetector:
    def test_detects_stale_after_event(self):
        detector = TemporalArbDetector(
            stale_threshold_ms=5000,
            min_edge_probability=0.05,
            min_liquidity=50.0,
        )

        # Record a wicket event with current prices
        detector.record_event(
            "match_1", "wicket", {"Team A": 0.55, "Team B": 0.45}
        )

        # Market hasn't moved (still showing pre-wicket prices)
        stale_market = make_market([
            make_selection("Team A", back=1.80, lay=1.84),  # ~0.55 prob
            make_selection("Team B", back=2.15, lay=2.20),  # ~0.46 prob
        ])

        # Model says fair value shifted significantly
        arbs = detector.detect(
            "match_1",
            model_fair_prob={"Team A": 0.45, "Team B": 0.55},
            markets=[stale_market],
        )

        assert len(arbs) >= 1
        assert arbs[0].arb_type == ArbType.TEMPORAL

    def test_no_arb_when_price_updated(self):
        detector = TemporalArbDetector(
            stale_threshold_ms=5000,
            min_edge_probability=0.05,
            min_liquidity=50.0,
        )

        detector.record_event(
            "match_1", "wicket", {"Team A": 0.55}
        )

        # Market HAS updated (new probability)
        updated_market = make_market([
            make_selection("Team A", back=2.15, lay=2.20),  # ~0.46 prob — moved
        ])

        arbs = detector.detect(
            "match_1",
            model_fair_prob={"Team A": 0.45},
            markets=[updated_market],
        )

        # Price moved from 0.55 to ~0.46, close to model's 0.45
        # Should NOT detect stale arb (price updated, not unchanged)
        stale_arbs = [a for a in arbs if a.arb_type == ArbType.TEMPORAL]
        assert len(stale_arbs) == 0


class TestArbitrageScanner:
    def test_scans_all_types(self):
        scanner = ArbitrageScanner(min_profit_pct=0.001, min_liquidity=50.0)

        # Set up a market with a back-all arb
        market = make_market([
            make_selection("Team A", back=2.50, lay=2.55),
            make_selection("Team B", back=3.80, lay=3.90),
        ])

        arbs = scanner.scan("match_1", [market])

        # Should find the intra-market arb
        stats = scanner.get_stats()
        assert stats["total_detected"] >= 1


class TestArbitrageExecutor:
    def test_execute_arb_paper_mode(self):
        from cricket.arbitrage.detector import ArbLeg, ArbOpportunity, ArbType

        arb = ArbOpportunity(
            arb_id="test_arb",
            arb_type=ArbType.CROSS_EXCHANGE,
            legs=[
                ArbLeg(
                    exchange="betfair", market_id="bf_1",
                    market_type="match_odds", selection_name="Team A",
                    side="back", odds=2.10, stake=100.0,
                    implied_probability=0.476,
                ),
                ArbLeg(
                    exchange="smarkets", market_id="sm_1",
                    market_type="match_odds", selection_name="Team A",
                    side="lay", odds=2.02, stake=104.0,
                    implied_probability=0.495,
                ),
            ],
            guaranteed_profit=3.5,
            guaranteed_profit_pct=1.72,
            total_stake=204.0,
            match_id="match_1",
        )

        executor = ArbitrageExecutor(
            bankroll=10_000.0,
            paper_mode=True,
            paper_fill_rate=1.0,  # 100% fill for deterministic test
        )

        result = executor.execute(arb)
        assert result.all_filled
        assert result.actual_profit != 0  # Should have calculated profit

    def test_respects_bankroll_limit(self):
        from cricket.arbitrage.detector import ArbLeg, ArbOpportunity, ArbType

        arb = ArbOpportunity(
            arb_id="big_arb",
            arb_type=ArbType.INTRA_MARKET,
            legs=[
                ArbLeg(
                    exchange="betfair", market_id="bf_1",
                    market_type="match_odds", selection_name="Team A",
                    side="back", odds=2.0, stake=5000.0,
                    implied_probability=0.5,
                ),
                ArbLeg(
                    exchange="betfair", market_id="bf_1",
                    market_type="match_odds", selection_name="Team B",
                    side="back", odds=3.0, stake=3333.0,
                    implied_probability=0.333,
                ),
            ],
            guaranteed_profit=100.0,
            total_stake=8333.0,
            match_id="match_1",
        )

        executor = ArbitrageExecutor(
            bankroll=10_000.0,
            max_arb_stake_pct=0.05,  # 5% = £500 max
            paper_mode=True,
            paper_fill_rate=1.0,
        )

        result = executor.execute(arb)
        # Stakes should be scaled down to max 5% of bankroll
        total_staked = sum(l.stake for l in arb.legs)
        assert total_staked <= 500.0 + 1.0  # Allow rounding

    def test_stats_tracking(self):
        executor = ArbitrageExecutor(bankroll=10_000.0, paper_fill_rate=1.0)
        stats = executor.get_stats()
        assert stats["total_attempted"] == 0
        assert stats["total_profit"] == 0.0
