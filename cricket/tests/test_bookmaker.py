"""Tests for the Bookmaker layer — punter profiling and risk routing."""

from __future__ import annotations

import pytest
from datetime import datetime

from cricket.bookmaker.punter_profiler import (
    BetRecord,
    PunterCategory,
    PunterProfile,
    PunterProfiler,
)
from cricket.bookmaker.risk_router import (
    LiabilityBook,
    RiskAction,
    RiskRouter,
    RoutingDecision,
)
from cricket.arbitrage.detector import (
    ArbitrageScanner,
    MarketPrices,
    SelectionPrice,
)


def make_bet(
    punter_id: str = "punter_1",
    match_id: str = "match_1",
    selection: str = "Team A",
    odds: float = 2.0,
    stake: float = 100.0,
    time_before: float = 30.0,
    market_type: str = "match_odds",
) -> BetRecord:
    return BetRecord(
        bet_id=f"bet_{punter_id}_{hash((match_id, selection, odds, stake)) % 10000}",
        punter_id=punter_id,
        match_id=match_id,
        market_type=market_type,
        selection=selection,
        side="back",
        odds=odds,
        stake=stake,
        time_before_start_mins=time_before,
    )


def make_exchange_market(
    selection: str = "Team A",
    lay_odds: float = 2.02,
    back_odds: float = 1.98,
    exchange: str = "betfair",
) -> MarketPrices:
    return MarketPrices(
        exchange=exchange,
        market_id=f"{exchange}_mkt",
        market_type="match_odds",
        match_id="match_1",
        selections=[
            SelectionPrice(
                exchange=exchange,
                market_id=f"{exchange}_mkt",
                market_type="match_odds",
                selection_name=selection,
                back_odds=back_odds,
                lay_odds=lay_odds,
                back_liquidity=5000.0,
                lay_liquidity=5000.0,
            )
        ],
    )


# ─── Punter Profiler Tests ──────────────────────────────────────────────────

class TestPunterProfiler:
    def test_new_punter_is_unknown(self):
        profiler = PunterProfiler(min_bets_to_classify=20)
        profile = profiler.get_profile("new_guy")
        assert profile.category == PunterCategory.UNKNOWN
        assert profile.total_bets == 0

    def test_records_bets(self):
        profiler = PunterProfiler()
        bet = make_bet(punter_id="alice")
        profile = profiler.record_bet(bet)
        assert profile.total_bets == 1
        assert profile.total_staked == 100.0

    def test_mug_classification(self):
        """A punter who consistently loses should be classified as mug."""
        profiler = PunterProfiler(min_bets_to_classify=10)

        # Place and settle 15 bets — lose 12, win 3
        for i in range(15):
            bet = make_bet(
                punter_id="bob",
                odds=2.0,
                stake=50.0 + i * 5,     # Erratic stakes (mug signal)
                time_before=2.0,        # Late bets (mug signal)
                match_id=f"match_{i}",
            )
            profiler.record_bet(bet)
            won = i < 3  # Only win first 3
            # Closing odds higher = they got worse value than the line (mug signal)
            profiler.settle_bet(bet.bet_id, "bob", won=won, closing_odds=2.20)

        profile = profiler.get_profile("bob")
        assert profile.category == PunterCategory.MUG
        assert profile.sharpness_score < 0.4

    def test_sharp_classification(self):
        """A punter who consistently wins and beats closing line = sharp."""
        profiler = PunterProfiler(min_bets_to_classify=10)

        for i in range(20):
            bet = make_bet(
                punter_id="pro",
                odds=2.10,
                stake=100.0,             # Consistent stakes (sharp signal)
                time_before=120.0,       # Early bets (sharp signal)
                match_id=f"match_{i}",
                market_type=["match_odds", "innings_runs", "session"][i % 3],
            )
            profiler.record_bet(bet)
            won = i % 3 != 0  # Win 66% — very strong
            # Closing odds worse than what they got (they beat the line)
            profiler.settle_bet(bet.bet_id, "pro", won=won, closing_odds=1.90)

        profile = profiler.get_profile("pro")
        assert profile.category == PunterCategory.SHARP
        assert profile.sharpness_score > 0.6

    def test_insufficient_data_stays_unknown(self):
        """Not enough settled bets = stays unknown regardless of results."""
        profiler = PunterProfiler(min_bets_to_classify=20)

        for i in range(5):
            bet = make_bet(punter_id="new", match_id=f"m_{i}")
            profiler.record_bet(bet)
            profiler.settle_bet(bet.bet_id, "new", won=True)

        profile = profiler.get_profile("new")
        assert profile.category == PunterCategory.UNKNOWN

    def test_pnl_tracking(self):
        """Track our P&L against punter correctly."""
        profiler = PunterProfiler(min_bets_to_classify=5)

        # Punter bets £100 at 2.0 and loses → we make £100
        bet = make_bet(punter_id="loser", odds=2.0, stake=100.0)
        profiler.record_bet(bet)
        profiler.settle_bet(bet.bet_id, "loser", won=False)

        profile = profiler.get_profile("loser")
        assert profile.total_pnl == 100.0  # We gained £100

        # Punter bets £100 at 2.0 and wins → we lose £100
        bet2 = make_bet(punter_id="loser", odds=2.0, stake=100.0, match_id="m2")
        profiler.record_bet(bet2)
        profiler.settle_bet(bet2.bet_id, "loser", won=True)

        profile = profiler.get_profile("loser")
        assert profile.total_pnl == 0.0  # Net zero

    def test_aggregate_stats(self):
        profiler = PunterProfiler()
        for pid in ["a", "b", "c"]:
            bet = make_bet(punter_id=pid)
            profiler.record_bet(bet)

        stats = profiler.get_stats()
        assert stats["total_punters"] == 3


# ─── Risk Router Tests ──────────────────────────────────────────────────────

class TestRiskRouter:
    def _build_router_with_sharp(self):
        """Helper: create a router and pre-classify a punter as sharp."""
        profiler = PunterProfiler(min_bets_to_classify=5)
        router = RiskRouter(profiler=profiler, bankroll=50_000.0)

        # Build up sharp profile
        for i in range(10):
            bet = make_bet(
                punter_id="sharp_guy", odds=2.10, stake=100.0,
                time_before=120.0, match_id=f"m_{i}",
                market_type=["match_odds", "innings_runs"][i % 2],
            )
            profiler.record_bet(bet)
            profiler.settle_bet(bet.bet_id, "sharp_guy", won=i % 3 != 0, closing_odds=1.90)

        return router, profiler

    def _build_router_with_mug(self):
        """Helper: create a router and pre-classify a punter as mug."""
        profiler = PunterProfiler(min_bets_to_classify=5)
        router = RiskRouter(profiler=profiler, bankroll=50_000.0)

        for i in range(10):
            bet = make_bet(
                punter_id="mug_guy", odds=2.0, stake=50.0,
                time_before=2.0, match_id=f"m_{i}",
            )
            profiler.record_bet(bet)
            profiler.settle_bet(bet.bet_id, "mug_guy", won=i < 2, closing_odds=2.10)

        return router, profiler

    def test_sharp_bet_passed_through(self):
        """Sharp punter's bet should be fully hedged on exchange."""
        router, _ = self._build_router_with_sharp()

        bet = make_bet(punter_id="sharp_guy", stake=500.0, odds=2.05)
        exchange = make_exchange_market(lay_odds=2.02)

        decision = router.route_bet(bet, exchange_markets=[exchange])

        assert decision.action == RiskAction.PASS_THROUGH
        assert decision.hedge_pct >= 0.99
        assert decision.absorb_pct <= 0.01
        assert decision.hedge_stake == 500.0
        assert decision.absorb_stake == 0.0

    def test_mug_bet_absorbed(self):
        """Mug punter's bet should be absorbed."""
        router, _ = self._build_router_with_mug()

        bet = make_bet(punter_id="mug_guy", stake=200.0, odds=2.0)
        exchange = make_exchange_market(lay_odds=2.02)

        decision = router.route_bet(bet, exchange_markets=[exchange])

        assert decision.action == RiskAction.ABSORB
        assert decision.absorb_pct >= 0.99
        assert decision.absorb_stake == 200.0
        assert decision.hedge_stake == 0.0

    def test_unknown_bet_partially_absorbed(self):
        """Unknown punter should get partial absorb, partial hedge."""
        profiler = PunterProfiler(min_bets_to_classify=20)
        router = RiskRouter(profiler=profiler, unknown_absorb_pct=0.30)

        bet = make_bet(punter_id="newbie", stake=1000.0, odds=2.0)
        exchange = make_exchange_market(lay_odds=2.02)

        decision = router.route_bet(bet, exchange_markets=[exchange])

        assert decision.action == RiskAction.PARTIAL
        assert abs(decision.absorb_pct - 0.30) < 0.01
        assert abs(decision.hedge_pct - 0.70) < 0.01
        assert abs(decision.absorb_stake - 300.0) < 1.0
        assert abs(decision.hedge_stake - 700.0) < 1.0

    def test_liability_limits_force_hedge(self):
        """Even mug bets get hedged when liability limits are hit."""
        router, profiler = self._build_router_with_mug()
        # Set very low liability limit
        router._max_match_liability_pct = 0.001  # £50 on 50k bankroll

        # First small bet absorbs fine
        bet1 = make_bet(punter_id="mug_guy", stake=20.0, odds=2.0)
        d1 = router.route_bet(bet1, exchange_markets=[make_exchange_market()])
        assert d1.action == RiskAction.ABSORB

        # Second larger bet should hit limit and hedge
        bet2 = make_bet(punter_id="mug_guy", stake=500.0, odds=2.0)
        d2 = router.route_bet(bet2, exchange_markets=[make_exchange_market()])
        assert d2.action in (RiskAction.PARTIAL, RiskAction.PASS_THROUGH)
        assert d2.hedge_pct > 0

    def test_passthrough_captures_margin(self):
        """Passing through should capture the odds difference as margin."""
        router, _ = self._build_router_with_sharp()

        # Punter gets 2.00, exchange lay is 2.02
        # We offered worse odds than exchange → we make money
        bet = make_bet(punter_id="sharp_guy", stake=1000.0, odds=1.95)
        exchange = make_exchange_market(lay_odds=2.02)

        decision = router.route_bet(bet, exchange_markets=[exchange])
        assert decision.margin_captured > 0

    def test_match_settlement(self):
        """Settling a match should calculate correct P&L."""
        router, _ = self._build_router_with_mug()

        # Mug bets £200 on Team A at 2.0
        bet = make_bet(punter_id="mug_guy", stake=200.0, odds=2.0, selection="Team A")
        router.route_bet(bet, exchange_markets=[make_exchange_market()])

        # Team B wins → mug loses → we profit
        result = router.settle_match("match_1", "Team B")
        assert result["absorbed_pnl"] == 200.0  # We kept the £200 stake
        assert result["total_pnl"] > 0

    def test_match_settlement_punter_wins(self):
        """When mug wins, we pay out on absorbed portion."""
        router, _ = self._build_router_with_mug()

        bet = make_bet(punter_id="mug_guy", stake=200.0, odds=2.0, selection="Team A")
        router.route_bet(bet, exchange_markets=[make_exchange_market()])

        # Team A wins → mug wins → we pay out
        result = router.settle_match("match_1", "Team A")
        assert result["absorbed_pnl"] == -200.0  # We pay 200 * (2.0 - 1)

    def test_odds_offer_wider_for_mugs(self):
        """Mugs should get worse odds than sharps."""
        profiler = PunterProfiler(min_bets_to_classify=5)
        router = RiskRouter(
            profiler=profiler,
            mug_margin=0.06,
            sharp_margin=0.02,
        )

        # Manually set categories for the test
        mug_profile = profiler.get_profile("mug")
        mug_profile.category = PunterCategory.MUG
        sharp_profile = profiler.get_profile("sharp")
        sharp_profile.category = PunterCategory.SHARP

        mug_offer = router.make_odds("mug", "Team A", fair_odds=2.0, best_exchange_back=1.98)
        sharp_offer = router.make_odds("sharp", "Team A", fair_odds=2.0, best_exchange_back=1.98)

        # Mug gets worse odds (lower for a back bet)
        assert mug_offer.offered_odds < sharp_offer.offered_odds
        assert mug_offer.margin_pct > sharp_offer.margin_pct

    def test_stats_tracking(self):
        """Router should track aggregate stats."""
        profiler = PunterProfiler(min_bets_to_classify=50)
        router = RiskRouter(profiler=profiler)
        stats = router.get_stats()
        assert stats["total_bets_routed"] == 0
        assert stats["bankroll"] == 50_000.0

    def test_arb_scanner_integration(self):
        """Router should pass exchange markets to arb scanner."""
        profiler = PunterProfiler(min_bets_to_classify=50)
        arb_scanner = ArbitrageScanner(min_liquidity=50.0)
        router = RiskRouter(profiler=profiler, arb_scanner=arb_scanner)

        bet = make_bet(punter_id="someone", stake=100.0)
        exchange = make_exchange_market(lay_odds=2.02)

        decision = router.route_bet(bet, exchange_markets=[exchange])
        # Should have attempted arb scan (even if no arbs found)
        assert isinstance(decision.arb_opportunities, list)


class TestLiabilityBook:
    def test_tracks_liability(self):
        book = LiabilityBook()
        book.add_liability("match_1", "Team A", payout_if_wins=200.0, stake=100.0)
        book.add_liability("match_1", "Team A", payout_if_wins=150.0, stake=75.0)

        assert book.get_selection_liability("match_1", "Team A") == 350.0
        assert book.get_match_liability("match_1") == 350.0
        assert book.total_liability == 350.0
        assert book.total_stakes_collected == 175.0

    def test_separate_matches(self):
        book = LiabilityBook()
        book.add_liability("m1", "Team A", payout_if_wins=100.0, stake=50.0)
        book.add_liability("m2", "Team X", payout_if_wins=200.0, stake=100.0)

        assert book.get_match_liability("m1") == 100.0
        assert book.get_match_liability("m2") == 200.0
        assert book.total_liability == 300.0
