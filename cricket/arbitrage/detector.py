"""
Arbitrage Detection Engine.

Identifies guaranteed-profit opportunities across four categories:

1. INTRA-MARKET: Back all outcomes in a single market where the combined
   implied probabilities sum to less than 100% (negative overround).
   On Betfair this happens briefly during fast price movements.

2. CROSS-MARKET: Exploit inconsistencies between correlated markets on the
   same exchange. E.g., Match Odds implies Team A wins 60% of the time,
   but Innings Runs Over/Under implies their expected score is only
   consistent with a 50% win rate. The markets disagree → arb exists.

3. CROSS-EXCHANGE: Same outcome priced differently across exchanges.
   Back on exchange A at higher odds, lay on exchange B at lower odds.
   Betfair vs Smarkets vs Betdaq vs traditional bookmakers.

4. TEMPORAL: After a wicket or significant event, some markets/exchanges
   update faster than others. Trade the lagging market before it corrects.
   This is latency arbitrage — the fastest feed wins.

Key insight: Pure arbitrage (guaranteed profit regardless of outcome) is
rare and fleeting on liquid markets. The more realistic opportunity is
*statistical arbitrage* — situations where the expected value is strongly
positive based on model confidence, even if not risk-free.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ArbType(Enum):
    INTRA_MARKET = "intra_market"       # Back-all / lay-all within one market
    CROSS_MARKET = "cross_market"       # Between correlated markets (Match Odds vs Runs)
    CROSS_EXCHANGE = "cross_exchange"   # Same outcome, different exchanges
    TEMPORAL = "temporal"               # Stale market after event


class ArbStatus(Enum):
    DETECTED = "detected"
    EXECUTING = "executing"
    FILLED = "filled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class ArbLeg:
    """One leg of an arbitrage trade."""

    exchange: str           # "betfair", "smarkets", "betdaq", "bookmaker"
    market_id: str
    market_type: str        # "match_odds", "innings_runs", etc.
    selection_name: str
    side: str               # "back" or "lay"
    odds: float
    stake: float            # GBP
    implied_probability: float

    # Execution state
    filled: bool = False
    fill_price: Optional[float] = None
    fill_stake: Optional[float] = None


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""

    arb_id: str
    arb_type: ArbType
    status: ArbStatus = ArbStatus.DETECTED

    legs: list[ArbLeg] = field(default_factory=list)

    # Profit metrics
    guaranteed_profit: float = 0.0          # GBP if all legs fill
    guaranteed_profit_pct: float = 0.0      # As % of total stake
    total_stake: float = 0.0
    overround: float = 0.0                  # Negative = arb exists

    # Context
    match_id: str = ""
    description: str = ""
    confidence: float = 0.0                 # 0-1, how likely all legs fill
    time_to_live_ms: int = 0                # Expected window before arb closes
    detected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_pure_arb(self) -> bool:
        """True if this is a risk-free guaranteed profit."""
        return self.guaranteed_profit > 0 and self.overround < 0

    @property
    def roi(self) -> float:
        """Return on investment."""
        return self.guaranteed_profit / self.total_stake if self.total_stake > 0 else 0


# ─── Price source abstraction ───────────────────────────────────────────────

@dataclass
class SelectionPrice:
    """Price for a single selection from any source."""

    exchange: str
    market_id: str
    market_type: str
    selection_name: str
    back_odds: float            # Best available back price
    lay_odds: float             # Best available lay price
    back_liquidity: float       # GBP available at back price
    lay_liquidity: float        # GBP available at lay price
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def back_probability(self) -> float:
        return 1.0 / self.back_odds if self.back_odds > 0 else 0

    @property
    def lay_probability(self) -> float:
        return 1.0 / self.lay_odds if self.lay_odds > 0 else 0

    @property
    def mid_probability(self) -> float:
        return (self.back_probability + self.lay_probability) / 2

    @property
    def spread(self) -> float:
        return self.lay_odds - self.back_odds

    @property
    def age_ms(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds() * 1000


@dataclass
class MarketPrices:
    """All selection prices for a single market."""

    exchange: str
    market_id: str
    market_type: str
    match_id: str
    selections: list[SelectionPrice] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def back_overround(self) -> float:
        """Sum of back implied probabilities. <1.0 = arb on backing all."""
        return sum(s.back_probability for s in self.selections)

    @property
    def lay_overround(self) -> float:
        """Sum of lay implied probabilities. >1.0 = arb on laying all."""
        return sum(s.lay_probability for s in self.selections)

    def get_selection(self, name: str) -> Optional[SelectionPrice]:
        for s in self.selections:
            if s.selection_name == name:
                return s
        return None


# ─── Intra-market arbitrage ─────────────────────────────────────────────────

class IntraMarketDetector:
    """Detects arbitrage within a single market.

    Type 1 - Back All: If you can back every outcome and the total implied
    probability is less than 100%, you profit regardless of result.
    This means: 1/odds_A + 1/odds_B + ... < 1.0

    Type 2 - Lay All: If you can lay every outcome and the total implied
    probability exceeds 100%, you profit.
    This means: 1/odds_A + 1/odds_B + ... > 1.0 (using lay prices)

    On Betfair, back-all arbs are extremely rare because the market is
    efficient. They appear for milliseconds during rapid price changes.
    The practical opportunity is *near-arbs* where the overround is very
    small and you have an edge from your model on which side to take.
    """

    def __init__(self, min_profit_pct: float = 0.001, min_liquidity: float = 50.0):
        self._min_profit_pct = min_profit_pct
        self._min_liquidity = min_liquidity

    def detect(self, market: MarketPrices) -> list[ArbOpportunity]:
        """Check a market for intra-market arbitrage."""
        arbs = []

        if len(market.selections) < 2:
            return arbs

        # Check back-all arb
        back_arb = self._check_back_all(market)
        if back_arb:
            arbs.append(back_arb)

        # Check lay-all arb
        lay_arb = self._check_lay_all(market)
        if lay_arb:
            arbs.append(lay_arb)

        return arbs

    def _check_back_all(self, market: MarketPrices) -> Optional[ArbOpportunity]:
        """Check if backing all outcomes produces guaranteed profit."""
        overround = market.back_overround

        if overround >= 1.0:
            return None  # No arb — probabilities sum to >= 100%

        # Calculate optimal stakes for equal profit on all outcomes
        # For back-all: stake_i = total_payout / odds_i
        # where total_payout is chosen so total_stake < total_payout
        total_payout = 100.0  # Normalize to £100 payout
        legs = []
        total_stake = 0.0

        for sel in market.selections:
            if sel.back_odds <= 1.0:
                return None  # Invalid price
            stake = total_payout / sel.back_odds
            total_stake += stake

            # Check liquidity
            if sel.back_liquidity < self._min_liquidity:
                return None

            legs.append(ArbLeg(
                exchange=market.exchange,
                market_id=market.market_id,
                market_type=market.market_type,
                selection_name=sel.selection_name,
                side="back",
                odds=sel.back_odds,
                stake=round(stake, 2),
                implied_probability=sel.back_probability,
            ))

        profit = total_payout - total_stake
        profit_pct = profit / total_stake

        if profit_pct < self._min_profit_pct:
            return None

        return ArbOpportunity(
            arb_id=f"IA-BACK-{uuid.uuid4().hex[:8]}",
            arb_type=ArbType.INTRA_MARKET,
            legs=legs,
            guaranteed_profit=round(profit, 2),
            guaranteed_profit_pct=round(profit_pct * 100, 3),
            total_stake=round(total_stake, 2),
            overround=round(overround - 1.0, 6),
            match_id=market.match_id,
            description=f"Back-all arb: overround {overround:.4f} ({(1-overround)*100:.2f}% margin)",
            confidence=0.7,  # Might not get filled on all legs
            time_to_live_ms=500,  # These disappear fast
        )

    def _check_lay_all(self, market: MarketPrices) -> Optional[ArbOpportunity]:
        """Check if laying all outcomes produces guaranteed profit."""
        lay_overround = market.lay_overround

        if lay_overround <= 1.0:
            return None  # No arb

        # For lay-all: we collect stakes and pay out to the winner
        # Profit = total_collected - max_payout
        total_collected = 100.0  # Normalize
        legs = []
        max_liability = 0.0

        for sel in market.selections:
            if sel.lay_odds <= 1.0:
                return None
            # Lay stake such that we collect proportional amounts
            stake = total_collected * sel.lay_probability
            liability = stake * (sel.lay_odds - 1.0)
            max_liability = max(max_liability, liability)

            if sel.lay_liquidity < self._min_liquidity:
                return None

            legs.append(ArbLeg(
                exchange=market.exchange,
                market_id=market.market_id,
                market_type=market.market_type,
                selection_name=sel.selection_name,
                side="lay",
                odds=sel.lay_odds,
                stake=round(stake, 2),
                implied_probability=sel.lay_probability,
            ))

        total_stake = sum(leg.stake for leg in legs)
        # Worst case: one selection wins, we pay liability
        # Best case: we keep all stakes minus liability of winner
        profit = total_stake - max_liability
        if total_stake <= 0:
            return None
        profit_pct = profit / total_stake

        if profit_pct < self._min_profit_pct:
            return None

        return ArbOpportunity(
            arb_id=f"IA-LAY-{uuid.uuid4().hex[:8]}",
            arb_type=ArbType.INTRA_MARKET,
            legs=legs,
            guaranteed_profit=round(profit, 2),
            guaranteed_profit_pct=round(profit_pct * 100, 3),
            total_stake=round(total_stake, 2),
            overround=round(lay_overround - 1.0, 6),
            match_id=market.match_id,
            description=f"Lay-all arb: lay overround {lay_overround:.4f}",
            confidence=0.6,
            time_to_live_ms=500,
        )


# ─── Cross-market arbitrage ────────────────────────────────────────────────

class CrossMarketDetector:
    """Detects inconsistencies between correlated markets.

    The key relationship: Match Odds ↔ Innings Runs.

    If Match Odds implies Team A has a 60% win probability, this implies
    a certain expected score distribution. If the Innings Runs Over/Under
    market is priced inconsistently with that distribution, an arb exists.

    Example:
    - Match Odds: Team A 1.65 (implied 60.6% win prob)
    - 1st Innings Runs O/U 165.5: Over 1.85 / Under 2.05
    - But a 60% win prob team at this venue typically scores 170+
    - The Over on innings runs is underpriced relative to Match Odds
    → Back Over 165.5 AND lay Team A slightly = correlated arb

    This is statistical arbitrage, not pure arb. But the edge is
    structural because the markets are priced by different pools
    of participants with different information.
    """

    # Expected score ranges by win probability bands (T20)
    # Derived from: if a team has X% win prob, their expected 1st innings
    # score tends to fall in this range at an average venue (160 avg)
    WIN_PROB_TO_EXPECTED_SCORE_T20 = {
        # (win_prob_low, win_prob_high) -> (expected_score_low, expected_score_high)
        (0.70, 1.00): (170, 200),
        (0.55, 0.70): (158, 175),
        (0.45, 0.55): (150, 165),
        (0.30, 0.45): (135, 155),
        (0.00, 0.30): (110, 140),
    }

    def __init__(
        self,
        min_inconsistency: float = 0.08,
        venue_avg_score: float = 160.0,
    ):
        self._min_inconsistency = min_inconsistency
        self._venue_avg = venue_avg_score

    def detect(
        self,
        match_odds: MarketPrices,
        innings_runs: Optional[MarketPrices] = None,
        session_market: Optional[MarketPrices] = None,
    ) -> list[ArbOpportunity]:
        """Detect cross-market inconsistencies.

        Args:
            match_odds: Match Odds market prices
            innings_runs: Innings Runs Over/Under market (if available)
            session_market: Session market (if available)

        Returns:
            List of arbitrage opportunities
        """
        arbs = []

        if innings_runs:
            arb = self._check_match_odds_vs_innings_runs(match_odds, innings_runs)
            if arb:
                arbs.append(arb)

        return arbs

    def _check_match_odds_vs_innings_runs(
        self,
        match_odds: MarketPrices,
        innings_runs: MarketPrices,
    ) -> Optional[ArbOpportunity]:
        """Check Match Odds vs Innings Runs consistency."""

        if len(match_odds.selections) < 2:
            return None

        # Get the favourite's implied win probability from Match Odds
        favourite = max(match_odds.selections, key=lambda s: s.mid_probability)
        win_prob = favourite.mid_probability

        # What does this win probability imply about expected score?
        expected_range = None
        for (low, high), score_range in self.WIN_PROB_TO_EXPECTED_SCORE_T20.items():
            if low <= win_prob < high:
                expected_range = score_range
                break

        if expected_range is None:
            return None

        expected_mid = (expected_range[0] + expected_range[1]) / 2

        # Get the innings runs line and prices
        over_sel = innings_runs.get_selection("Over")
        under_sel = innings_runs.get_selection("Under")

        if not over_sel or not under_sel:
            return None

        # The Over/Under line is embedded in the market name typically
        # For now, extract from the mid-probability
        # If Over and Under are both ~50%, the line is roughly fair
        # If Over is cheap (high odds), market expects lower score
        over_prob = over_sel.mid_probability
        under_prob = under_sel.mid_probability

        # Market-implied expected score direction
        # If over_prob > 0.5, market expects score to go over the line
        market_expects_high = over_prob > 0.5

        # Match Odds implies the favourite's score expectation
        match_odds_expects_high = expected_mid > self._venue_avg

        # Inconsistency: Match Odds says team should score high,
        # but Innings Runs market is pricing Over cheaply
        inconsistency = 0.0

        if match_odds_expects_high and not market_expects_high:
            # Match Odds bullish on runs, Innings Runs bearish
            inconsistency = win_prob - over_prob
        elif not match_odds_expects_high and market_expects_high:
            # Match Odds bearish on runs, Innings Runs bullish
            inconsistency = over_prob - win_prob

        if abs(inconsistency) < self._min_inconsistency:
            return None

        # Build the arb legs
        legs = []
        if match_odds_expects_high and not market_expects_high:
            # Back Over (underpriced) + Lay favourite slightly
            legs.append(ArbLeg(
                exchange=innings_runs.exchange,
                market_id=innings_runs.market_id,
                market_type="innings_runs",
                selection_name="Over",
                side="back",
                odds=over_sel.back_odds,
                stake=50.0,
                implied_probability=over_sel.back_probability,
            ))
            legs.append(ArbLeg(
                exchange=match_odds.exchange,
                market_id=match_odds.market_id,
                market_type="match_odds",
                selection_name=favourite.selection_name,
                side="lay",
                odds=favourite.lay_odds,
                stake=25.0,  # Partial hedge
                implied_probability=favourite.lay_probability,
            ))
        else:
            # Back Under + Back underdog
            legs.append(ArbLeg(
                exchange=innings_runs.exchange,
                market_id=innings_runs.market_id,
                market_type="innings_runs",
                selection_name="Under",
                side="back",
                odds=under_sel.back_odds,
                stake=50.0,
                implied_probability=under_sel.back_probability,
            ))

        return ArbOpportunity(
            arb_id=f"XM-{uuid.uuid4().hex[:8]}",
            arb_type=ArbType.CROSS_MARKET,
            legs=legs,
            guaranteed_profit=0.0,  # Statistical arb, not guaranteed
            guaranteed_profit_pct=round(abs(inconsistency) * 100, 2),
            total_stake=sum(l.stake for l in legs),
            overround=round(-abs(inconsistency), 4),
            match_id=match_odds.match_id,
            description=(
                f"Match Odds vs Innings Runs inconsistency: "
                f"{favourite.selection_name} win prob {win_prob:.1%} implies "
                f"expected score ~{expected_mid:.0f}, but runs market disagrees "
                f"(inconsistency: {abs(inconsistency):.1%})"
            ),
            confidence=0.5,  # Statistical, not pure arb
            time_to_live_ms=30_000,  # These persist longer
        )


# ─── Cross-exchange arbitrage ──────────────────────────────────────────────

class CrossExchangeDetector:
    """Detects the same outcome priced differently across exchanges.

    The classic arb: Back on exchange A at 2.10, lay on exchange B at 2.05.
    Your back wins if the outcome happens (profit = stake * (2.10-1) = 1.10x).
    Your lay wins if it doesn't (profit = lay_stake).
    Sized correctly, you profit regardless.

    Sources of cross-exchange arbs in cricket:
    - Betfair vs Smarkets (different liquidity pools, different commission)
    - Betfair vs Betdaq
    - Exchange vs traditional bookmaker (bookies are slow to adjust in-play)
    - Exchange vs Polymarket/prediction markets (different participant base)

    Commission matters: Betfair charges 2-5% on net winnings, Smarkets 2%.
    The arb must exceed the commission drag to be profitable.
    """

    def __init__(
        self,
        commission_rates: Optional[dict[str, float]] = None,
        min_profit_pct: float = 0.005,
        min_liquidity: float = 100.0,
    ):
        self._commissions = commission_rates or {
            "betfair": 0.05,
            "smarkets": 0.02,
            "betdaq": 0.02,
            "bookmaker": 0.0,  # No commission, but margin is in the odds
        }
        self._min_profit_pct = min_profit_pct
        self._min_liquidity = min_liquidity

    def detect(
        self,
        markets: list[MarketPrices],
    ) -> list[ArbOpportunity]:
        """Detect cross-exchange arbs across multiple price sources.

        Args:
            markets: List of MarketPrices from different exchanges
                     for the same match/market type

        Returns:
            List of arbitrage opportunities
        """
        if len(markets) < 2:
            return []

        arbs = []

        # Get all unique selection names
        all_selections: set[str] = set()
        for market in markets:
            for sel in market.selections:
                all_selections.add(sel.selection_name)

        # For each selection, find best back across all exchanges
        # and best lay across all exchanges
        for selection_name in all_selections:
            best_back: Optional[tuple[SelectionPrice, MarketPrices]] = None
            best_lay: Optional[tuple[SelectionPrice, MarketPrices]] = None

            for market in markets:
                sel = market.get_selection(selection_name)
                if not sel:
                    continue

                if sel.back_liquidity >= self._min_liquidity:
                    if best_back is None or sel.back_odds > best_back[0].back_odds:
                        best_back = (sel, market)

                if sel.lay_liquidity >= self._min_liquidity:
                    if best_lay is None or sel.lay_odds < best_lay[0].lay_odds:
                        best_lay = (sel, market)

            if not best_back or not best_lay:
                continue

            # Only an arb if back and lay are on different exchanges
            if best_back[1].exchange == best_lay[1].exchange:
                continue

            # Check if back odds > lay odds (the basic arb condition)
            if best_back[0].back_odds <= best_lay[0].lay_odds:
                continue

            # Calculate profit after commission
            arb = self._calculate_cross_exchange_arb(
                selection_name,
                best_back[0], best_back[1],
                best_lay[0], best_lay[1],
            )
            if arb:
                arbs.append(arb)

        return arbs

    def _calculate_cross_exchange_arb(
        self,
        selection_name: str,
        back_sel: SelectionPrice,
        back_market: MarketPrices,
        lay_sel: SelectionPrice,
        lay_market: MarketPrices,
    ) -> Optional[ArbOpportunity]:
        """Calculate profit for a back/lay cross-exchange arb."""

        back_odds = back_sel.back_odds
        lay_odds = lay_sel.lay_odds
        back_comm = self._commissions.get(back_market.exchange, 0.05)
        lay_comm = self._commissions.get(lay_market.exchange, 0.05)

        # Size for equal profit: solve for stakes where profit is same
        # regardless of outcome.
        #
        # If outcome happens:
        #   back_profit = back_stake * (back_odds - 1) * (1 - back_comm)
        #   lay_loss = lay_stake * (lay_odds - 1)
        #   net = back_profit - lay_loss
        #
        # If outcome doesn't happen:
        #   back_loss = back_stake
        #   lay_profit = lay_stake * (1 - lay_comm)
        #   net = lay_profit - back_loss
        #
        # Set equal: back_stake * (back_odds-1) * (1-bc) - lay_stake * (lay_odds-1)
        #          = lay_stake * (1-lc) - back_stake
        #
        # Solve for lay_stake/back_stake ratio:
        # lay_stake = back_stake * (1 + (back_odds-1)*(1-bc)) / (1-lc + lay_odds-1)

        back_stake = 100.0  # Normalize
        denominator = (1 - lay_comm) + (lay_odds - 1)
        if denominator <= 0:
            return None

        lay_stake = back_stake * (1 + (back_odds - 1) * (1 - back_comm)) / denominator

        # Calculate profit for each scenario
        if_wins = back_stake * (back_odds - 1) * (1 - back_comm) - lay_stake * (lay_odds - 1)
        if_loses = lay_stake * (1 - lay_comm) - back_stake

        # Guaranteed profit is the minimum of the two scenarios
        guaranteed_profit = min(if_wins, if_loses)
        total_stake = back_stake + lay_stake

        if guaranteed_profit <= 0:
            return None

        profit_pct = guaranteed_profit / total_stake
        if profit_pct < self._min_profit_pct:
            return None

        legs = [
            ArbLeg(
                exchange=back_market.exchange,
                market_id=back_market.market_id,
                market_type=back_market.market_type,
                selection_name=selection_name,
                side="back",
                odds=back_odds,
                stake=round(back_stake, 2),
                implied_probability=back_sel.back_probability,
            ),
            ArbLeg(
                exchange=lay_market.exchange,
                market_id=lay_market.market_id,
                market_type=lay_market.market_type,
                selection_name=selection_name,
                side="lay",
                odds=lay_odds,
                stake=round(lay_stake, 2),
                implied_probability=lay_sel.lay_probability,
            ),
        ]

        return ArbOpportunity(
            arb_id=f"XE-{uuid.uuid4().hex[:8]}",
            arb_type=ArbType.CROSS_EXCHANGE,
            legs=legs,
            guaranteed_profit=round(guaranteed_profit, 2),
            guaranteed_profit_pct=round(profit_pct * 100, 3),
            total_stake=round(total_stake, 2),
            overround=round(-(back_sel.back_probability - lay_sel.lay_probability), 4),
            match_id=back_market.match_id,
            description=(
                f"Cross-exchange: Back {selection_name} @ {back_odds} on "
                f"{back_market.exchange}, Lay @ {lay_odds} on {lay_market.exchange} "
                f"(after {back_comm:.0%}/{lay_comm:.0%} commission)"
            ),
            confidence=0.8,  # High if liquidity is there
            time_to_live_ms=2_000,
        )


# ─── Temporal arbitrage ────────────────────────────────────────────────────

class TemporalArbDetector:
    """Detects stale markets that haven't reacted to recent events.

    After a wicket, boundary cluster, or other significant event, different
    markets and exchanges update at different speeds. The fastest-updating
    market reveals the new fair value; any market still showing old prices
    is an arb opportunity.

    This is the closest thing to guaranteed money in exchange trading,
    but it requires:
    1. Faster data feed than competitors (SportRadar < 1s latency)
    2. Sub-second execution capability
    3. Monitoring multiple markets simultaneously

    The approach: maintain a "fair value" from our pricing model that
    updates on every ball. Any market priced more than N ticks from
    fair value is potentially stale.
    """

    def __init__(
        self,
        stale_threshold_ms: int = 3000,
        min_edge_probability: float = 0.05,
        min_liquidity: float = 100.0,
    ):
        self._stale_threshold_ms = stale_threshold_ms
        self._min_edge = min_edge_probability
        self._min_liquidity = min_liquidity
        self._last_event_time: dict[str, datetime] = {}
        self._pre_event_prices: dict[str, dict[str, float]] = {}

    def record_event(
        self,
        match_id: str,
        event_type: str,
        market_prices: dict[str, float],
    ) -> None:
        """Record a significant match event and current prices.

        Call this when a wicket falls, boundary is hit, etc.
        """
        self._last_event_time[match_id] = datetime.utcnow()
        self._pre_event_prices[match_id] = dict(market_prices)

    def detect(
        self,
        match_id: str,
        model_fair_prob: dict[str, float],
        markets: list[MarketPrices],
    ) -> list[ArbOpportunity]:
        """Check if any market is stale relative to model fair value.

        Args:
            match_id: Match identifier
            model_fair_prob: Model's current fair probability per selection
            markets: Current prices from all monitored markets/exchanges

        Returns:
            List of temporal arb opportunities
        """
        arbs = []
        last_event = self._last_event_time.get(match_id)

        for market in markets:
            for sel in market.selections:
                fair_prob = model_fair_prob.get(sel.selection_name)
                if fair_prob is None:
                    continue

                market_prob = sel.mid_probability
                edge = fair_prob - market_prob

                if abs(edge) < self._min_edge:
                    continue

                # Check if this market is potentially stale
                is_stale = False
                staleness_reason = ""

                # Price hasn't moved since last event
                pre_prices = self._pre_event_prices.get(match_id, {})
                pre_price = pre_prices.get(sel.selection_name)
                if pre_price and last_event:
                    time_since_event = (datetime.utcnow() - last_event).total_seconds() * 1000
                    price_unchanged = abs(market_prob - pre_price) < 0.01
                    if price_unchanged and time_since_event < self._stale_threshold_ms:
                        is_stale = True
                        staleness_reason = (
                            f"Price unchanged {time_since_event:.0f}ms after event "
                            f"(pre: {pre_price:.3f}, now: {market_prob:.3f})"
                        )

                # Market timestamp is old
                if sel.age_ms > self._stale_threshold_ms:
                    is_stale = True
                    staleness_reason = f"Price data {sel.age_ms:.0f}ms old"

                if not is_stale:
                    continue

                # Build arb opportunity
                if edge > 0:
                    # Model says higher prob than market → back
                    side = "back"
                    odds = sel.back_odds
                    liquidity = sel.back_liquidity
                else:
                    # Model says lower prob than market → lay
                    side = "lay"
                    odds = sel.lay_odds
                    liquidity = sel.lay_liquidity

                if liquidity < self._min_liquidity:
                    continue

                arbs.append(ArbOpportunity(
                    arb_id=f"TA-{uuid.uuid4().hex[:8]}",
                    arb_type=ArbType.TEMPORAL,
                    legs=[ArbLeg(
                        exchange=market.exchange,
                        market_id=market.market_id,
                        market_type=market.market_type,
                        selection_name=sel.selection_name,
                        side=side,
                        odds=odds,
                        stake=0.0,  # Sized by execution engine
                        implied_probability=sel.mid_probability,
                    )],
                    guaranteed_profit=0.0,  # Not guaranteed — needs mean reversion
                    guaranteed_profit_pct=round(abs(edge) * 100, 2),
                    total_stake=0.0,
                    overround=round(-abs(edge), 4),
                    match_id=match_id,
                    description=(
                        f"Stale market on {market.exchange}: "
                        f"{sel.selection_name} {side} @ {odds:.2f}, "
                        f"model fair prob {fair_prob:.1%} vs market {market_prob:.1%}. "
                        f"{staleness_reason}"
                    ),
                    confidence=0.6,
                    time_to_live_ms=1_000,
                ))

        return arbs


# ─── Unified arbitrage scanner ─────────────────────────────────────────────

class ArbitrageScanner:
    """Unified scanner that runs all arbitrage detectors.

    This is the main entry point. Feed it market data from all sources
    and it will identify every type of arbitrage opportunity.
    """

    def __init__(
        self,
        commission_rates: Optional[dict[str, float]] = None,
        min_profit_pct: float = 0.001,
        min_liquidity: float = 50.0,
    ):
        self.intra_market = IntraMarketDetector(
            min_profit_pct=min_profit_pct,
            min_liquidity=min_liquidity,
        )
        self.cross_market = CrossMarketDetector()
        self.cross_exchange = CrossExchangeDetector(
            commission_rates=commission_rates,
            min_profit_pct=min_profit_pct,
            min_liquidity=min_liquidity,
        )
        self.temporal = TemporalArbDetector(min_liquidity=min_liquidity)

        self._arb_log: list[ArbOpportunity] = []

    def scan(
        self,
        match_id: str,
        match_odds_markets: list[MarketPrices],
        innings_runs_market: Optional[MarketPrices] = None,
        model_fair_prob: Optional[dict[str, float]] = None,
    ) -> list[ArbOpportunity]:
        """Run all arbitrage detectors and return opportunities.

        Args:
            match_id: Match identifier
            match_odds_markets: Match Odds from all exchanges
            innings_runs_market: Innings Runs O/U if available
            model_fair_prob: Model's fair probabilities (for temporal arbs)

        Returns:
            List of all detected arb opportunities, sorted by profit
        """
        all_arbs: list[ArbOpportunity] = []

        # 1. Intra-market on each exchange
        for market in match_odds_markets:
            arbs = self.intra_market.detect(market)
            all_arbs.extend(arbs)

        # 2. Cross-market (Match Odds vs Innings Runs)
        if innings_runs_market and match_odds_markets:
            primary = match_odds_markets[0]
            arbs = self.cross_market.detect(primary, innings_runs_market)
            all_arbs.extend(arbs)

        # 3. Cross-exchange
        if len(match_odds_markets) >= 2:
            arbs = self.cross_exchange.detect(match_odds_markets)
            all_arbs.extend(arbs)

        # 4. Temporal
        if model_fair_prob:
            arbs = self.temporal.detect(match_id, model_fair_prob, match_odds_markets)
            all_arbs.extend(arbs)

        # Sort by expected profit
        all_arbs.sort(key=lambda a: a.guaranteed_profit_pct, reverse=True)

        self._arb_log.extend(all_arbs)

        if all_arbs:
            logger.info(
                "Scan found %d arb opportunities for match %s",
                len(all_arbs), match_id,
            )
            for arb in all_arbs:
                logger.info(
                    "  [%s] %s | profit: %.2f%% | %s",
                    arb.arb_type.value, arb.arb_id,
                    arb.guaranteed_profit_pct, arb.description,
                )

        return all_arbs

    def record_match_event(
        self,
        match_id: str,
        event_type: str,
        market_prices: dict[str, float],
    ) -> None:
        """Record a match event for temporal arb detection."""
        self.temporal.record_event(match_id, event_type, market_prices)

    def get_arb_history(self) -> list[ArbOpportunity]:
        return list(self._arb_log)

    def get_stats(self) -> dict:
        """Get aggregate arb detection statistics."""
        by_type: dict[str, int] = {}
        total_profit = 0.0
        for arb in self._arb_log:
            by_type[arb.arb_type.value] = by_type.get(arb.arb_type.value, 0) + 1
            total_profit += arb.guaranteed_profit

        return {
            "total_detected": len(self._arb_log),
            "by_type": by_type,
            "total_theoretical_profit": round(total_profit, 2),
            "pure_arbs": sum(1 for a in self._arb_log if a.is_pure_arb),
            "statistical_arbs": sum(1 for a in self._arb_log if not a.is_pure_arb),
        }
