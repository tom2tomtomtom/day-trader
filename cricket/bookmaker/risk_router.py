"""
Risk Routing Engine.

The decision engine at the heart of the bookmaker model.
For every incoming bet, decides: absorb, pass-through, or partial hedge.

Decision tree:

    Incoming bet
        │
        ├─ Punter is SHARP (high win rate, beats closing line)
        │   → PASS THROUGH to Betfair/best exchange price
        │   → We take a small margin (offer 2.05, lay on Betfair at 2.00)
        │   → Zero risk, guaranteed small profit
        │
        ├─ Punter is MUG (low win rate, poor timing, erratic stakes)
        │   → ABSORB the bet, we take the other side
        │   → Expected value is strongly in our favour
        │   → Monitor liability and partially hedge if it gets too large
        │
        ├─ Punter is UNKNOWN (new customer, insufficient data)
        │   → PARTIAL: absorb a small amount, hedge the rest
        │   → Gather data while limiting downside
        │
        └─ Regardless of punter:
            → Check if arb scanner has a better price on another exchange
            → If passing through, use the best available cross-exchange price
            → If absorbing, check if we can cheaply hedge via cross-market arb

The margin we build into odds depends on punter type:
- Mug: wider margin (they don't notice/care)
- Unknown: standard margin
- Sharp: thinnest margin (they'll go elsewhere if we're too wide)
  But for sharps we also consider restricting stakes or limiting access.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from cricket.bookmaker.punter_profiler import (
    BetRecord,
    PunterCategory,
    PunterProfile,
    PunterProfiler,
    RiskAction,
)
from cricket.arbitrage.detector import (
    ArbitrageScanner,
    ArbOpportunity,
    MarketPrices,
    SelectionPrice,
)

logger = logging.getLogger(__name__)


@dataclass
class OddsOffer:
    """The odds we offer to a punter for a specific selection."""

    selection: str
    offered_odds: float         # What the punter sees
    fair_odds: float            # Our model's fair price
    best_exchange_odds: float   # Best available on exchanges
    margin_pct: float           # Our built-in margin
    max_stake: float            # Maximum we'll accept


@dataclass
class RoutingDecision:
    """The routing decision for a single bet."""

    bet: BetRecord
    punter: PunterProfile
    action: RiskAction

    # How much we absorb vs hedge
    absorb_pct: float = 0.0        # 0-1, portion we keep
    hedge_pct: float = 0.0         # 0-1, portion we lay off
    absorb_stake: float = 0.0
    hedge_stake: float = 0.0

    # Hedge details
    hedge_exchange: str = ""       # Where to hedge
    hedge_odds: float = 0.0        # Price we'd get on exchange
    hedge_cost: float = 0.0        # Cost of hedging (commission etc.)

    # Arb opportunities found for this bet
    arb_opportunities: list[ArbOpportunity] = field(default_factory=list)

    # Expected P&L
    expected_pnl: float = 0.0      # Our expected profit on this bet
    worst_case_pnl: float = 0.0    # If the punter wins

    # Margin captured
    margin_captured: float = 0.0   # Difference between offered odds and fair/exchange

    reason: str = ""

    @property
    def is_risk_free(self) -> bool:
        """True if we've hedged everything — guaranteed profit."""
        return self.hedge_pct >= 0.99 and self.margin_captured > 0


@dataclass
class LiabilityBook:
    """Tracks our total exposure across all absorbed bets."""

    # Per-selection liability: if this selection wins, how much do we pay out?
    selection_liability: dict[str, float] = field(default_factory=dict)
    # Per-match total liability
    match_liability: dict[str, float] = field(default_factory=dict)
    # Total across all matches
    total_liability: float = 0.0
    # Total stakes collected (from absorbed bets where punter loses)
    total_stakes_collected: float = 0.0

    def add_liability(
        self, match_id: str, selection: str, payout_if_wins: float, stake: float,
    ) -> None:
        key = f"{match_id}:{selection}"
        self.selection_liability[key] = self.selection_liability.get(key, 0) + payout_if_wins
        self.match_liability[match_id] = self.match_liability.get(match_id, 0) + payout_if_wins
        self.total_liability += payout_if_wins
        self.total_stakes_collected += stake

    def get_match_liability(self, match_id: str) -> float:
        return self.match_liability.get(match_id, 0.0)

    def get_selection_liability(self, match_id: str, selection: str) -> float:
        return self.selection_liability.get(f"{match_id}:{selection}", 0.0)


class RiskRouter:
    """Routes incoming bets based on punter classification.

    The profit model:

    1. MUG bets (absorb): We keep the stake when they lose (~55% of the time).
       On a £10 bet at 2.0, we make £10 when they lose, pay £10 when they win.
       With 55% mug loss rate: EV = 0.55 * 10 - 0.45 * 10 = +£1.00 per bet.

    2. SHARP bets (pass-through): We offer 2.05, lay on Betfair at 2.00.
       On a £100 bet: if punter wins, we pay £105 but collect £100 from Betfair = -£5.
       But we also collect the £100 stake and pay £100 to Betfair. Net: we keep the
       margin regardless. Actual profit = stake * (offered_odds - exchange_odds) / offered_odds.
       On £100 at 2.05 vs 2.00: profit ≈ £2.44 risk-free.

    3. UNKNOWN bets (partial): Absorb 30%, hedge 70%. Builds data while limiting risk.
    """

    def __init__(
        self,
        profiler: PunterProfiler,
        arb_scanner: Optional[ArbitrageScanner] = None,
        bankroll: float = 50_000.0,
        max_match_liability_pct: float = 0.10,     # Max 10% of bankroll per match
        max_selection_liability_pct: float = 0.05,  # Max 5% per selection
        mug_margin: float = 0.06,       # 6% margin for mugs
        unknown_margin: float = 0.04,   # 4% for unknowns
        sharp_margin: float = 0.02,     # 2% for sharps (thin, but hedged)
        unknown_absorb_pct: float = 0.30,  # Absorb 30% of unknown bets
        mug_max_stake: float = 10_000.0,
        sharp_max_stake: float = 500.0,    # Limit sharp stakes
        unknown_max_stake: float = 2_000.0,
    ):
        self._profiler = profiler
        self._arb_scanner = arb_scanner
        self._bankroll = bankroll
        self._max_match_liability_pct = max_match_liability_pct
        self._max_selection_liability_pct = max_selection_liability_pct

        self._margins = {
            PunterCategory.MUG: mug_margin,
            PunterCategory.UNKNOWN: unknown_margin,
            PunterCategory.SHARP: sharp_margin,
        }
        self._max_stakes = {
            PunterCategory.MUG: mug_max_stake,
            PunterCategory.UNKNOWN: unknown_max_stake,
            PunterCategory.SHARP: sharp_max_stake,
        }
        self._unknown_absorb_pct = unknown_absorb_pct

        self._liability = LiabilityBook()
        self._decisions: list[RoutingDecision] = []
        self._total_absorbed_pnl: float = 0.0
        self._total_passthrough_pnl: float = 0.0

    def make_odds(
        self,
        punter_id: str,
        selection: str,
        fair_odds: float,
        best_exchange_back: float,
    ) -> OddsOffer:
        """Generate the odds we offer to a specific punter.

        We always offer worse odds than fair value — the margin depends
        on who the punter is.
        """
        profile = self._profiler.get_profile(punter_id)
        margin = self._margins.get(profile.category, 0.04)
        max_stake = self._max_stakes.get(profile.category, 2000.0)

        # Offered odds = fair odds minus our margin
        # For a back bet: lower odds = worse for punter
        offered_odds = fair_odds * (1 - margin)
        offered_odds = max(1.01, round(offered_odds, 2))

        return OddsOffer(
            selection=selection,
            offered_odds=offered_odds,
            fair_odds=fair_odds,
            best_exchange_odds=best_exchange_back,
            margin_pct=margin * 100,
            max_stake=max_stake,
        )

    def route_bet(
        self,
        bet: BetRecord,
        exchange_markets: Optional[list[MarketPrices]] = None,
    ) -> RoutingDecision:
        """Route an incoming bet: absorb, pass-through, or partial.

        Args:
            bet: The incoming bet from the punter
            exchange_markets: Current exchange prices for hedging

        Returns:
            RoutingDecision with full details
        """
        profile = self._profiler.get_profile(bet.punter_id)

        # Record the bet in the profiler
        self._profiler.record_bet(bet)

        # Find best hedge price across exchanges
        best_hedge_odds = 0.0
        best_hedge_exchange = ""
        if exchange_markets:
            for market in exchange_markets:
                sel = market.get_selection(bet.selection)
                if sel and sel.lay_odds > 0:
                    # We need to LAY on the exchange to hedge a punter's BACK bet
                    if best_hedge_odds == 0 or sel.lay_odds < best_hedge_odds:
                        best_hedge_odds = sel.lay_odds
                        best_hedge_exchange = market.exchange

        # Check for arb opportunities that could reduce hedging cost
        arb_opps = []
        if self._arb_scanner and exchange_markets:
            arb_opps = self._arb_scanner.scan(
                bet.match_id, exchange_markets
            )

        # Make the routing decision
        decision = self._decide(bet, profile, best_hedge_odds, best_hedge_exchange)
        decision.arb_opportunities = arb_opps

        # Calculate expected P&L
        self._calculate_expected_pnl(decision, bet)

        # Update liability book for absorbed portion
        if decision.absorb_stake > 0:
            payout_if_wins = decision.absorb_stake * (bet.odds - 1)
            self._liability.add_liability(
                bet.match_id, bet.selection, payout_if_wins, decision.absorb_stake,
            )

        self._decisions.append(decision)

        logger.info(
            "Routed bet %s from %s (%s): %s %.0f%% / hedge %.0f%% | "
            "stake=£%.2f, margin=£%.2f | %s",
            bet.bet_id, bet.punter_id, profile.category.value,
            decision.action.value, decision.absorb_pct * 100,
            decision.hedge_pct * 100, bet.stake, decision.margin_captured,
            decision.reason,
        )

        return decision

    def _decide(
        self,
        bet: BetRecord,
        profile: PunterProfile,
        hedge_odds: float,
        hedge_exchange: str,
    ) -> RoutingDecision:
        """Core routing logic."""

        decision = RoutingDecision(bet=bet, punter=profile, action=RiskAction.ABSORB)
        decision.hedge_odds = hedge_odds
        decision.hedge_exchange = hedge_exchange

        # Check liability limits — look ahead at what this bet would add
        new_payout = bet.stake * (bet.odds - 1) if bet.odds > 1 else 0

        match_liability = self._liability.get_match_liability(bet.match_id)
        max_match = self._bankroll * self._max_match_liability_pct
        at_match_limit = (match_liability + new_payout) > max_match

        sel_liability = self._liability.get_selection_liability(bet.match_id, bet.selection)
        max_sel = self._bankroll * self._max_selection_liability_pct
        at_selection_limit = (sel_liability + new_payout) > max_sel

        if profile.category == PunterCategory.SHARP:
            # Sharp punter — pass through everything, take the margin
            decision.action = RiskAction.PASS_THROUGH
            decision.absorb_pct = 0.0
            decision.hedge_pct = 1.0
            decision.absorb_stake = 0.0
            decision.hedge_stake = bet.stake
            decision.margin_captured = self._calc_passthrough_margin(bet, hedge_odds)
            decision.reason = (
                f"Sharp punter (score={profile.sharpness_score:.2f}). "
                f"Full pass-through to {hedge_exchange} @ {hedge_odds:.2f}"
            )

        elif profile.category == PunterCategory.MUG:
            if at_match_limit or at_selection_limit:
                # We want to absorb but we're at our limit — hedge overflow
                absorbable = max(0, min(
                    max_match - match_liability,
                    max_sel - sel_liability,
                    bet.stake,
                ))
                if absorbable <= 0:
                    # Fully at limit — pass through
                    decision.action = RiskAction.PASS_THROUGH
                    decision.absorb_pct = 0.0
                    decision.hedge_pct = 1.0
                    decision.absorb_stake = 0.0
                    decision.hedge_stake = bet.stake
                    decision.margin_captured = self._calc_passthrough_margin(bet, hedge_odds)
                    decision.reason = (
                        f"Mug (score={profile.sharpness_score:.2f}) but at liability limit "
                        f"(match: £{match_liability:.0f}/£{max_match:.0f}). Passing through."
                    )
                else:
                    decision.action = RiskAction.PARTIAL
                    decision.absorb_pct = absorbable / bet.stake
                    decision.hedge_pct = 1.0 - decision.absorb_pct
                    decision.absorb_stake = absorbable
                    decision.hedge_stake = bet.stake - absorbable
                    decision.margin_captured = self._calc_passthrough_margin(
                        bet, hedge_odds, stake=decision.hedge_stake
                    )
                    decision.reason = (
                        f"Mug (score={profile.sharpness_score:.2f}) but near liability limit. "
                        f"Absorbing £{absorbable:.0f}, hedging £{decision.hedge_stake:.0f}."
                    )
            else:
                # Mug, within limits — absorb everything
                decision.action = RiskAction.ABSORB
                decision.absorb_pct = 1.0
                decision.hedge_pct = 0.0
                decision.absorb_stake = bet.stake
                decision.hedge_stake = 0.0
                decision.margin_captured = 0.0  # No immediate margin, profit comes from EV
                decision.reason = (
                    f"Mug punter (score={profile.sharpness_score:.2f}, "
                    f"win_rate={profile.win_rate:.2f}). Full absorb — "
                    f"expected to lose {(1 - profile.win_rate) * 100:.0f}% of bets."
                )

        elif profile.category == PunterCategory.UNKNOWN:
            # Unknown — absorb a fraction, hedge the rest
            absorb_frac = self._unknown_absorb_pct

            # If approaching limits, reduce absorb fraction
            if at_match_limit or at_selection_limit:
                absorb_frac = 0.0

            decision.action = RiskAction.PARTIAL if absorb_frac > 0 else RiskAction.PASS_THROUGH
            decision.absorb_pct = absorb_frac
            decision.hedge_pct = 1.0 - absorb_frac
            decision.absorb_stake = round(bet.stake * absorb_frac, 2)
            decision.hedge_stake = round(bet.stake * (1 - absorb_frac), 2)
            decision.margin_captured = self._calc_passthrough_margin(
                bet, hedge_odds, stake=decision.hedge_stake
            )
            decision.reason = (
                f"Unknown punter ({profile.total_settled}/{self._profiler._min_bets} bets settled). "
                f"Absorbing {absorb_frac:.0%}, hedging rest. "
                f"Gathering data for classification."
            )

        return decision

    def _calc_passthrough_margin(
        self, bet: BetRecord, hedge_odds: float, stake: Optional[float] = None,
    ) -> float:
        """Calculate our guaranteed margin on the pass-through portion.

        We offered the punter odds of X, we hedge on the exchange at Y.
        If X < Y (our offered odds are worse for the punter), we profit.
        """
        if hedge_odds <= 0:
            return 0.0

        s = stake if stake is not None else bet.stake
        if bet.odds <= 0:
            return 0.0

        # If punter wins: we pay stake * (offered - 1), collect stake * (hedge - 1)
        # If punter loses: we collect stake, pay stake to exchange
        # Net margin = stake * (hedge_odds - offered_odds) / hedge_odds
        # But only if we're offering worse odds (lower) than the exchange
        if bet.odds < hedge_odds:
            # We offered lower odds than exchange → margin exists
            margin = s * (hedge_odds - bet.odds) / hedge_odds
        else:
            # We're offering better than exchange? Negative margin.
            margin = s * (hedge_odds - bet.odds) / hedge_odds

        return round(margin, 2)

    def _calculate_expected_pnl(self, decision: RoutingDecision, bet: BetRecord) -> None:
        """Estimate expected P&L of this routing decision."""
        profile = decision.punter

        # Pass-through portion: guaranteed margin
        passthrough_pnl = decision.margin_captured

        # Absorbed portion: depends on punter's expected loss rate
        if decision.absorb_stake > 0:
            if profile.category == PunterCategory.MUG:
                # Mugs lose ~55% of even-money bets. Adjust for actual odds.
                implied_prob = 1.0 / bet.odds if bet.odds > 0 else 0.5
                # Mug's actual win rate is worse than implied probability
                mug_win_rate = implied_prob * 0.88  # Mugs underperform by ~12%
                ev_absorbed = decision.absorb_stake * (1 - mug_win_rate) - \
                    decision.absorb_stake * (bet.odds - 1) * mug_win_rate
            else:
                # Unknown/sharp — assume fair odds
                implied_prob = 1.0 / bet.odds if bet.odds > 0 else 0.5
                ev_absorbed = decision.absorb_stake * (1 - implied_prob) - \
                    decision.absorb_stake * (bet.odds - 1) * implied_prob
        else:
            ev_absorbed = 0.0

        decision.expected_pnl = round(passthrough_pnl + ev_absorbed, 2)
        decision.worst_case_pnl = round(
            -decision.absorb_stake * (bet.odds - 1) + passthrough_pnl, 2
        )

    def settle_match(self, match_id: str, winning_selection: str) -> dict:
        """Settle all bets for a completed match.

        Returns P&L breakdown.
        """
        match_decisions = [d for d in self._decisions if d.bet.match_id == match_id]

        absorbed_pnl = 0.0
        passthrough_pnl = 0.0

        for decision in match_decisions:
            bet = decision.bet
            won = bet.selection == winning_selection

            # Settle in the profiler
            self._profiler.settle_bet(bet.bet_id, bet.punter_id, won)

            # Our P&L on absorbed portion
            if decision.absorb_stake > 0:
                if won:
                    absorbed_pnl -= decision.absorb_stake * (bet.odds - 1)
                else:
                    absorbed_pnl += decision.absorb_stake

            # Pass-through margin is already locked in
            passthrough_pnl += decision.margin_captured

        self._total_absorbed_pnl += absorbed_pnl
        self._total_passthrough_pnl += passthrough_pnl

        total = absorbed_pnl + passthrough_pnl
        self._bankroll += total

        return {
            "match_id": match_id,
            "total_bets": len(match_decisions),
            "absorbed_pnl": round(absorbed_pnl, 2),
            "passthrough_pnl": round(passthrough_pnl, 2),
            "total_pnl": round(total, 2),
            "bankroll": round(self._bankroll, 2),
        }

    @property
    def liability(self) -> LiabilityBook:
        return self._liability

    def get_stats(self) -> dict:
        """Full statistics."""
        total_absorbed = sum(d.absorb_stake for d in self._decisions)
        total_hedged = sum(d.hedge_stake for d in self._decisions)
        total_margin = sum(d.margin_captured for d in self._decisions)

        by_action = {}
        for d in self._decisions:
            a = d.action.value
            if a not in by_action:
                by_action[a] = {"count": 0, "stake": 0.0}
            by_action[a]["count"] += 1
            by_action[a]["stake"] += d.bet.stake

        return {
            "total_bets_routed": len(self._decisions),
            "total_absorbed": round(total_absorbed, 2),
            "total_hedged": round(total_hedged, 2),
            "total_margin_captured": round(total_margin, 2),
            "absorbed_pnl": round(self._total_absorbed_pnl, 2),
            "passthrough_pnl": round(self._total_passthrough_pnl, 2),
            "total_pnl": round(self._total_absorbed_pnl + self._total_passthrough_pnl, 2),
            "bankroll": round(self._bankroll, 2),
            "liability": {
                "total": round(self._liability.total_liability, 2),
                "stakes_collected": round(self._liability.total_stakes_collected, 2),
            },
            "by_action": by_action,
        }
