"""
Punter Profiling System.

Every customer who places a bet gets profiled. Over time we build a
picture of whether they're sharp (informed, winning) or mug (uninformed,
losing). This classification drives the entire business model:

- SHARP punters: their bets contain information. When they bet, the
  market is likely to move their way. We don't want this risk — pass
  it straight to Betfair and take a small margin.

- MUG punters: their bets are noise. They bet on favourites, chase
  losses, follow tips, and generally lose over time. We WANT this risk
  — absorb the bet ourselves because the expected value is in our favour.

- UNKNOWN punters: new customers without enough history. Treat cautiously
  — pass through to Betfair until we have enough data to classify.

Profiling signals:
- Win rate over last N bets
- Closing line value (did they beat the closing price?)
- Bet timing (sharp money comes early, mug money comes late)
- Stake patterns (consistent = sharp, erratic = mug)
- Market selection (obscure markets = sharp, Match Winner only = mug)
- Correlation with known sharp moves on Betfair
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


class PunterCategory(Enum):
    SHARP = "sharp"         # Informed bettor — pass to exchange
    MUG = "mug"             # Uninformed — absorb the risk
    UNKNOWN = "unknown"     # Not enough data yet


class RiskAction(Enum):
    ABSORB = "absorb"       # Keep the bet, we take the other side
    PASS_THROUGH = "pass"   # Lay off on Betfair, zero risk
    PARTIAL = "partial"     # Absorb some, hedge some


@dataclass
class BetRecord:
    """A single historical bet from a punter."""

    bet_id: str
    punter_id: str
    match_id: str
    market_type: str            # "match_odds", "innings_runs", etc.
    selection: str              # "Team A", "Over 165.5", etc.
    side: str                   # "back" (punter backs, we lay)
    odds: float                 # Odds given to punter
    stake: float                # Punter's stake (GBP)
    placed_at: datetime = field(default_factory=datetime.utcnow)

    # Closing line comparison
    closing_odds: Optional[float] = None    # Betfair price at market close
    got_closing_value: Optional[bool] = None  # Did they beat the closing line?

    # Outcome
    settled: bool = False
    won: bool = False
    pnl: float = 0.0           # Our P&L (positive = we made money)

    # Timing
    time_before_start_mins: Optional[float] = None  # How early before match

    def settle(self, won: bool, closing_odds: Optional[float] = None) -> None:
        """Settle the bet and calculate P&L."""
        self.settled = True
        self.won = won
        self.closing_odds = closing_odds

        if won:
            # Punter won — we pay out
            self.pnl = -(self.stake * (self.odds - 1))
        else:
            # Punter lost — we keep their stake
            self.pnl = self.stake

        # Closing line value: did they get better odds than the market closed at?
        if closing_odds and closing_odds > 0:
            self.got_closing_value = self.odds > closing_odds


@dataclass
class PunterProfile:
    """Accumulated profile of a single punter."""

    punter_id: str
    category: PunterCategory = PunterCategory.UNKNOWN
    confidence: float = 0.0     # 0-1, how sure we are of the category

    # Bet history
    bets: list[BetRecord] = field(default_factory=list)
    total_bets: int = 0
    total_settled: int = 0

    # Win/loss record
    wins: int = 0
    losses: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0     # Our P&L against this punter

    # Sharpness indicators (all 0-1, higher = sharper)
    win_rate: float = 0.0
    clv_rate: float = 0.0          # % of bets that beat closing line
    stake_consistency: float = 0.0  # Low variance = sharp
    timing_score: float = 0.0      # Early bets = sharp
    market_diversity: float = 0.0   # Diverse market selection = sharp

    # Composite score: 0 = pure mug, 1 = pure sharp
    sharpness_score: float = 0.5

    # Metadata
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    @property
    def avg_stake(self) -> float:
        return self.total_staked / self.total_bets if self.total_bets > 0 else 0

    @property
    def yield_pct(self) -> float:
        """Punter's yield (their perspective). Positive = they're winning."""
        if self.total_staked <= 0:
            return 0.0
        return -self.total_pnl / self.total_staked * 100


class PunterProfiler:
    """Builds and maintains profiles for all punters.

    Classification thresholds are calibrated from industry norms:
    - Recreational bettors win ~45-47% of even-money bets
    - Sharp bettors win ~53-56% of even-money bets
    - Closing line value is the #1 predictor of long-term profitability
    """

    def __init__(
        self,
        min_bets_to_classify: int = 20,
        sharp_threshold: float = 0.65,
        mug_threshold: float = 0.35,
        clv_weight: float = 0.35,
        win_rate_weight: float = 0.25,
        timing_weight: float = 0.15,
        consistency_weight: float = 0.15,
        diversity_weight: float = 0.10,
    ):
        self._min_bets = min_bets_to_classify
        self._sharp_threshold = sharp_threshold
        self._mug_threshold = mug_threshold

        # Weights for composite sharpness score
        self._weights = {
            "clv": clv_weight,
            "win_rate": win_rate_weight,
            "timing": timing_weight,
            "consistency": consistency_weight,
            "diversity": diversity_weight,
        }

        self._profiles: dict[str, PunterProfile] = {}

    def get_profile(self, punter_id: str) -> PunterProfile:
        """Get or create a punter profile."""
        if punter_id not in self._profiles:
            self._profiles[punter_id] = PunterProfile(punter_id=punter_id)
        return self._profiles[punter_id]

    def record_bet(self, bet: BetRecord) -> PunterProfile:
        """Record a new bet and update the punter's profile."""
        profile = self.get_profile(bet.punter_id)
        profile.bets.append(bet)
        profile.total_bets += 1
        profile.total_staked += bet.stake
        profile.last_seen = datetime.utcnow()
        return profile

    def settle_bet(
        self,
        bet_id: str,
        punter_id: str,
        won: bool,
        closing_odds: Optional[float] = None,
    ) -> PunterProfile:
        """Settle a bet and reclassify the punter."""
        profile = self.get_profile(punter_id)

        for bet in profile.bets:
            if bet.bet_id == bet_id and not bet.settled:
                bet.settle(won, closing_odds)
                profile.total_settled += 1
                profile.total_pnl += bet.pnl

                if won:
                    profile.wins += 1
                else:
                    profile.losses += 1

                break

        # Reclassify after settlement
        self._update_scores(profile)
        self._classify(profile)

        return profile

    def _update_scores(self, profile: PunterProfile) -> None:
        """Recalculate all sharpness indicators."""
        settled = [b for b in profile.bets if b.settled]
        if not settled:
            return

        # 1. Win rate (normalized: 0.5 = break-even, mapped to 0-1)
        raw_wr = profile.wins / len(settled) if settled else 0.5
        profile.win_rate = min(1.0, max(0.0, (raw_wr - 0.3) / 0.4))
        # 0.3 win rate → 0.0, 0.5 → 0.5, 0.7 → 1.0

        # 2. Closing line value rate
        clv_bets = [b for b in settled if b.got_closing_value is not None]
        if clv_bets:
            profile.clv_rate = sum(1 for b in clv_bets if b.got_closing_value) / len(clv_bets)
        else:
            profile.clv_rate = 0.5  # Neutral if no data

        # 3. Stake consistency (coefficient of variation — low = sharp)
        stakes = [b.stake for b in profile.bets]
        if len(stakes) >= 2:
            mean_stake = sum(stakes) / len(stakes)
            if mean_stake > 0:
                variance = sum((s - mean_stake) ** 2 for s in stakes) / len(stakes)
                cv = math.sqrt(variance) / mean_stake
                # CV < 0.2 = very consistent (sharp), CV > 1.0 = erratic (mug)
                profile.stake_consistency = max(0.0, min(1.0, 1.0 - cv))
            else:
                profile.stake_consistency = 0.5
        else:
            profile.stake_consistency = 0.5

        # 4. Timing score (early bets = sharp)
        timed_bets = [b for b in profile.bets if b.time_before_start_mins is not None]
        if timed_bets:
            avg_mins = sum(b.time_before_start_mins for b in timed_bets) / len(timed_bets)
            # > 60 mins before = sharp (0.8+), < 5 mins = mug (0.2)
            profile.timing_score = min(1.0, max(0.0, avg_mins / 75.0))
        else:
            profile.timing_score = 0.5

        # 5. Market diversity (betting multiple market types = sharp)
        market_types = set(b.market_type for b in profile.bets)
        # 1 market = mug, 3+ = sharp
        profile.market_diversity = min(1.0, (len(market_types) - 1) / 2.0)

        # Composite sharpness score
        profile.sharpness_score = (
            self._weights["clv"] * profile.clv_rate
            + self._weights["win_rate"] * profile.win_rate
            + self._weights["timing"] * profile.timing_score
            + self._weights["consistency"] * profile.stake_consistency
            + self._weights["diversity"] * profile.market_diversity
        )

    def _classify(self, profile: PunterProfile) -> None:
        """Classify punter based on sharpness score."""
        if profile.total_settled < self._min_bets:
            profile.category = PunterCategory.UNKNOWN
            profile.confidence = profile.total_settled / self._min_bets
            return

        if profile.sharpness_score >= self._sharp_threshold:
            profile.category = PunterCategory.SHARP
            # Confidence increases as score moves further from threshold
            profile.confidence = min(1.0, (profile.sharpness_score - self._sharp_threshold) / 0.2 + 0.5)
        elif profile.sharpness_score <= self._mug_threshold:
            profile.category = PunterCategory.MUG
            profile.confidence = min(1.0, (self._mug_threshold - profile.sharpness_score) / 0.2 + 0.5)
        else:
            # In the grey zone — could go either way
            profile.category = PunterCategory.UNKNOWN
            profile.confidence = 0.3

        logger.info(
            "Punter %s classified as %s (score=%.2f, confidence=%.2f, bets=%d)",
            profile.punter_id, profile.category.value,
            profile.sharpness_score, profile.confidence, profile.total_settled,
        )

    def get_all_profiles(self) -> list[PunterProfile]:
        return list(self._profiles.values())

    def get_stats(self) -> dict:
        """Aggregate stats across all punters."""
        profiles = list(self._profiles.values())
        by_cat = {c.value: [] for c in PunterCategory}
        for p in profiles:
            by_cat[p.category.value].append(p)

        return {
            "total_punters": len(profiles),
            "sharp": len(by_cat["sharp"]),
            "mug": len(by_cat["mug"]),
            "unknown": len(by_cat["unknown"]),
            "total_pnl": round(sum(p.total_pnl for p in profiles), 2),
            "pnl_from_mugs": round(sum(p.total_pnl for p in by_cat["mug"]), 2),
            "pnl_from_sharps": round(sum(p.total_pnl for p in by_cat["sharp"]), 2),
        }
