"""
Match State Engine - Layer 2.

Maintains a complete real-time model of the current match situation.
After each ball, recomputes all features that the pricing model requires.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from cricket.config import (
    FORMAT_OVERS,
    InningsPhase,
    MatchFormat,
    ODI_PHASES,
    T20_PHASES,
)
from cricket.data.ball_event import BallEvent, MatchInfo, WicketType

logger = logging.getLogger(__name__)

# DLS resource percentages (simplified lookup table)
# Format: (wickets_lost, overs_remaining) -> resource_pct
# Based on standard DLS tables for T20
DLS_RESOURCES_T20 = {
    (0, 20): 100.0, (0, 15): 84.2, (0, 10): 63.7, (0, 5): 37.3,
    (1, 20): 92.1, (1, 15): 78.1, (1, 10): 59.8, (1, 5): 35.6,
    (2, 20): 83.8, (2, 15): 71.7, (2, 10): 55.8, (2, 5): 33.8,
    (3, 20): 74.9, (3, 15): 64.8, (3, 10): 51.4, (3, 5): 31.8,
    (4, 20): 65.2, (4, 15): 57.2, (4, 10): 46.4, (4, 5): 29.6,
    (5, 20): 54.4, (5, 15): 48.6, (5, 10): 40.6, (5, 5): 26.9,
    (6, 20): 42.6, (6, 15): 38.9, (6, 10): 33.6, (6, 5): 23.4,
    (7, 20): 30.3, (7, 15): 28.3, (7, 10): 25.5, (7, 5): 19.0,
    (8, 20): 18.4, (8, 15): 17.6, (8, 10): 16.4, (8, 5): 13.4,
    (9, 20): 7.8, (9, 15): 7.6, (9, 10): 7.3, (9, 5): 6.6,
}


@dataclass
class PartnershipState:
    """Current batting partnership."""
    runs: int = 0
    balls: int = 0
    striker: str = ""
    non_striker: str = ""

    @property
    def strike_rate(self) -> float:
        return (self.runs / self.balls * 100) if self.balls > 0 else 0.0


@dataclass
class InningsState:
    """State for a single innings."""

    batting_team: str = ""
    bowling_team: str = ""
    score: int = 0
    wickets: int = 0
    overs_completed: float = 0.0
    legal_balls: int = 0
    target: Optional[int] = None  # Only for 2nd innings

    fours: int = 0
    sixes: int = 0
    dot_balls: int = 0
    extras_total: int = 0

    partnership: PartnershipState = field(default_factory=PartnershipState)
    fall_of_wickets: list[tuple[int, float, str]] = field(
        default_factory=list
    )  # (score, overs, player)

    # Recent ball history for momentum calculations
    recent_balls: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def run_rate(self) -> float:
        overs = self.legal_balls / 6.0
        return self.score / overs if overs > 0 else 0.0

    @property
    def required_run_rate(self) -> Optional[float]:
        if self.target is None:
            return None
        runs_needed = self.target - self.score
        balls_remaining = max(1, 120 - self.legal_balls)  # Assume T20
        overs_remaining = balls_remaining / 6.0
        return runs_needed / overs_remaining if overs_remaining > 0 else float("inf")

    @property
    def run_rate_last_5_overs(self) -> float:
        """Run rate over the last 30 legal deliveries."""
        if not self.recent_balls:
            return 0.0
        recent_runs = sum(b.total_runs for b in self.recent_balls)
        recent_legal = sum(1 for b in self.recent_balls if b.is_legal_delivery)
        overs = recent_legal / 6.0
        return recent_runs / overs if overs > 0 else 0.0

    @property
    def wickets_last_5_overs(self) -> int:
        return sum(1 for b in self.recent_balls if b.is_wicket)


@dataclass
class MatchState:
    """Complete match state at any point during the game.

    This is the central data structure that feeds into the pricing model.
    It is updated ball-by-ball by the MatchStateEngine.
    """

    match_info: MatchInfo
    match_format: MatchFormat
    innings: dict[int, InningsState] = field(default_factory=dict)
    current_innings: int = 1
    is_complete: bool = False
    winner: Optional[str] = None

    # Exchange prices at this point
    back_price_team_a: Optional[float] = None
    lay_price_team_a: Optional[float] = None
    back_price_team_b: Optional[float] = None
    lay_price_team_b: Optional[float] = None

    @property
    def current_innings_state(self) -> InningsState:
        return self.innings.get(
            self.current_innings, InningsState()
        )

    @property
    def overs_in_format(self) -> Optional[int]:
        return FORMAT_OVERS.get(self.match_format)


class MatchStateEngine:
    """Processes ball events and maintains match state.

    This is the core Layer 2 component. It receives BallEvents from
    the data ingestion layer and produces updated MatchState objects
    with all derived features computed.
    """

    def __init__(self, match_info: MatchInfo):
        fmt = MatchFormat(match_info.format)
        self._state = MatchState(
            match_info=match_info,
            match_format=fmt,
        )
        # Initialize first innings
        self._state.innings[1] = InningsState()
        self._balls_processed = 0

    @property
    def state(self) -> MatchState:
        return self._state

    def process_ball(self, event: BallEvent) -> MatchState:
        """Process a single ball event and update match state.

        Returns the updated match state with all features recomputed.
        """
        # Switch innings if needed
        if event.innings != self._state.current_innings:
            self._switch_innings(event)

        inn = self._state.innings[event.innings]
        inn.batting_team = event.batting_team
        inn.bowling_team = event.bowling_team

        # Update score
        inn.score = event.cumulative_score
        inn.wickets = event.cumulative_wickets
        inn.overs_completed = event.cumulative_overs

        if event.is_legal_delivery:
            inn.legal_balls += 1

        # Track boundaries
        if event.is_boundary_four:
            inn.fours += 1
        if event.is_boundary_six:
            inn.sixes += 1
        if event.is_dot_ball:
            inn.dot_balls += 1
        if event.extras > 0:
            inn.extras_total += event.extras

        # Update partnership
        if event.is_wicket and event.wicket_type != WicketType.RETIRED_HURT:
            inn.fall_of_wickets.append(
                (inn.score, inn.overs_completed, event.player_dismissed or "")
            )
            inn.partnership = PartnershipState()
        else:
            inn.partnership.runs += event.total_runs
            if event.is_legal_delivery:
                inn.partnership.balls += 1
            inn.partnership.striker = event.striker
            inn.partnership.non_striker = event.non_striker

        # Add to recent balls
        inn.recent_balls.append(event)

        # Update exchange prices if present
        if event.back_price is not None:
            if event.batting_team == self._state.match_info.team_a:
                self._state.back_price_team_a = event.back_price
                self._state.lay_price_team_a = event.lay_price
            else:
                self._state.back_price_team_b = event.back_price
                self._state.lay_price_team_b = event.lay_price

        self._balls_processed += 1
        return self._state

    def _switch_innings(self, event: BallEvent) -> None:
        """Handle innings transition."""
        prev_innings = self._state.current_innings
        self._state.current_innings = event.innings

        if event.innings not in self._state.innings:
            self._state.innings[event.innings] = InningsState()

        # Set target for 2nd innings
        if event.innings == 2 and prev_innings == 1:
            first_inn = self._state.innings[1]
            self._state.innings[2].target = first_inn.score + 1

    def get_features(self) -> dict[str, float]:
        """Extract ML features from current match state.

        Returns a flat dictionary of numeric features suitable
        for input to the pricing model.
        """
        inn = self._state.current_innings_state
        info = self._state.match_info
        fmt = self._state.match_format
        total_overs = FORMAT_OVERS.get(fmt, 20) or 90

        features: dict[str, float] = {}

        # Core state
        features["score"] = float(inn.score)
        features["wickets"] = float(inn.wickets)
        features["legal_balls"] = float(inn.legal_balls)
        features["overs"] = inn.legal_balls / 6.0
        features["run_rate"] = inn.run_rate
        features["innings"] = float(self._state.current_innings)

        # Resources remaining
        balls_remaining = max(0, total_overs * 6 - inn.legal_balls)
        features["balls_remaining"] = float(balls_remaining)
        features["overs_remaining"] = balls_remaining / 6.0
        features["wickets_remaining"] = float(10 - inn.wickets)
        features["resources_remaining"] = self._calculate_resources(
            inn.wickets, balls_remaining / 6.0
        )

        # Required rate (2nd innings)
        if inn.target is not None:
            features["target"] = float(inn.target)
            features["runs_needed"] = float(inn.target - inn.score)
            features["required_run_rate"] = inn.required_run_rate or 0.0
            features["run_rate_pressure"] = (
                (inn.required_run_rate or 0.0) - inn.run_rate
            )
        else:
            features["target"] = 0.0
            features["runs_needed"] = 0.0
            features["required_run_rate"] = 0.0
            features["run_rate_pressure"] = 0.0

        # Momentum
        features["run_rate_last_5"] = inn.run_rate_last_5_overs
        features["wickets_last_5"] = float(inn.wickets_last_5_overs)
        features["momentum"] = inn.run_rate_last_5_overs - inn.run_rate
        features["dot_ball_pct"] = (
            inn.dot_balls / inn.legal_balls if inn.legal_balls > 0 else 0.0
        )
        features["boundary_pct"] = (
            (inn.fours + inn.sixes) / inn.legal_balls
            if inn.legal_balls > 0
            else 0.0
        )

        # Partnership
        features["partnership_runs"] = float(inn.partnership.runs)
        features["partnership_balls"] = float(inn.partnership.balls)
        features["partnership_sr"] = inn.partnership.strike_rate

        # Score trajectory
        venue_avg = info.venue_avg_first_innings_score or 160.0
        expected_at_this_point = venue_avg * (inn.legal_balls / (total_overs * 6))
        features["score_trajectory_delta"] = inn.score - expected_at_this_point
        features["venue_avg_score"] = venue_avg

        # Phase encoding
        phase = self._get_phase(inn.legal_balls, fmt)
        features["is_powerplay"] = 1.0 if phase == InningsPhase.POWERPLAY else 0.0
        features["is_middle"] = 1.0 if phase == InningsPhase.MIDDLE else 0.0
        features["is_death"] = 1.0 if phase == InningsPhase.DEATH else 0.0

        # Team strength (Elo-based)
        features["team_a_elo"] = info.team_a_elo
        features["team_b_elo"] = info.team_b_elo
        features["elo_diff"] = info.team_a_elo - info.team_b_elo

        # First innings score (available in 2nd innings)
        if self._state.current_innings >= 2 and 1 in self._state.innings:
            features["first_innings_score"] = float(self._state.innings[1].score)
        else:
            features["first_innings_score"] = 0.0

        return features

    def _calculate_resources(self, wickets: int, overs_remaining: float) -> float:
        """Calculate DLS-style resources remaining (0-100)."""
        overs_key = min(20, max(0, round(overs_remaining / 5) * 5))
        key = (wickets, overs_key)
        if key in DLS_RESOURCES_T20:
            return DLS_RESOURCES_T20[key] / 100.0

        # Interpolate
        if wickets >= 10:
            return 0.0
        base = max(0, 100 - wickets * 10 - (20 - overs_remaining) * 3)
        return base / 100.0

    @staticmethod
    def _get_phase(legal_balls: int, fmt: MatchFormat) -> InningsPhase:
        """Determine current innings phase."""
        current_over = legal_balls // 6 + 1
        phases = T20_PHASES if fmt == MatchFormat.T20 else ODI_PHASES

        for phase, (start, end) in phases.items():
            if start <= current_over <= end:
                return phase
        return InningsPhase.DEATH
