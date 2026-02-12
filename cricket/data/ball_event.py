"""
Ball-by-ball event data model.

Defines the canonical event that flows through the entire pipeline
from data ingestion through to signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class WicketType(Enum):
    BOWLED = "bowled"
    CAUGHT = "caught"
    LBW = "lbw"
    RUN_OUT = "run_out"
    STUMPED = "stumped"
    HIT_WICKET = "hit_wicket"
    RETIRED_HURT = "retired_hurt"
    RETIRED_OUT = "retired_out"
    OBSTRUCTING = "obstructing_the_field"
    TIMED_OUT = "timed_out"
    HANDLED_BALL = "handled_the_ball"


class ExtrasType(Enum):
    WIDE = "wide"
    NO_BALL = "no_ball"
    BYE = "bye"
    LEG_BYE = "leg_bye"
    PENALTY = "penalty"


@dataclass
class BallEvent:
    """A single delivery in a cricket match."""

    match_id: str
    innings: int  # 1 or 2 (3/4 for Tests)
    over: int  # 0-indexed over number
    ball: int  # Ball within over (1-6, can exceed for extras)
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str

    runs_off_bat: int = 0
    extras: int = 0
    extras_type: Optional[ExtrasType] = None
    total_runs: int = 0  # runs_off_bat + extras

    is_wicket: bool = False
    wicket_type: Optional[WicketType] = None
    player_dismissed: Optional[str] = None

    is_boundary_four: bool = False
    is_boundary_six: bool = False

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Cumulative state after this ball
    cumulative_score: int = 0
    cumulative_wickets: int = 0
    cumulative_overs: float = 0.0  # e.g. 5.3 = 5 overs, 3 balls

    # Optional exchange price snapshot at this ball
    back_price: Optional[float] = None
    lay_price: Optional[float] = None
    volume_matched: Optional[float] = None

    @property
    def over_ball_str(self) -> str:
        """Human-readable over.ball string, e.g. '5.3'."""
        return f"{self.over}.{self.ball}"

    @property
    def is_dot_ball(self) -> bool:
        return self.runs_off_bat == 0 and not self.is_wicket and self.extras == 0

    @property
    def is_legal_delivery(self) -> bool:
        return self.extras_type not in (ExtrasType.WIDE, ExtrasType.NO_BALL)

    @property
    def balls_bowled(self) -> int:
        """Total legal deliveries bowled so far (from cumulative_overs)."""
        overs = int(self.cumulative_overs)
        balls = round((self.cumulative_overs - overs) * 10)
        return overs * 6 + balls


@dataclass
class MatchInfo:
    """Pre-match metadata."""

    match_id: str
    format: str  # t20, odi, test
    team_a: str
    team_b: str
    venue: str
    city: str = ""
    date: str = ""
    toss_winner: str = ""
    toss_decision: str = ""  # bat / field
    venue_avg_first_innings_score: Optional[float] = None
    team_a_elo: float = 1500.0
    team_b_elo: float = 1500.0
    season: str = ""
    competition: str = ""
    gender: str = "male"
