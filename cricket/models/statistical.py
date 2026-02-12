"""
Statistical Base Model - Model A (30% ensemble weight).

A parametric model based on normal distribution of scoring margins.
Uses pre-match Elo ratings and in-match state to calculate win probability.

This provides a fast, interpretable baseline that doesn't require training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# Standard deviation of scoring margins by format
# Based on historical match data analysis
FORMAT_MARGIN_STDDEV = {
    "t20": 28.0,
    "odi": 42.0,
    "test": 120.0,  # Run-based, approximate
}

# Average first innings scores by format
FORMAT_AVG_SCORE = {
    "t20": 160.0,
    "odi": 260.0,
    "test": 350.0,
}

# Run rate standard deviations by phase
PHASE_RR_STDDEV = {
    "powerplay": 2.1,
    "middle": 1.5,
    "death": 2.8,
}


def normal_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class StatisticalPrediction:
    """Output from the statistical model."""
    team_a_win_prob: float
    team_b_win_prob: float
    draw_prob: float = 0.0  # Only for Tests
    confidence: float = 0.5

    @property
    def team_a_odds(self) -> float:
        return 1.0 / self.team_a_win_prob if self.team_a_win_prob > 0 else 999.0

    @property
    def team_b_odds(self) -> float:
        return 1.0 / self.team_b_win_prob if self.team_b_win_prob > 0 else 999.0


class StatisticalModel:
    """Parametric statistical model for cricket match outcome prediction.

    Uses normal distribution of scoring margins adjusted for:
    - Pre-match team ratings (Elo)
    - Current score trajectory vs expectation
    - Resources remaining (DLS-based)
    - Venue scoring patterns
    """

    def __init__(self, format_type: str = "t20"):
        self.format_type = format_type
        self.margin_stddev = FORMAT_MARGIN_STDDEV.get(format_type, 28.0)
        self.avg_score = FORMAT_AVG_SCORE.get(format_type, 160.0)

    def predict_pre_match(
        self,
        team_a_elo: float = 1500.0,
        team_b_elo: float = 1500.0,
        venue_avg_score: Optional[float] = None,
    ) -> StatisticalPrediction:
        """Pre-match win probability based on Elo ratings.

        Elo difference is converted to expected margin, then probability
        is derived from the normal distribution of margins.
        """
        # Elo to expected margin: ~30 Elo points = 1 run advantage
        expected_margin = (team_a_elo - team_b_elo) / 30.0

        prob_a = normal_cdf(expected_margin / self.margin_stddev)
        prob_a = max(0.01, min(0.99, prob_a))

        return StatisticalPrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=1.0 - prob_a,
            confidence=0.4,  # Pre-match has lower confidence
        )

    def predict_first_innings(
        self,
        features: dict[str, float],
        batting_is_team_a: bool = True,
    ) -> StatisticalPrediction:
        """First innings prediction based on current scoring trajectory.

        The key insight: if the batting team is tracking above/below
        expected score at this point, we adjust the expected final margin.
        """
        score = features.get("score", 0)
        overs = features.get("overs", 0)
        wickets = features.get("wickets", 0)
        resources_remaining = features.get("resources_remaining", 1.0)
        elo_diff = features.get("elo_diff", 0)
        venue_avg = features.get("venue_avg_score", self.avg_score)

        if overs < 0.1:
            return self.predict_pre_match(
                features.get("team_a_elo", 1500),
                features.get("team_b_elo", 1500),
            )

        # Project final score based on current trajectory
        total_overs = 20 if self.format_type == "t20" else 50
        proportion_completed = overs / total_overs
        run_rate = features.get("run_rate", 0)

        if proportion_completed > 0:
            # Projected final score = current + remaining * adjusted RR
            # RR tends to increase in death overs, decrease with wickets
            wicket_deceleration = max(0.5, 1.0 - wickets * 0.05)
            death_acceleration = 1.0 + max(0, (overs - 15) / 5 * 0.3) if self.format_type == "t20" else 1.0
            projected_rr = run_rate * wicket_deceleration * death_acceleration
            overs_remaining = total_overs - overs
            projected_score = score + projected_rr * overs_remaining
        else:
            projected_score = venue_avg

        # Expected margin based on projected score vs venue average
        score_advantage = projected_score - venue_avg
        elo_margin = elo_diff / 30.0

        if batting_is_team_a:
            total_expected_margin = score_advantage + elo_margin
        else:
            total_expected_margin = -score_advantage + elo_margin

        # Reduce variance as more of the innings is completed
        remaining_variance = self.margin_stddev * math.sqrt(
            max(0.1, 1.0 - proportion_completed * 0.7)
        )

        prob_a = normal_cdf(total_expected_margin / remaining_variance)
        prob_a = max(0.01, min(0.99, prob_a))

        # Confidence increases with overs bowled
        confidence = min(0.8, 0.3 + proportion_completed * 0.5)

        return StatisticalPrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=1.0 - prob_a,
            confidence=confidence,
        )

    def predict_second_innings(
        self,
        features: dict[str, float],
        chasing_is_team_a: bool = True,
    ) -> StatisticalPrediction:
        """Second innings prediction based on chase state.

        In a chase, the situation is more deterministic:
        we know the target and can calculate probability of
        reaching it given current resources.
        """
        target = features.get("target", 0)
        score = features.get("score", 0)
        runs_needed = features.get("runs_needed", target)
        overs = features.get("overs", 0)
        wickets = features.get("wickets", 0)
        resources_remaining = features.get("resources_remaining", 1.0)

        total_overs = 20 if self.format_type == "t20" else 50

        if target <= 0:
            return StatisticalPrediction(
                team_a_win_prob=0.5, team_b_win_prob=0.5, confidence=0.3
            )

        # Already won
        if runs_needed <= 0:
            prob_chaser = 0.99
        # Already lost (all out or overs exhausted)
        elif wickets >= 10 or (overs >= total_overs and runs_needed > 0):
            prob_chaser = 0.01
        else:
            # Expected runs from remaining resources
            overs_remaining = total_overs - overs
            run_rate = features.get("run_rate", 7.0)
            required_rr = features.get("required_run_rate", 0)

            # Expected remaining score based on resources
            expected_remaining = run_rate * overs_remaining * resources_remaining
            margin = expected_remaining - runs_needed

            # Variance decreases as match progresses
            remaining_stddev = self.margin_stddev * math.sqrt(
                max(0.05, resources_remaining)
            )

            prob_chaser = normal_cdf(margin / remaining_stddev)
            prob_chaser = max(0.01, min(0.99, prob_chaser))

        if chasing_is_team_a:
            prob_a = prob_chaser
        else:
            prob_a = 1.0 - prob_chaser

        # Confidence is higher in 2nd innings (known target)
        proportion_completed = overs / total_overs if total_overs > 0 else 0
        confidence = min(0.9, 0.4 + proportion_completed * 0.6)

        return StatisticalPrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=1.0 - prob_a,
            confidence=confidence,
        )

    def predict(
        self,
        features: dict[str, float],
        batting_is_team_a: bool = True,
    ) -> StatisticalPrediction:
        """Unified prediction interface.

        Routes to the appropriate innings model based on features.
        """
        innings = features.get("innings", 1)
        if innings >= 2 and features.get("target", 0) > 0:
            return self.predict_second_innings(features, batting_is_team_a)
        return self.predict_first_innings(features, batting_is_team_a)
