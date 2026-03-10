"""Tests for the Match State Engine (Layer 2)."""

from __future__ import annotations

import pytest

from cricket.data.ball_event import BallEvent, MatchInfo, WicketType
from cricket.state.match_state import MatchStateEngine


@pytest.fixture
def match_info() -> MatchInfo:
    return MatchInfo(
        match_id="test_001",
        format="t20",
        team_a="Thunder",
        team_b="Strikers",
        venue="Test Ground",
        venue_avg_first_innings_score=160.0,
        team_a_elo=1550,
        team_b_elo=1480,
    )


@pytest.fixture
def state_engine(match_info: MatchInfo) -> MatchStateEngine:
    return MatchStateEngine(match_info)


def make_ball(
    over: int,
    ball: int,
    runs: int = 0,
    wicket: bool = False,
    cum_score: int = 0,
    cum_wickets: int = 0,
    batting_team: str = "Thunder",
) -> BallEvent:
    return BallEvent(
        match_id="test_001",
        innings=1,
        over=over,
        ball=ball,
        batting_team=batting_team,
        bowling_team="Strikers" if batting_team == "Thunder" else "Thunder",
        striker="Bat_1",
        non_striker="Bat_2",
        bowler="Bowl_1",
        runs_off_bat=runs,
        total_runs=runs,
        is_wicket=wicket,
        wicket_type=WicketType.BOWLED if wicket else None,
        is_boundary_four=(runs == 4),
        is_boundary_six=(runs == 6),
        cumulative_score=cum_score,
        cumulative_wickets=cum_wickets,
        cumulative_overs=over + ball / 10.0,
    )


class TestMatchStateEngine:
    def test_initial_state(self, state_engine: MatchStateEngine):
        state = state_engine.state
        assert state.current_innings == 1
        assert state.match_info.match_id == "test_001"
        assert not state.is_complete

    def test_process_single_ball(self, state_engine: MatchStateEngine):
        event = make_ball(0, 1, runs=4, cum_score=4)
        state = state_engine.process_ball(event)

        inn = state.current_innings_state
        assert inn.score == 4
        assert inn.wickets == 0
        assert inn.fours == 1

    def test_process_wicket(self, state_engine: MatchStateEngine):
        event = make_ball(0, 1, runs=0, wicket=True, cum_score=0, cum_wickets=1)
        state = state_engine.process_ball(event)

        inn = state.current_innings_state
        assert inn.wickets == 1
        assert len(inn.fall_of_wickets) == 1

    def test_run_rate_calculation(self, state_engine: MatchStateEngine):
        # 6 balls, 12 runs = RR of 12.0
        score = 0
        for ball in range(1, 7):
            score += 2
            event = make_ball(0, ball, runs=2, cum_score=score, cum_wickets=0)
            state_engine.process_ball(event)

        inn = state_engine.state.current_innings_state
        assert inn.run_rate == pytest.approx(12.0, abs=0.5)

    def test_feature_extraction(self, state_engine: MatchStateEngine):
        # Process a few balls
        events = [
            make_ball(0, 1, runs=4, cum_score=4),
            make_ball(0, 2, runs=1, cum_score=5),
            make_ball(0, 3, runs=0, cum_score=5),
            make_ball(0, 4, runs=6, cum_score=11),
            make_ball(0, 5, runs=0, wicket=True, cum_score=11, cum_wickets=1),
            make_ball(0, 6, runs=2, cum_score=13, cum_wickets=1),
        ]
        for e in events:
            state_engine.process_ball(e)

        features = state_engine.get_features()

        assert features["score"] == 13.0
        assert features["wickets"] == 1.0
        assert features["innings"] == 1.0
        assert features["is_powerplay"] == 1.0
        assert features["elo_diff"] == 70.0  # 1550 - 1480
        assert features["wickets_remaining"] == 9.0
        assert "run_rate" in features
        assert "resources_remaining" in features
        assert "score_trajectory_delta" in features

    def test_innings_switch(self, state_engine: MatchStateEngine):
        # First innings ball
        e1 = make_ball(0, 1, runs=4, cum_score=4)
        state_engine.process_ball(e1)

        # Switch to second innings
        e2 = BallEvent(
            match_id="test_001",
            innings=2,
            over=0,
            ball=1,
            batting_team="Strikers",
            bowling_team="Thunder",
            striker="Bat_A",
            non_striker="Bat_B",
            bowler="Bowl_X",
            runs_off_bat=1,
            total_runs=1,
            cumulative_score=1,
            cumulative_wickets=0,
            cumulative_overs=0.1,
        )
        state = state_engine.process_ball(e2)

        assert state.current_innings == 2
        inn = state.current_innings_state
        assert inn.score == 1
        assert inn.target == 5  # First innings score (4) + 1

    def test_boundary_tracking(self, state_engine: MatchStateEngine):
        events = [
            make_ball(0, 1, runs=4, cum_score=4),
            make_ball(0, 2, runs=6, cum_score=10),
            make_ball(0, 3, runs=4, cum_score=14),
            make_ball(0, 4, runs=1, cum_score=15),
        ]
        for e in events:
            state_engine.process_ball(e)

        inn = state_engine.state.current_innings_state
        assert inn.fours == 2
        assert inn.sixes == 1

    def test_dot_ball_tracking(self, state_engine: MatchStateEngine):
        events = [
            make_ball(0, 1, runs=0, cum_score=0),
            make_ball(0, 2, runs=0, cum_score=0),
            make_ball(0, 3, runs=1, cum_score=1),
            make_ball(0, 4, runs=0, cum_score=1),
        ]
        for e in events:
            state_engine.process_ball(e)

        inn = state_engine.state.current_innings_state
        assert inn.dot_balls == 3

    def test_phase_detection(self, state_engine: MatchStateEngine):
        # Powerplay (overs 1-6)
        event = make_ball(2, 1, runs=1, cum_score=1)
        state_engine.process_ball(event)
        features = state_engine.get_features()
        assert features["is_powerplay"] == 1.0

    def test_second_innings_features(self, state_engine: MatchStateEngine):
        # Set up first innings score
        e1 = make_ball(19, 6, runs=4, cum_score=165)
        state_engine.process_ball(e1)

        # Second innings
        e2 = BallEvent(
            match_id="test_001",
            innings=2,
            over=5,
            ball=3,
            batting_team="Strikers",
            bowling_team="Thunder",
            striker="Bat_A",
            non_striker="Bat_B",
            bowler="Bowl_X",
            runs_off_bat=2,
            total_runs=2,
            cumulative_score=50,
            cumulative_wickets=2,
            cumulative_overs=5.3,
        )
        state_engine.process_ball(e2)

        features = state_engine.get_features()
        assert features["target"] == 166.0
        assert features["runs_needed"] == 116.0
        assert features["first_innings_score"] == 165.0
        assert features["required_run_rate"] > 0
