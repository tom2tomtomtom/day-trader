"""Tests for the Pricing Models (Layer 3)."""

from __future__ import annotations

import pytest

from cricket.models.statistical import StatisticalModel
from cricket.models.xgboost_model import CricketXGBoostModel
from cricket.models.ensemble import EnsemblePricingModel


class TestStatisticalModel:
    def test_pre_match_equal_teams(self):
        model = StatisticalModel("t20")
        pred = model.predict_pre_match(1500, 1500)
        assert pred.team_a_win_prob == pytest.approx(0.5, abs=0.02)
        assert pred.team_b_win_prob == pytest.approx(0.5, abs=0.02)

    def test_pre_match_stronger_team(self):
        model = StatisticalModel("t20")
        pred = model.predict_pre_match(1600, 1400)
        assert pred.team_a_win_prob > 0.55
        assert pred.team_b_win_prob < 0.45

    def test_first_innings_above_expectation(self):
        model = StatisticalModel("t20")
        features = {
            "score": 80.0,
            "overs": 8.0,
            "wickets": 1.0,
            "run_rate": 10.0,
            "resources_remaining": 0.7,
            "venue_avg_score": 160.0,
            "team_a_elo": 1500.0,
            "team_b_elo": 1500.0,
            "elo_diff": 0.0,
            "momentum": 1.0,
        }
        pred = model.predict_first_innings(features, batting_is_team_a=True)
        # Tracking above expectation → should favor batting team
        assert pred.team_a_win_prob > 0.5

    def test_second_innings_chasing_easily(self):
        model = StatisticalModel("t20")
        features = {
            "target": 140.0,
            "score": 130.0,
            "runs_needed": 10.0,
            "overs": 16.0,
            "wickets": 1.0,
            "run_rate": 8.125,
            "required_run_rate": 2.5,
            "resources_remaining": 0.35,
        }
        pred = model.predict_second_innings(features, chasing_is_team_a=True)
        # 10 runs needed, 4 overs left, RR 8+ → chasing team favored
        assert pred.team_a_win_prob > 0.5

    def test_second_innings_behind(self):
        model = StatisticalModel("t20")
        features = {
            "target": 200.0,
            "score": 40.0,
            "runs_needed": 160.0,
            "overs": 8.0,
            "wickets": 5.0,
            "run_rate": 5.0,
            "required_run_rate": 13.3,
            "resources_remaining": 0.25,
        }
        pred = model.predict_second_innings(features, chasing_is_team_a=True)
        assert pred.team_a_win_prob < 0.3  # Chasing team in deep trouble

    def test_probability_bounds(self):
        model = StatisticalModel("t20")
        # Even extreme scenarios should stay within [0.01, 0.99]
        pred = model.predict_pre_match(2000, 1000)
        assert 0.01 <= pred.team_a_win_prob <= 0.99
        assert 0.01 <= pred.team_b_win_prob <= 0.99


class TestXGBoostHeuristic:
    def test_heuristic_mode_active(self):
        model = CricketXGBoostModel()
        assert not model.is_trained

    def test_first_innings_prediction(self):
        model = CricketXGBoostModel()
        features = {
            "score": 50.0,
            "wickets": 2.0,
            "overs": 6.0,
            "run_rate": 8.33,
            "venue_avg_score": 160.0,
            "elo_diff": 50.0,
            "score_trajectory_delta": 2.0,
            "resources_remaining": 0.75,
            "momentum": 0.5,
            "innings": 1.0,
            "target": 0.0,
        }
        pred = model.predict(features, batting_is_team_a=True)
        assert 0.02 <= pred.team_a_win_prob <= 0.98

    def test_chase_prediction(self):
        model = CricketXGBoostModel()
        features = {
            "score": 80.0,
            "wickets": 1.0,
            "overs": 10.0,
            "run_rate": 8.0,
            "required_run_rate": 7.0,
            "overs_remaining": 10.0,
            "resources_remaining": 0.65,
            "innings": 2.0,
            "target": 150.0,
            "runs_needed": 70.0,
        }
        pred = model.predict(features, batting_is_team_a=True)
        # Cruising in the chase
        assert pred.team_a_win_prob > 0.5


class TestEnsembleModel:
    def test_ensemble_combines_models(self):
        model = EnsemblePricingModel(match_format="t20")
        features = {
            "score": 60.0,
            "wickets": 2.0,
            "overs": 8.0,
            "legal_balls": 48,
            "run_rate": 7.5,
            "innings": 1.0,
            "balls_remaining": 72.0,
            "overs_remaining": 12.0,
            "wickets_remaining": 8.0,
            "resources_remaining": 0.7,
            "target": 0.0,
            "runs_needed": 0.0,
            "required_run_rate": 0.0,
            "run_rate_pressure": 0.0,
            "run_rate_last_5": 7.0,
            "wickets_last_5": 1.0,
            "momentum": -0.5,
            "dot_ball_pct": 0.35,
            "boundary_pct": 0.15,
            "partnership_runs": 20.0,
            "partnership_balls": 15.0,
            "partnership_sr": 133.3,
            "score_trajectory_delta": -4.0,
            "venue_avg_score": 160.0,
            "is_powerplay": 0.0,
            "is_middle": 1.0,
            "is_death": 0.0,
            "team_a_elo": 1500.0,
            "team_b_elo": 1500.0,
            "elo_diff": 0.0,
            "first_innings_score": 0.0,
        }
        pred = model.predict(features)

        assert 0.01 <= pred.team_a_win_prob <= 0.99
        assert pred.team_a_win_prob + pred.team_b_win_prob == pytest.approx(1.0, abs=0.01)
        assert pred.confidence in ("HIGH", "MEDIUM", "LOW")
        assert pred.team_a_fair_odds > 1.0
        assert pred.team_b_fair_odds > 1.0

    def test_ensemble_fair_odds(self):
        model = EnsemblePricingModel(match_format="t20")
        features = {
            "score": 0.0, "wickets": 0.0, "overs": 0.0, "legal_balls": 0,
            "run_rate": 0.0, "innings": 1.0, "balls_remaining": 120.0,
            "overs_remaining": 20.0, "wickets_remaining": 10.0,
            "resources_remaining": 1.0, "target": 0.0, "runs_needed": 0.0,
            "required_run_rate": 0.0, "run_rate_pressure": 0.0,
            "run_rate_last_5": 0.0, "wickets_last_5": 0.0, "momentum": 0.0,
            "dot_ball_pct": 0.0, "boundary_pct": 0.0, "partnership_runs": 0.0,
            "partnership_balls": 0.0, "partnership_sr": 0.0,
            "score_trajectory_delta": 0.0, "venue_avg_score": 160.0,
            "is_powerplay": 1.0, "is_middle": 0.0, "is_death": 0.0,
            "team_a_elo": 1600.0, "team_b_elo": 1400.0, "elo_diff": 200.0,
            "first_innings_score": 0.0,
        }
        pred = model.predict(features)

        # Team A stronger → lower odds (higher probability)
        assert pred.team_a_fair_odds < pred.team_b_fair_odds
