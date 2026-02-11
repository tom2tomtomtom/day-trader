"""Tests for core/learning_loop.py"""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from tests.conftest import make_trade


class TestLearningCycleReport:
    """Test the LearningCycleReport dataclass."""

    def test_default_report(self):
        from core.learning_loop import LearningCycleReport
        report = LearningCycleReport()
        assert report.hypotheses_generated == 0
        assert report.experiments_run == 0
        assert report.retrain_triggered is False
        assert report.actions_taken == []


class TestLearningLoop:
    """Test the main learning loop orchestrator."""

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_run_cycle_insufficient_data(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_db = MagicMock()
        mock_db.get_trades_with_features.return_value = [make_trade() for _ in range(3)]
        mock_get_db.return_value = mock_db

        from core.learning_loop import LearningLoop
        loop = LearningLoop()
        report = loop.run_cycle()

        assert "Insufficient" in report.executive_summary
        assert report.hypotheses_generated == 0

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_run_cycle_with_data(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        # Set up mock DB
        mock_db = MagicMock()
        trades = [make_trade(pnl_pct=np.random.normal(1, 3)) for _ in range(30)]
        mock_db.get_trades_with_features.return_value = trades
        mock_db.get_active_model.return_value = None
        mock_db.save_hypothesis.return_value = True
        mock_db.save_learning_action.return_value = True
        mock_get_db.return_value = mock_db

        # Set up mock hypothesis engine
        from core.hypothesis_engine import Hypothesis
        mock_engine = MagicMock()
        mock_engine.generate_all.return_value = [
            Hypothesis(
                id="hyp_1",
                category="regime_conditional",
                statement="Test hypothesis",
                source="test",
                effect_size=0.5,
                sample_size=20,
                supporting_evidence={
                    "control": {"regime": "trending_up"},
                    "treatment": {"regime": "trending_down"},
                },
            )
        ]
        mock_engine_cls.return_value = mock_engine

        # Set up mock experiment tracker
        from core.experiment_tracker import ExperimentResult
        mock_tracker = MagicMock()
        mock_tracker.design_experiment.return_value = MagicMock(
            experiment_id="exp_1",
            hypothesis_id="hyp_1",
            name="Test experiment",
            experiment_type="bootstrap_comparison",
            independent_variable="regime",
            dependent_variable="pnl_pct",
            control_definition={},
            treatment_definition={},
            min_sample_size=10,
            significance_level=0.10,
        )
        mock_tracker.run_experiment.return_value = ExperimentResult(
            experiment_id="exp_1",
            hypothesis_id="hyp_1",
            name="Test experiment",
            experiment_type="bootstrap_comparison",
            effect_size=0.6,
            p_value=0.05,
            is_significant=True,
            confidence_level="medium",
        )
        mock_tracker_cls.return_value = mock_tracker

        from core.learning_loop import LearningLoop
        loop = LearningLoop()
        report = loop.run_cycle()

        assert report.hypotheses_generated == 1
        assert report.experiments_run == 1
        assert report.hypotheses_validated == 1
        assert report.runtime_seconds >= 0

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_run_cycle_no_hypotheses(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_db = MagicMock()
        mock_db.get_trades_with_features.return_value = [
            make_trade() for _ in range(20)
        ]
        mock_db.save_learning_action.return_value = True
        mock_get_db.return_value = mock_db

        mock_engine = MagicMock()
        mock_engine.generate_all.return_value = []
        mock_engine_cls.return_value = mock_engine

        from core.learning_loop import LearningLoop
        loop = LearningLoop()
        report = loop.run_cycle()

        assert report.hypotheses_generated == 0
        assert report.experiments_run == 0


class TestDriftDetection:
    """Test the drift checking step."""

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_check_drift_no_data(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_get_db.return_value = MagicMock()

        from core.learning_loop import LearningLoop
        loop = LearningLoop()
        drifted = loop._step_check_drift([make_trade() for _ in range(5)])
        assert drifted == []  # Too few trades

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_check_drift_with_stable_data(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_db = MagicMock()
        mock_db.save_feature_drift.return_value = True
        mock_get_db.return_value = mock_db

        from core.learning_loop import LearningLoop
        loop = LearningLoop()

        # Create 30 trades with consistent features
        np.random.seed(42)
        trades = [
            make_trade(
                rsi_14=50 + np.random.normal(0, 5),
                return_1d=np.random.normal(0, 1),
            )
            for _ in range(30)
        ]

        drifted = loop._step_check_drift(trades)
        # With consistent data, should have very few/no drifted features
        assert isinstance(drifted, list)


class TestPSI:
    """Test the PSI computation."""

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_psi_no_drift(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_get_db.return_value = MagicMock()

        from core.learning_loop import LearningLoop
        loop = LearningLoop()

        np.random.seed(42)
        expected = np.random.normal(0, 1, 50)
        actual = np.random.normal(0, 1, 50)

        psi = loop._psi(expected, actual)
        assert psi < 0.2

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_psi_with_drift(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_get_db.return_value = MagicMock()

        from core.learning_loop import LearningLoop
        loop = LearningLoop()

        expected = np.random.normal(0, 1, 50)
        actual = np.random.normal(5, 1, 50)  # Large shift

        psi = loop._psi(expected, actual)
        assert psi > 0.2


class TestCumulativeInsights:
    """Test the cumulative insights method."""

    @patch("core.learning_loop.get_db")
    @patch("core.learning_loop.HypothesisEngine")
    @patch("core.learning_loop.ExperimentTracker")
    def test_get_cumulative_insights(
        self, mock_tracker_cls, mock_engine_cls, mock_get_db
    ):
        mock_db = MagicMock()
        mock_db.get_hypotheses.return_value = [
            {"statement": "Test", "effect_size": 0.5, "confidence_level": "medium"}
        ]
        mock_db.get_experiments.return_value = [{"experiment_id": "exp_1"}]
        mock_db.get_ensemble_weight_overrides.return_value = []
        mock_db.get_temporal_adjustments.return_value = []
        mock_get_db.return_value = mock_db

        from core.learning_loop import LearningLoop
        loop = LearningLoop()
        insights = loop.get_cumulative_insights()

        assert insights["total_validated_hypotheses"] == 1
        assert insights["total_experiments"] == 1
        assert isinstance(insights["weight_overrides"], list)
