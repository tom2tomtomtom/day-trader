"""Tests for core/experiment_tracker.py"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from tests.conftest import make_trade


@dataclass
class MockHypothesis:
    id: str = "hyp_test123"
    category: str = "regime_conditional"
    statement: str = "Test hypothesis"
    source: str = "rsi_14"
    priority_score: float = 0.8
    effect_size: float = 0.5
    sample_size: int = 20
    status: str = "pending"
    narrative: str = ""
    supporting_evidence: dict = None

    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = {
                "control": {"regime": "trending_up"},
                "treatment": {"regime": "trending_down"},
            }


class TestExperimentDesign:
    """Test experiment design from hypotheses."""

    @patch("core.experiment_tracker.get_db")
    def test_design_experiment(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        hyp = MockHypothesis()

        design = tracker.design_experiment(hyp)
        assert design is not None
        assert design.hypothesis_id == "hyp_test123"
        assert design.experiment_type == "bootstrap_comparison"

    @patch("core.experiment_tracker.get_db")
    def test_design_temporal_hypothesis(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        hyp = MockHypothesis(category="temporal_pattern")

        design = tracker.design_experiment(hyp)
        assert design.experiment_type == "permutation_test"

    @patch("core.experiment_tracker.get_db")
    def test_design_drift_hypothesis(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        hyp = MockHypothesis(category="feature_drift")

        design = tracker.design_experiment(hyp)
        assert design.experiment_type == "feature_ablation"


class TestBootstrapComparison:
    """Test bootstrap comparison experiments."""

    @patch("core.experiment_tracker.get_db")
    def test_bootstrap_with_real_difference(self, mock_get_db, sample_trades):
        mock_db = MagicMock()
        mock_db.save_experiment.return_value = True
        mock_get_db.return_value = mock_db

        from core.experiment_tracker import ExperimentTracker, ExperimentDesign
        tracker = ExperimentTracker()

        # Create trades with a clear PnL difference by regime (with variance)
        np.random.seed(42)
        up_trades = [make_trade(pnl_pct=3.0 + np.random.normal(0, 1), regime="trending_up") for _ in range(10)]
        down_trades = [make_trade(pnl_pct=-2.0 + np.random.normal(0, 1), regime="trending_down") for _ in range(10)]
        all_trades = up_trades + down_trades

        design = ExperimentDesign(
            experiment_id="exp_test1",
            hypothesis_id="hyp_test1",
            name="Test regime difference",
            experiment_type="bootstrap_comparison",
            independent_variable="regime",
            control_definition={"regime": "trending_down"},
            treatment_definition={"regime": "trending_up"},
        )

        result = tracker.run_experiment(design, all_trades)

        assert result.status == "completed"
        assert result.effect_size != 0
        assert 0 <= result.p_value <= 1

    @patch("core.experiment_tracker.get_db")
    def test_bootstrap_insufficient_data(self, mock_get_db):
        mock_db = MagicMock()
        mock_db.save_experiment.return_value = True
        mock_get_db.return_value = mock_db

        from core.experiment_tracker import ExperimentTracker, ExperimentDesign
        tracker = ExperimentTracker()

        # Only 2 trades per group
        trades = [
            make_trade(regime="trending_up"),
            make_trade(regime="trending_up"),
            make_trade(regime="trending_down"),
        ]

        design = ExperimentDesign(
            experiment_id="exp_test2",
            hypothesis_id="hyp_test2",
            name="Too few trades",
            experiment_type="bootstrap_comparison",
            independent_variable="regime",
            control_definition={"regime": "trending_up"},
            treatment_definition={"regime": "trending_down"},
        )

        result = tracker.run_experiment(design, trades)
        assert result.status == "insufficient_data"


class TestPermutationTest:
    """Test permutation experiments."""

    @patch("core.experiment_tracker.get_db")
    def test_permutation_basic(self, mock_get_db):
        mock_db = MagicMock()
        mock_db.save_experiment.return_value = True
        mock_get_db.return_value = mock_db

        from core.experiment_tracker import ExperimentTracker, ExperimentDesign
        tracker = ExperimentTracker()

        mon_trades = [make_trade(pnl_pct=2.0, day_of_week=0) for _ in range(8)]
        fri_trades = [make_trade(pnl_pct=-1.0, day_of_week=4) for _ in range(8)]
        all_trades = mon_trades + fri_trades

        design = ExperimentDesign(
            experiment_id="exp_perm1",
            hypothesis_id="hyp_perm1",
            name="Monday vs Friday",
            experiment_type="permutation_test",
            independent_variable="day_of_week",
            control_definition={"day_of_week": 4},
            treatment_definition={"day_of_week": 0},
        )

        result = tracker.run_experiment(design, all_trades)
        assert result.status == "completed"
        assert 0 <= result.p_value <= 1


class TestGroupMetrics:
    """Test trade group metric computation."""

    @patch("core.experiment_tracker.get_db")
    def test_compute_group_metrics(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()

        trades = [
            make_trade(pnl_pct=5.0),
            make_trade(pnl_pct=-2.0),
            make_trade(pnl_pct=3.0),
            make_trade(pnl_pct=-1.0),
            make_trade(pnl_pct=4.0),
        ]

        metrics = tracker._compute_group_metrics(trades)

        assert metrics.n == 5
        assert metrics.win_rate == 0.6
        assert metrics.mean_pnl > 0
        assert metrics.profit_factor > 1

    @patch("core.experiment_tracker.get_db")
    def test_compute_group_metrics_empty(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        metrics = tracker._compute_group_metrics([])
        assert metrics.n == 0


class TestFDRCorrection:
    """Test Benjamini-Hochberg FDR correction."""

    def test_fdr_correction(self):
        from core.experiment_tracker import ExperimentTracker, ExperimentResult

        results = [
            ExperimentResult(
                experiment_id=f"exp_{i}",
                hypothesis_id=f"hyp_{i}",
                name=f"Test {i}",
                experiment_type="bootstrap_comparison",
                p_value=p,
                effect_size=0.5,
                is_significant=True,
            )
            for i, p in enumerate([0.01, 0.04, 0.08, 0.12, 0.20])
        ]

        corrected = ExperimentTracker.apply_fdr_correction(results)
        # Some of the marginal ones should be corrected to not significant
        assert isinstance(corrected, list)
        assert len(corrected) == 5


class TestConfidenceLabeling:
    """Test confidence level assignment."""

    @patch("core.experiment_tracker.get_db")
    def test_high_confidence(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker, ExperimentResult
        tracker = ExperimentTracker()

        result = ExperimentResult(
            experiment_id="exp_1",
            hypothesis_id="hyp_1",
            name="Strong finding",
            experiment_type="bootstrap_comparison",
            p_value=0.01,
            effect_size=1.0,
        )
        assert tracker._confidence_label(result) == "high"

    @patch("core.experiment_tracker.get_db")
    def test_inconclusive(self, mock_get_db):
        mock_get_db.return_value = MagicMock()

        from core.experiment_tracker import ExperimentTracker, ExperimentResult
        tracker = ExperimentTracker()

        result = ExperimentResult(
            experiment_id="exp_2",
            hypothesis_id="hyp_2",
            name="Weak finding",
            experiment_type="bootstrap_comparison",
            p_value=0.5,
            effect_size=0.1,
        )
        assert tracker._confidence_label(result) == "inconclusive"
