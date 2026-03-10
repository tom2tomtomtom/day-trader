"""
XGBoost Gradient Boosted Model - Model B (45% ensemble weight).

A gradient-boosted decision tree trained on historical ball-by-ball data.
This is the primary model in the ensemble, providing the most accurate
probability estimates after sufficient training data.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Features used by the XGBoost model (in order)
FEATURE_COLUMNS = [
    "score",
    "wickets",
    "legal_balls",
    "overs",
    "run_rate",
    "innings",
    "balls_remaining",
    "overs_remaining",
    "wickets_remaining",
    "resources_remaining",
    "target",
    "runs_needed",
    "required_run_rate",
    "run_rate_pressure",
    "run_rate_last_5",
    "wickets_last_5",
    "momentum",
    "dot_ball_pct",
    "boundary_pct",
    "partnership_runs",
    "partnership_balls",
    "partnership_sr",
    "score_trajectory_delta",
    "venue_avg_score",
    "is_powerplay",
    "is_middle",
    "is_death",
    "elo_diff",
    "first_innings_score",
]


@dataclass
class XGBoostPrediction:
    """Output from the XGBoost model."""
    team_a_win_prob: float
    team_b_win_prob: float
    confidence: float
    feature_importance: Optional[dict[str, float]] = None


class CricketXGBoostModel:
    """XGBoost-based cricket match outcome predictor.

    Can operate in two modes:
    1. Trained mode: Uses a trained XGBoost model loaded from disk
    2. Heuristic mode: Falls back to a rule-based approximation
       when no trained model is available (for initial deployment)
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = model_path
        self._feature_columns = FEATURE_COLUMNS
        self._is_trained = False

        if model_path and model_path.exists():
            self._load_model(model_path)

    def _load_model(self, path: Path) -> None:
        """Load a trained XGBoost model from disk."""
        try:
            import xgboost as xgb

            self._model = xgb.XGBClassifier()
            self._model.load_model(str(path))
            self._is_trained = True
            logger.info("Loaded trained XGBoost model from %s", path)
        except ImportError:
            logger.warning("xgboost not installed, using heuristic mode")
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", path, e)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict(
        self,
        features: dict[str, float],
        batting_is_team_a: bool = True,
    ) -> XGBoostPrediction:
        """Predict match outcome from current state features.

        Args:
            features: Feature dictionary from MatchStateEngine.get_features()
            batting_is_team_a: Whether the batting team is team A

        Returns:
            XGBoostPrediction with win probabilities
        """
        if self._is_trained and self._model is not None:
            return self._predict_trained(features, batting_is_team_a)
        return self._predict_heuristic(features, batting_is_team_a)

    def _predict_trained(
        self,
        features: dict[str, float],
        batting_is_team_a: bool,
    ) -> XGBoostPrediction:
        """Use the trained XGBoost model for prediction."""
        feature_vector = np.array(
            [[features.get(col, 0.0) for col in self._feature_columns]]
        )

        proba = self._model.predict_proba(feature_vector)[0]
        prob_a = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Get feature importance
        importance = None
        if hasattr(self._model, "feature_importances_"):
            importance = dict(
                zip(self._feature_columns, self._model.feature_importances_)
            )

        return XGBoostPrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=1.0 - prob_a,
            confidence=0.7,
            feature_importance=importance,
        )

    def _predict_heuristic(
        self,
        features: dict[str, float],
        batting_is_team_a: bool,
    ) -> XGBoostPrediction:
        """Heuristic fallback when no trained model is available.

        Uses a weighted combination of key features to approximate
        what a trained XGBoost model would predict. This allows the
        system to operate before training data is available.
        """
        innings = features.get("innings", 1)
        overs = features.get("overs", 0)

        if innings >= 2 and features.get("target", 0) > 0:
            return self._chase_heuristic(features, batting_is_team_a)

        return self._first_innings_heuristic(features, batting_is_team_a)

    def _first_innings_heuristic(
        self,
        features: dict[str, float],
        batting_is_team_a: bool,
    ) -> XGBoostPrediction:
        """First innings heuristic prediction."""
        score = features.get("score", 0)
        wickets = features.get("wickets", 0)
        overs = features.get("overs", 0)
        run_rate = features.get("run_rate", 0)
        venue_avg = features.get("venue_avg_score", 160.0)
        elo_diff = features.get("elo_diff", 0)
        trajectory = features.get("score_trajectory_delta", 0)
        resources = features.get("resources_remaining", 1.0)

        # Base probability from Elo
        base_prob = 0.5 + (elo_diff / 30.0) / (2 * 28.0)

        # Score trajectory adjustment
        trajectory_adj = trajectory / venue_avg * 0.15

        # Wicket pressure adjustment
        wicket_adj = -wickets * 0.02

        # Run rate momentum
        momentum = features.get("momentum", 0)
        momentum_adj = momentum / 20.0 * 0.05

        if batting_is_team_a:
            prob_a = base_prob + trajectory_adj + wicket_adj + momentum_adj
        else:
            prob_a = base_prob - trajectory_adj - wicket_adj - momentum_adj

        prob_a = max(0.02, min(0.98, prob_a))

        # Lower confidence for heuristic
        confidence = min(0.5, 0.2 + overs / 40.0)

        return XGBoostPrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=1.0 - prob_a,
            confidence=confidence,
        )

    def _chase_heuristic(
        self,
        features: dict[str, float],
        batting_is_team_a: bool,
    ) -> XGBoostPrediction:
        """Second innings chase heuristic prediction."""
        runs_needed = features.get("runs_needed", 100)
        overs_remaining = features.get("overs_remaining", 20)
        wickets = features.get("wickets", 0)
        run_rate = features.get("run_rate", 7.0)
        required_rr = features.get("required_run_rate", 7.0)
        resources = features.get("resources_remaining", 1.0)

        if runs_needed <= 0:
            prob_chaser = 0.99
        elif wickets >= 10:
            prob_chaser = 0.01
        else:
            # Core metric: required rate vs current rate
            rate_diff = required_rr - run_rate
            # Positive = under pressure, negative = cruising

            # Resources-weighted probability
            if overs_remaining > 0:
                expected_remaining = run_rate * overs_remaining * (1 - wickets * 0.05)
                surplus = expected_remaining - runs_needed

                # Normalize by expected remaining
                if expected_remaining > 0:
                    surplus_ratio = surplus / expected_remaining
                else:
                    surplus_ratio = -1.0

                # Sigmoid-like mapping
                prob_chaser = 1.0 / (1.0 + np.exp(-surplus_ratio * 5))
            else:
                prob_chaser = 0.01

            prob_chaser = float(max(0.02, min(0.98, prob_chaser)))

        if batting_is_team_a:
            prob_a = prob_chaser
        else:
            prob_a = 1.0 - prob_chaser

        overs = features.get("overs", 0)
        total_overs = overs + overs_remaining
        confidence = min(0.6, 0.3 + overs / (total_overs * 2) if total_overs > 0 else 0.3)

        return XGBoostPrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=1.0 - prob_a,
            confidence=confidence,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> dict[str, float]:
        """Train the XGBoost model on historical data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (1 = team_a_win, 0 = team_b_win)
            save_path: Path to save the trained model

        Returns:
            Training metrics dict
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score
        except ImportError:
            raise RuntimeError(
                "xgboost and scikit-learn required for training. "
                "Install with: pip install xgboost scikit-learn"
            )

        self._model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )

        # Cross-validation
        cv_scores = cross_val_score(self._model, X, y, cv=5, scoring="accuracy")

        # Full training
        self._model.fit(X, y)
        self._is_trained = True

        metrics = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_samples": len(y),
            "n_features": X.shape[1],
        }

        if save_path:
            self._model.save_model(str(save_path))
            logger.info("Saved trained model to %s", save_path)

        logger.info(
            "XGBoost trained: accuracy=%.3f (+/- %.3f) on %d samples",
            metrics["cv_accuracy_mean"],
            metrics["cv_accuracy_std"],
            metrics["n_samples"],
        )
        return metrics
