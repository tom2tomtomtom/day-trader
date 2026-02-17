"""
Ensemble Pricing Model - Layer 3 Coordinator.

Combines predictions from the statistical base model, XGBoost model,
and (future) LSTM model into a single probability estimate with
confidence scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cricket.config import ModelConfig
from cricket.models.statistical import StatisticalModel, StatisticalPrediction
from cricket.models.xgboost_model import CricketXGBoostModel, XGBoostPrediction

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Combined prediction from all models."""

    team_a_win_prob: float
    team_b_win_prob: float
    draw_prob: float = 0.0

    # Per-model probabilities for transparency
    statistical_prob: float = 0.5
    xgboost_prob: float = 0.5
    lstm_prob: float = 0.5  # Placeholder for future LSTM

    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    confidence_score: float = 0.5
    model_agreement: float = 0.0  # Max spread between models

    @property
    def team_a_fair_odds(self) -> float:
        """Fair decimal odds for team A."""
        return 1.0 / self.team_a_win_prob if self.team_a_win_prob > 0.001 else 999.0

    @property
    def team_b_fair_odds(self) -> float:
        """Fair decimal odds for team B."""
        return 1.0 / self.team_b_win_prob if self.team_b_win_prob > 0.001 else 999.0

    @property
    def has_edge(self) -> bool:
        """Whether the prediction has meaningful confidence."""
        return self.confidence in ("HIGH", "MEDIUM")


class EnsemblePricingModel:
    """Coordinates the ensemble of pricing models.

    Combines:
    - Model A: Statistical Base Model (30% weight)
    - Model B: XGBoost Model (45% weight)
    - Model C: LSTM Model (25% weight) - placeholder for future

    When LSTM is not available, weights are renormalized between
    the statistical and XGBoost models.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        match_format: str = "t20",
        xgboost_model_path: Optional[Path] = None,
    ):
        self._config = config or ModelConfig()
        self._format = match_format

        # Initialize models
        self._statistical = StatisticalModel(format_type=match_format)
        self._xgboost = CricketXGBoostModel(model_path=xgboost_model_path)

        # LSTM placeholder - will be implemented later
        self._lstm_available = False

        # Calculate effective weights
        self._recalculate_weights()

    def _recalculate_weights(self) -> None:
        """Recalculate model weights based on availability."""
        stat_w = self._config.statistical_weight
        xgb_w = self._config.xgboost_weight
        lstm_w = self._config.lstm_weight

        if not self._lstm_available:
            # Redistribute LSTM weight proportionally
            total = stat_w + xgb_w
            self._stat_weight = stat_w / total
            self._xgb_weight = xgb_w / total
            self._lstm_weight = 0.0
        else:
            self._stat_weight = stat_w
            self._xgb_weight = xgb_w
            self._lstm_weight = lstm_w

        logger.info(
            "Ensemble weights: stat=%.2f, xgb=%.2f, lstm=%.2f",
            self._stat_weight, self._xgb_weight, self._lstm_weight,
        )

    def predict(
        self,
        features: dict[str, float],
        batting_is_team_a: bool = True,
    ) -> EnsemblePrediction:
        """Generate ensemble prediction from current match state.

        Args:
            features: Feature dict from MatchStateEngine.get_features()
            batting_is_team_a: Whether the batting team is team A

        Returns:
            EnsemblePrediction with combined probabilities
        """
        # Model A: Statistical
        stat_pred = self._statistical.predict(features, batting_is_team_a)

        # Model B: XGBoost
        xgb_pred = self._xgboost.predict(features, batting_is_team_a)

        # Model C: LSTM (placeholder - uses XGBoost prediction for now)
        lstm_prob_a = xgb_pred.team_a_win_prob

        # Weighted ensemble
        prob_a = (
            self._stat_weight * stat_pred.team_a_win_prob
            + self._xgb_weight * xgb_pred.team_a_win_prob
            + self._lstm_weight * lstm_prob_a
        )
        prob_a = max(0.01, min(0.99, prob_a))
        prob_b = 1.0 - prob_a

        # Model agreement (max spread)
        probs = [stat_pred.team_a_win_prob, xgb_pred.team_a_win_prob]
        if self._lstm_available:
            probs.append(lstm_prob_a)
        model_spread = max(probs) - min(probs)

        # Confidence classification
        if model_spread <= self._config.confidence_high_threshold:
            confidence = "HIGH"
            confidence_score = 0.9
        elif model_spread <= self._config.confidence_medium_threshold:
            confidence = "MEDIUM"
            confidence_score = 0.6
        else:
            confidence = "LOW"
            confidence_score = 0.3

        # Weight by individual model confidences too
        avg_model_conf = (
            stat_pred.confidence * self._stat_weight
            + xgb_pred.confidence * self._xgb_weight
        )
        confidence_score = confidence_score * 0.6 + avg_model_conf * 0.4

        return EnsemblePrediction(
            team_a_win_prob=prob_a,
            team_b_win_prob=prob_b,
            statistical_prob=stat_pred.team_a_win_prob,
            xgboost_prob=xgb_pred.team_a_win_prob,
            lstm_prob=lstm_prob_a,
            confidence=confidence,
            confidence_score=confidence_score,
            model_agreement=model_spread,
        )

    @property
    def statistical_model(self) -> StatisticalModel:
        return self._statistical

    @property
    def xgboost_model(self) -> CricketXGBoostModel:
        return self._xgboost
