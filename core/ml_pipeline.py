#!/usr/bin/env python3
"""
ML PIPELINE — Gradient boosting models for trade quality prediction.

Two models:
1. Signal model: GradientBoostingClassifier predicting profitable/not-profitable
2. Sizing model: GradientBoostingRegressor predicting optimal position fraction

Uses TimeSeriesSplit cross-validation (no future leak).
Falls back to rule-based when <30 trades in DB.
"""

import logging
import json
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from .config import get_config
from .db import get_db
from .feature_engine import FeatureVector

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class MLPrediction:
    """Output from ML pipeline."""
    quality_score: float      # 0-1, probability of profitable trade
    size_multiplier: float    # 0-2, multiplier for position size
    confidence: float         # 0-1, model confidence
    model_version: int
    using_ml: bool            # False if falling back to rule-based


class MLPipeline:
    """Gradient boosting pipeline for trade quality and sizing prediction."""

    def __init__(self):
        self.signal_model: Optional[GradientBoostingClassifier] = None
        self.sizing_model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_version: int = 0
        self.feature_importance: Dict[str, float] = {}
        self._load_models()

    def predict(self, features: FeatureVector) -> MLPrediction:
        """Predict trade quality and optimal sizing for given features."""
        if self.signal_model is None or self.scaler is None:
            return self._rule_based_prediction(features)

        try:
            X = features.to_ml_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Quality prediction (probability of profitable trade)
            quality_proba = self.signal_model.predict_proba(X_scaled)[0]
            # Index 1 is probability of class "profitable"
            quality_score = float(quality_proba[1]) if len(quality_proba) > 1 else float(quality_proba[0])

            # Sizing prediction
            if self.sizing_model is not None:
                size_mult = float(self.sizing_model.predict(X_scaled)[0])
                size_mult = max(0.1, min(2.0, size_mult))
            else:
                size_mult = quality_score  # Use quality as size proxy

            confidence = abs(quality_score - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%

            return MLPrediction(
                quality_score=round(quality_score, 4),
                size_multiplier=round(size_mult, 4),
                confidence=round(confidence, 4),
                model_version=self.model_version,
                using_ml=True,
            )
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._rule_based_prediction(features)

    def train(self, force: bool = False) -> bool:
        """Train models from trade history in DB.

        Returns True if training succeeded.
        """
        db = get_db()
        cfg = get_config()

        trades = db.get_trades_with_features(limit=1000)
        if len(trades) < cfg.trading.min_trades_for_ml and not force:
            logger.info(f"Only {len(trades)} trades with features — need {cfg.trading.min_trades_for_ml} for ML")
            return False

        if not trades:
            return False

        # Build training data
        X_list = []
        y_quality = []  # 1=profitable, 0=not
        y_sizing = []   # optimal position fraction (based on actual PnL)

        for trade in trades:
            features_raw = trade.get("entry_features", "{}")
            if isinstance(features_raw, str):
                features_dict = json.loads(features_raw)
            else:
                features_dict = features_raw

            if not features_dict:
                continue

            # Reconstruct feature vector
            try:
                fv = FeatureVector(
                    symbol=trade["symbol"],
                    timestamp=trade.get("entry_date", ""),
                    **{k: v for k, v in features_dict.items()
                       if k in FeatureVector.__dataclass_fields__ and k not in ("symbol", "timestamp")}
                )
                X_list.append(fv.to_ml_array())
            except Exception:
                continue

            pnl_pct = float(trade.get("pnl_pct", 0))
            y_quality.append(1 if pnl_pct > 0 else 0)

            # Sizing target: normalize PnL to 0-2 range
            # Profitable trades > 5% get 2.0, breakeven gets 1.0, big losses get 0.1
            if pnl_pct > 5:
                y_sizing.append(2.0)
            elif pnl_pct > 0:
                y_sizing.append(1.0 + pnl_pct / 5)
            elif pnl_pct > -5:
                y_sizing.append(max(0.1, 1.0 + pnl_pct / 10))
            else:
                y_sizing.append(0.1)

        if len(X_list) < 10:
            logger.warning(f"Only {len(X_list)} valid training samples — skipping")
            return False

        X = np.array(X_list)
        y_q = np.array(y_quality)
        y_s = np.array(y_sizing)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # TimeSeriesSplit cross-validation
        n_splits = min(5, len(X) // 10)
        if n_splits < 2:
            n_splits = 2
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Train signal model (classifier)
        self.signal_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )

        # Cross-validate
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_q[train_idx], y_q[val_idx]
            self.signal_model.fit(X_train, y_train)
            cv_scores.append(self.signal_model.score(X_val, y_val))

        # Final fit on all data
        self.signal_model.fit(X_scaled, y_q)
        y_pred = self.signal_model.predict(X_scaled)

        metrics = {
            "accuracy": float(accuracy_score(y_q, y_pred)),
            "precision": float(precision_score(y_q, y_pred, zero_division=0)),
            "recall": float(recall_score(y_q, y_pred, zero_division=0)),
            "f1": float(f1_score(y_q, y_pred, zero_division=0)),
            "cv_mean": float(np.mean(cv_scores)),
        }

        # Feature importance
        feature_names = FeatureVector.feature_names()
        importances = self.signal_model.feature_importances_
        self.feature_importance = {
            name: round(float(imp), 4)
            for name, imp in sorted(zip(feature_names, importances),
                                     key=lambda x: -x[1])
        }

        # Train sizing model (regressor)
        self.sizing_model = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        self.sizing_model.fit(X_scaled, y_s)

        # Save models
        self.model_version += 1
        self._save_models()

        # Log to DB
        db.save_ml_model({
            "model_name": "signal_quality",
            "model_type": "gradient_boosting",
            "version": self.model_version,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "feature_importance": self.feature_importance,
            "training_samples": len(X),
            "is_active": True,
        })

        logger.info(
            f"ML models trained v{self.model_version}: "
            f"accuracy={metrics['accuracy']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"cv={metrics['cv_mean']:.3f} "
            f"samples={len(X)}"
        )
        return True

    def needs_retrain(self) -> bool:
        """Check if models should be retrained."""
        db = get_db()
        model_info = db.get_active_model("signal_quality")
        if model_info is None:
            return True

        # Retrain if we have 50%+ more trades than last training
        last_samples = model_info.get("training_samples", 0)
        current_trades = len(db.get_trades_with_features(limit=last_samples * 2))
        return current_trades > last_samples * 1.5

    def _rule_based_prediction(self, features: FeatureVector) -> MLPrediction:
        """Fallback when ML model is not available."""
        score = abs(features.composite_score) / 100
        # Boost for strong signals, penalize weak ones
        quality = 0.5 + (score - 0.25) * 2 if score > 0.25 else 0.3
        quality = max(0.1, min(0.95, quality))

        return MLPrediction(
            quality_score=round(quality, 4),
            size_multiplier=round(quality, 4),
            confidence=round(score, 4),
            model_version=0,
            using_ml=False,
        )

    def _save_models(self):
        """Save trained models to disk."""
        try:
            if self.signal_model:
                joblib.dump(self.signal_model, MODELS_DIR / "signal_quality.joblib")
            if self.sizing_model:
                joblib.dump(self.sizing_model, MODELS_DIR / "sizing_model.joblib")
            if self.scaler:
                joblib.dump(self.scaler, MODELS_DIR / "feature_scaler.joblib")
            # Save version
            meta = {
                "version": self.model_version,
                "feature_importance": self.feature_importance,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            (MODELS_DIR / "ml_meta.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            signal_path = MODELS_DIR / "signal_quality.joblib"
            sizing_path = MODELS_DIR / "sizing_model.joblib"
            scaler_path = MODELS_DIR / "feature_scaler.joblib"
            meta_path = MODELS_DIR / "ml_meta.json"

            if signal_path.exists() and scaler_path.exists():
                self.signal_model = joblib.load(signal_path)
                self.scaler = joblib.load(scaler_path)
                if sizing_path.exists():
                    self.sizing_model = joblib.load(sizing_path)
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self.model_version = meta.get("version", 1)
                    self.feature_importance = meta.get("feature_importance", {})
                logger.info(f"Loaded ML models v{self.model_version}")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")


# Singleton
_pipeline: Optional[MLPipeline] = None


def get_ml_pipeline() -> MLPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MLPipeline()
    return _pipeline
