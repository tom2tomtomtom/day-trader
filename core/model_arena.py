#!/usr/bin/env python3
"""
MODEL ARENA — Trains and compares multiple ML model types.

Used by the experiment tracker to find the best model configuration
for different market conditions and feature subsets.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

# LightGBM with graceful fallback
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelArena:
    """Trains and compares multiple ML model types using walk-forward validation.

    Supports LightGBM, Random Forest, Gradient Boosting, and SVM classifiers.
    Each model is evaluated with time-series-aware cross-validation and ranked
    by mean walk-forward accuracy.
    """

    # Registry mapping model type names to their training methods
    _MODEL_REGISTRY = [
        "lightgbm",
        "random_forest",
        "gradient_boosting",
        "svm",
    ]

    def __init__(self):
        self._train_methods: Dict[str, callable] = {
            "lightgbm": self._train_lightgbm,
            "random_forest": self._train_random_forest,
            "gradient_boosting": self._train_gradient_boosting,
            "svm": self._train_svm,
        }

    def train_and_compare(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_types: List[str] = None,
    ) -> Dict:
        """Train all requested model types and compare via walk-forward validation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target array of shape (n_samples,).
            feature_names: Names corresponding to each feature column.
            model_types: Which models to train. Defaults to all available.

        Returns:
            Dict with keys:
                - rankings: list of dicts sorted by score (best first)
                - best_model_type: str name of the winning model
                - best_score: float best mean walk-forward accuracy
                - models: dict of model_type -> trained model object
                - feature_importance: dict of model_type -> importance dict
        """
        empty_result = {
            "rankings": [],
            "best_model_type": None,
            "best_score": 0.0,
            "models": {},
            "feature_importance": {},
        }

        if X.shape[0] < 10:
            logger.warning(
                f"Only {X.shape[0]} samples provided — need at least 10 for arena comparison"
            )
            return empty_result

        # Determine which models to train
        if model_types is None:
            model_types = [
                mt for mt in self._MODEL_REGISTRY
                if mt != "lightgbm" or HAS_LIGHTGBM
            ]

        # Scale features once for all models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = []
        trained_models: Dict[str, object] = {}
        importance_map: Dict[str, Dict[str, float]] = {}

        for model_type in model_types:
            if model_type not in self._train_methods:
                logger.warning(f"Unknown model type '{model_type}' — skipping")
                continue

            if model_type == "lightgbm" and not HAS_LIGHTGBM:
                logger.info("LightGBM not installed — skipping")
                continue

            logger.info(f"Arena: training {model_type}...")

            try:
                # Train on full scaled data
                model = self._train_methods[model_type](X_scaled, y)
                if model is None:
                    logger.warning(f"Training {model_type} returned None — skipping")
                    continue

                # Walk-forward evaluation
                eval_result = self._walk_forward_evaluate(model, X_scaled, y)
                if not eval_result["fold_results"]:
                    logger.warning(
                        f"Walk-forward evaluation for {model_type} produced no folds — skipping"
                    )
                    continue

                # Re-train on full dataset for the final model
                final_model = self._train_methods[model_type](X_scaled, y)
                if final_model is None:
                    continue

                trained_models[model_type] = final_model

                # Extract feature importance
                importance = self._extract_feature_importance(
                    final_model, model_type, feature_names
                )
                importance_map[model_type] = importance

                results.append({
                    "model_type": model_type,
                    "mean_accuracy": eval_result["mean_accuracy"],
                    "mean_f1": eval_result["mean_f1"],
                    "std_accuracy": eval_result["std_accuracy"],
                    "n_folds": len(eval_result["fold_results"]),
                    "fold_results": eval_result["fold_results"],
                })

                logger.info(
                    f"Arena: {model_type} — "
                    f"accuracy={eval_result['mean_accuracy']:.4f} "
                    f"(+/-{eval_result['std_accuracy']:.4f}) "
                    f"f1={eval_result['mean_f1']:.4f}"
                )

            except Exception as e:
                logger.error(f"Arena: {model_type} failed — {e}", exc_info=True)
                continue

        if not results:
            logger.warning("Arena: no models trained successfully")
            return empty_result

        # Rank by mean walk-forward accuracy (descending)
        rankings = sorted(results, key=lambda r: r["mean_accuracy"], reverse=True)

        best = rankings[0]
        logger.info(
            f"Arena winner: {best['model_type']} with accuracy={best['mean_accuracy']:.4f}"
        )

        return {
            "rankings": rankings,
            "best_model_type": best["model_type"],
            "best_score": best["mean_accuracy"],
            "models": trained_models,
            "feature_importance": importance_map,
        }

    def _train_lightgbm(
        self, X: np.ndarray, y: np.ndarray, params: Dict = None
    ) -> Optional[object]:
        """Train a LightGBM classifier.

        Args:
            X: Scaled feature matrix.
            y: Binary target array.
            params: Override default hyperparameters.

        Returns:
            Trained LGBMClassifier, or None if LightGBM is unavailable or training fails.
        """
        if not HAS_LIGHTGBM:
            return None

        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            defaults.update(params)

        try:
            model = LGBMClassifier(**defaults)
            model.fit(X, y)
            return model
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return None

    def _train_random_forest(
        self, X: np.ndarray, y: np.ndarray, params: Dict = None
    ) -> Optional[object]:
        """Train a Random Forest classifier.

        Args:
            X: Scaled feature matrix.
            y: Binary target array.
            params: Override default hyperparameters.

        Returns:
            Trained RandomForestClassifier, or None if training fails.
        """
        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_leaf": 5,
            "random_state": 42,
        }
        if params:
            defaults.update(params)

        try:
            model = RandomForestClassifier(**defaults)
            model.fit(X, y)
            return model
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return None

    def _train_gradient_boosting(
        self, X: np.ndarray, y: np.ndarray, params: Dict = None
    ) -> Optional[object]:
        """Train a Gradient Boosting classifier.

        Args:
            X: Scaled feature matrix.
            y: Binary target array.
            params: Override default hyperparameters.

        Returns:
            Trained GradientBoostingClassifier, or None if training fails.
        """
        defaults = {
            "n_estimators": 150,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "subsample": 0.8,
            "random_state": 42,
        }
        if params:
            defaults.update(params)

        try:
            model = GradientBoostingClassifier(**defaults)
            model.fit(X, y)
            return model
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
            return None

    def _train_svm(
        self, X: np.ndarray, y: np.ndarray, params: Dict = None
    ) -> Optional[object]:
        """Train a Support Vector Machine classifier with probability estimates.

        Args:
            X: Scaled feature matrix.
            y: Binary target array.
            params: Override default hyperparameters.

        Returns:
            Trained SVC, or None if training fails.
        """
        defaults = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        }
        if params:
            defaults.update(params)

        try:
            model = SVC(**defaults)
            model.fit(X, y)
            return model
        except Exception as e:
            logger.error(f"SVM training failed: {e}")
            return None

    def _walk_forward_evaluate(
        self, model, X: np.ndarray, y: np.ndarray, n_folds: int = 5
    ) -> Dict:
        """Evaluate a model using walk-forward (expanding window) cross-validation.

        Uses sklearn TimeSeriesSplit to respect temporal ordering. Each fold trains
        on all data up to the split point and validates on the subsequent block.

        Args:
            model: A sklearn-compatible classifier (must support fit/predict).
            X: Scaled feature matrix.
            y: Binary target array.
            n_folds: Number of walk-forward folds. Reduced if dataset is too small.

        Returns:
            Dict with keys:
                - mean_accuracy: float
                - mean_f1: float
                - std_accuracy: float
                - fold_results: list of per-fold dicts with accuracy, f1, n_train, n_val
        """
        n_samples = len(X)

        # Need at least 2 samples per fold (1 train, 1 val minimum)
        max_folds = n_samples // 2
        actual_folds = min(n_folds, max_folds)
        if actual_folds < 2:
            actual_folds = 2

        # TimeSeriesSplit needs n_splits >= 2 and enough data
        if n_samples < actual_folds + 1:
            logger.warning(
                f"Not enough samples ({n_samples}) for {actual_folds} folds — "
                f"reducing to {max(2, n_samples - 1)} folds"
            )
            actual_folds = max(2, n_samples - 1)

        tscv = TimeSeriesSplit(n_splits=actual_folds)

        fold_results = []
        accuracies = []
        f1_scores_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Clone the model type and train on this fold's training data
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)

                y_pred = fold_model.predict(X_val)

                acc = float(accuracy_score(y_val, y_pred))
                f1 = float(f1_score(y_val, y_pred, zero_division=0))

                accuracies.append(acc)
                f1_scores_list.append(f1)

                fold_results.append({
                    "fold": fold_idx + 1,
                    "accuracy": acc,
                    "f1": f1,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                })
            except Exception as e:
                logger.warning(f"Walk-forward fold {fold_idx + 1} failed: {e}")
                continue

        if not accuracies:
            return {
                "mean_accuracy": 0.0,
                "mean_f1": 0.0,
                "std_accuracy": 0.0,
                "fold_results": [],
            }

        return {
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_f1": float(np.mean(f1_scores_list)),
            "std_accuracy": float(np.std(accuracies)),
            "fold_results": fold_results,
        }

    def _extract_feature_importance(
        self, model, model_type: str, feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract and normalize feature importance from a trained model.

        LightGBM, Random Forest, and Gradient Boosting expose .feature_importances_.
        SVM uses permutation importance as a fallback (returns empty dict if unavailable).
        All importances are normalized to sum to 1.0.

        Args:
            model: Trained sklearn-compatible model.
            model_type: One of "lightgbm", "random_forest", "gradient_boosting", "svm".
            feature_names: Feature names matching the columns of X.

        Returns:
            Dict mapping feature name to normalized importance (sums to 1.0).
        """
        try:
            if model_type in ("lightgbm", "random_forest", "gradient_boosting"):
                raw_importances = model.feature_importances_

                total = float(np.sum(raw_importances))
                if total == 0:
                    return {name: 0.0 for name in feature_names}

                normalized = raw_importances / total
                importance_dict = {
                    name: round(float(imp), 6)
                    for name, imp in zip(feature_names, normalized)
                }
                # Sort by importance descending
                return dict(
                    sorted(importance_dict.items(), key=lambda x: -x[1])
                )

            elif model_type == "svm":
                # SVM does not have .feature_importances_
                # Use permutation importance if dataset is reasonably sized
                try:
                    from sklearn.inspection import permutation_importance

                    # Only compute if we can access training data via a quick check
                    # In practice, we'd need X and y — but since we don't store them
                    # on the model, return empty. The caller can compute this externally.
                    logger.debug(
                        "SVM feature importance requires permutation_importance "
                        "with held-out data — returning empty dict"
                    )
                    return {}
                except ImportError:
                    return {}

            else:
                logger.warning(f"Unknown model type for importance extraction: {model_type}")
                return {}

        except Exception as e:
            logger.error(f"Feature importance extraction failed for {model_type}: {e}")
            return {}
