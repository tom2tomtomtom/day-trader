"""Tests for core/model_arena.py"""

import numpy as np
import pytest
from unittest.mock import patch


class TestModelArena:
    """Test the multi-model comparison framework."""

    def _make_data(self, n=50, n_features=10):
        """Create synthetic classification data."""
        np.random.seed(42)
        X = np.random.randn(n, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        return X, y, feature_names

    def test_train_and_compare_basic(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, names = self._make_data(n=60)

        results = arena.train_and_compare(X, y, names)

        assert results is not None
        assert "rankings" in results
        assert "best_model_type" in results
        assert "best_score" in results
        assert len(results["rankings"]) > 0

    def test_train_and_compare_too_few_samples(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, names = self._make_data(n=5)

        results = arena.train_and_compare(X, y, names)

        assert results is not None
        assert len(results.get("rankings", [])) == 0

    def test_train_random_forest(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, _ = self._make_data(n=40)

        model = arena._train_random_forest(X, y)
        assert model is not None
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_train_gradient_boosting(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, _ = self._make_data(n=40)

        model = arena._train_gradient_boosting(X, y)
        assert model is not None
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_train_svm(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, _ = self._make_data(n=40)

        model = arena._train_svm(X, y)
        assert model is not None
        probas = model.predict_proba(X[:5])
        assert probas.shape == (5, 2)

    def test_walk_forward_evaluate(self):
        from core.model_arena import ModelArena
        from sklearn.ensemble import RandomForestClassifier
        arena = ModelArena()
        X, y, _ = self._make_data(n=60)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        metrics = arena._walk_forward_evaluate(model, X, y, n_folds=3)

        assert "mean_accuracy" in metrics
        assert "mean_f1" in metrics
        assert "fold_results" in metrics
        assert metrics["mean_accuracy"] > 0
        assert len(metrics["fold_results"]) == 3

    def test_extract_feature_importance_rf(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, names = self._make_data(n=40, n_features=5)

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance = arena._extract_feature_importance(model, "random_forest", names)
        assert len(importance) == 5
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_specific_model_types(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, names = self._make_data(n=60)

        results = arena.train_and_compare(
            X, y, names, model_types=["random_forest", "svm"]
        )

        assert len(results["rankings"]) <= 2
        model_types = [r["model_type"] for r in results["rankings"]]
        for mt in model_types:
            assert mt in ["random_forest", "svm"]

    def test_feature_importance_in_results(self):
        from core.model_arena import ModelArena
        arena = ModelArena()
        X, y, names = self._make_data(n=60)

        results = arena.train_and_compare(
            X, y, names, model_types=["random_forest"]
        )

        assert "feature_importance" in results
        if "random_forest" in results["feature_importance"]:
            fi = results["feature_importance"]["random_forest"]
            assert len(fi) > 0
