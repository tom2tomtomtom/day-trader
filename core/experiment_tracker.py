#!/usr/bin/env python3
"""
EXPERIMENT TRACKER — Designs and runs statistical experiments.

Validates hypotheses from the hypothesis engine using small-sample-safe
methods: bootstrap comparison, permutation tests, walk-forward validation,
and feature ablation studies.

Small-sample safeguards:
- Min 5 trades per group
- Relaxed alpha = 0.10
- Effect size filter d > 0.3
- Benjamini-Hochberg FDR correction at q = 0.15
- Confidence labeling (high/medium/low/inconclusive)
"""

import json
import logging
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .db import get_db
from .feature_engine import FeatureVector

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
MIN_GROUP_SIZE = 5
ALPHA = 0.10
MIN_EFFECT_SIZE = 0.3
FDR_Q = 0.15
BOOTSTRAP_ITERATIONS = 1000
PERMUTATION_ITERATIONS = 1000

# Maps hypothesis category -> experiment type
EXPERIMENT_TYPE_MAP = {
    "regime_conditional": "bootstrap_comparison",
    "feature_interaction": "bootstrap_comparison",
    "temporal_pattern": "permutation_test",
    "feature_drift": "feature_ablation",
    "signal_accuracy": "walk_forward",
    "exit_reason": "bootstrap_comparison",
    "feature_threshold": "bootstrap_comparison",
}


@dataclass
class ExperimentDesign:
    """Blueprint for an experiment before execution."""
    experiment_id: str
    hypothesis_id: str
    name: str
    experiment_type: str
    independent_variable: str
    dependent_variable: str = "pnl_pct"
    control_definition: Dict = field(default_factory=dict)
    treatment_definition: Dict = field(default_factory=dict)
    min_sample_size: int = MIN_GROUP_SIZE * 2
    significance_level: float = ALPHA


@dataclass
class GroupMetrics:
    """Metrics for a group of trades."""
    n: int = 0
    mean_pnl: float = 0.0
    median_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    total_pnl: float = 0.0


@dataclass
class ExperimentResult:
    """Full result from a completed experiment."""
    experiment_id: str
    hypothesis_id: str
    name: str
    experiment_type: str
    status: str = "completed"
    control_metrics: Dict = field(default_factory=dict)
    treatment_metrics: Dict = field(default_factory=dict)
    effect_size: float = 0.0
    p_value: float = 1.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    is_significant: bool = False
    confidence_level: str = "inconclusive"
    model_results: Dict = field(default_factory=dict)
    best_model_type: str = ""
    runtime_seconds: float = 0.0
    narrative: str = ""


class ExperimentTracker:
    """Designs and runs statistical experiments to validate hypotheses."""

    def __init__(self):
        self.db = get_db()

    def design_experiment(self, hypothesis) -> Optional[ExperimentDesign]:
        """Map a hypothesis to an experiment design."""
        try:
            exp_type = EXPERIMENT_TYPE_MAP.get(
                hypothesis.category, "bootstrap_comparison"
            )

            design = ExperimentDesign(
                experiment_id=f"exp_{uuid.uuid4().hex[:12]}",
                hypothesis_id=hypothesis.id,
                name=f"Test: {hypothesis.statement[:80]}",
                experiment_type=exp_type,
                independent_variable=hypothesis.source,
                control_definition=hypothesis.supporting_evidence.get(
                    "control", {"group": "baseline"}
                ),
                treatment_definition=hypothesis.supporting_evidence.get(
                    "treatment", {"group": "treatment"}
                ),
            )
            return design
        except Exception as e:
            logger.error(f"Failed to design experiment: {e}")
            return None

    def run_experiment(
        self, design: ExperimentDesign, trades: List[Dict]
    ) -> ExperimentResult:
        """Execute an experiment and return full metrics."""
        t0 = time.time()

        result = ExperimentResult(
            experiment_id=design.experiment_id,
            hypothesis_id=design.hypothesis_id,
            name=design.name,
            experiment_type=design.experiment_type,
        )

        try:
            if design.experiment_type == "bootstrap_comparison":
                result = self._run_bootstrap_comparison(design, trades, result)
            elif design.experiment_type == "permutation_test":
                result = self._run_permutation_test(design, trades, result)
            elif design.experiment_type == "walk_forward":
                result = self._run_walk_forward(design, trades, result)
            elif design.experiment_type == "feature_ablation":
                result = self._run_feature_ablation(design, trades, result)
            else:
                result.status = "failed"
                result.narrative = f"Unknown experiment type: {design.experiment_type}"

            # Assign confidence level
            result.confidence_level = self._confidence_label(result)

        except Exception as e:
            logger.error(f"Experiment {design.experiment_id} failed: {e}")
            result.status = "failed"
            result.narrative = f"Experiment failed: {str(e)}"

        result.runtime_seconds = round(time.time() - t0, 2)

        # Persist to DB
        self._save_result(design, result)

        return result

    # ── Experiment Types ───────────────────────────────────────────

    def _run_bootstrap_comparison(
        self,
        design: ExperimentDesign,
        trades: List[Dict],
        result: ExperimentResult,
    ) -> ExperimentResult:
        """Split trades by condition, bootstrap PnL difference."""
        control_trades = self._select_trades(trades, design.control_definition)
        treatment_trades = self._select_trades(trades, design.treatment_definition)

        if len(control_trades) < MIN_GROUP_SIZE or len(treatment_trades) < MIN_GROUP_SIZE:
            result.status = "insufficient_data"
            result.narrative = (
                f"Not enough trades: control={len(control_trades)}, "
                f"treatment={len(treatment_trades)} (need {MIN_GROUP_SIZE} each)"
            )
            return result

        control_pnl = np.array([float(t.get("pnl_pct", 0)) for t in control_trades])
        treatment_pnl = np.array([float(t.get("pnl_pct", 0)) for t in treatment_trades])

        # Bootstrap mean difference
        observed_diff = float(np.mean(treatment_pnl) - np.mean(control_pnl))
        boot_diffs = []
        rng = np.random.default_rng(42)
        for _ in range(BOOTSTRAP_ITERATIONS):
            c_sample = rng.choice(control_pnl, size=len(control_pnl), replace=True)
            t_sample = rng.choice(treatment_pnl, size=len(treatment_pnl), replace=True)
            boot_diffs.append(float(np.mean(t_sample) - np.mean(c_sample)))

        boot_diffs = np.array(boot_diffs)
        ci_lower = float(np.percentile(boot_diffs, 5))
        ci_upper = float(np.percentile(boot_diffs, 95))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_pnl) * (len(control_pnl) - 1)
             + np.var(treatment_pnl) * (len(treatment_pnl) - 1))
            / (len(control_pnl) + len(treatment_pnl) - 2)
        )
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0

        # P-value: proportion of bootstrap samples crossing zero
        if observed_diff > 0:
            p_value = float(np.mean(boot_diffs <= 0))
        else:
            p_value = float(np.mean(boot_diffs >= 0))

        is_significant = p_value < ALPHA and abs(effect_size) >= MIN_EFFECT_SIZE

        result.control_metrics = asdict(self._compute_group_metrics(control_trades))
        result.treatment_metrics = asdict(self._compute_group_metrics(treatment_trades))
        result.effect_size = round(effect_size, 4)
        result.p_value = round(p_value, 4)
        result.ci_lower = round(ci_lower, 4)
        result.ci_upper = round(ci_upper, 4)
        result.is_significant = is_significant

        direction = "higher" if observed_diff > 0 else "lower"
        result.narrative = (
            f"Treatment group shows {direction} PnL (diff={observed_diff:.2f}%, "
            f"d={effect_size:.2f}, p={p_value:.3f}, 90% CI=[{ci_lower:.2f}, {ci_upper:.2f}]). "
            f"{'Statistically significant.' if is_significant else 'Not statistically significant.'}"
        )
        return result

    def _run_permutation_test(
        self,
        design: ExperimentDesign,
        trades: List[Dict],
        result: ExperimentResult,
    ) -> ExperimentResult:
        """Shuffle group labels N times to build null distribution."""
        control_trades = self._select_trades(trades, design.control_definition)
        treatment_trades = self._select_trades(trades, design.treatment_definition)

        if len(control_trades) < MIN_GROUP_SIZE or len(treatment_trades) < MIN_GROUP_SIZE:
            result.status = "insufficient_data"
            result.narrative = (
                f"Not enough trades: control={len(control_trades)}, "
                f"treatment={len(treatment_trades)}"
            )
            return result

        control_pnl = np.array([float(t.get("pnl_pct", 0)) for t in control_trades])
        treatment_pnl = np.array([float(t.get("pnl_pct", 0)) for t in treatment_trades])

        observed_diff = float(np.mean(treatment_pnl) - np.mean(control_pnl))
        combined = np.concatenate([control_pnl, treatment_pnl])
        n_control = len(control_pnl)

        rng = np.random.default_rng(42)
        null_diffs = []
        for _ in range(PERMUTATION_ITERATIONS):
            perm = rng.permutation(combined)
            perm_control = perm[:n_control]
            perm_treatment = perm[n_control:]
            null_diffs.append(float(np.mean(perm_treatment) - np.mean(perm_control)))

        null_diffs = np.array(null_diffs)

        # Two-sided p-value
        p_value = float(np.mean(np.abs(null_diffs) >= abs(observed_diff)))

        # Effect size
        pooled_std = np.sqrt(
            (np.var(control_pnl) * (len(control_pnl) - 1)
             + np.var(treatment_pnl) * (len(treatment_pnl) - 1))
            / (len(control_pnl) + len(treatment_pnl) - 2)
        )
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        is_significant = p_value < ALPHA and abs(effect_size) >= MIN_EFFECT_SIZE

        result.control_metrics = asdict(self._compute_group_metrics(control_trades))
        result.treatment_metrics = asdict(self._compute_group_metrics(treatment_trades))
        result.effect_size = round(effect_size, 4)
        result.p_value = round(p_value, 4)
        result.is_significant = is_significant

        result.narrative = (
            f"Permutation test: observed diff={observed_diff:.2f}%, "
            f"d={effect_size:.2f}, p={p_value:.3f}. "
            f"{'Significant pattern.' if is_significant else 'No significant pattern.'}"
        )
        return result

    def _run_walk_forward(
        self,
        design: ExperimentDesign,
        trades: List[Dict],
        result: ExperimentResult,
    ) -> ExperimentResult:
        """Train 70% / test 30% with feature under test via model arena."""
        if len(trades) < 20:
            result.status = "insufficient_data"
            result.narrative = f"Need 20+ trades for walk-forward, have {len(trades)}"
            return result

        try:
            from .model_arena import ModelArena

            # Build feature matrix from trades
            X_list, y_list = [], []
            feature_names = FeatureVector.feature_names()

            for trade in trades:
                features_raw = trade.get("entry_features", "{}")
                if isinstance(features_raw, str):
                    features_dict = json.loads(features_raw)
                else:
                    features_dict = features_raw

                if not features_dict:
                    continue

                try:
                    fv = FeatureVector(
                        symbol=trade["symbol"],
                        timestamp=trade.get("entry_date", ""),
                        **{k: v for k, v in features_dict.items()
                           if k in FeatureVector.__dataclass_fields__
                           and k not in ("symbol", "timestamp")}
                    )
                    X_list.append(fv.to_ml_array())
                    y_list.append(1 if float(trade.get("pnl_pct", 0)) > 0 else 0)
                except Exception:
                    continue

            if len(X_list) < 20:
                result.status = "insufficient_data"
                result.narrative = f"Only {len(X_list)} valid samples for walk-forward"
                return result

            X = np.array(X_list)
            y = np.array(y_list)

            arena = ModelArena()
            arena_results = arena.train_and_compare(X, y, feature_names)

            if arena_results and arena_results.get("rankings"):
                result.model_results = {
                    "rankings": arena_results["rankings"],
                }
                result.best_model_type = arena_results.get("best_model_type", "")
                best_score = arena_results.get("best_score", 0)

                # Compare to baseline (random = 0.5)
                improvement = best_score - 0.5
                result.effect_size = round(improvement * 2, 4)  # Scale to d-like metric
                result.is_significant = best_score > 0.55
                result.p_value = round(max(0.01, 1.0 - best_score), 4)

                result.narrative = (
                    f"Walk-forward: best model={result.best_model_type} "
                    f"(accuracy={best_score:.3f}). "
                    f"{'Above baseline.' if result.is_significant else 'At or below baseline.'}"
                )
            else:
                result.status = "failed"
                result.narrative = "Model arena returned no results"

        except ImportError:
            result.status = "failed"
            result.narrative = "ModelArena not available"
        except Exception as e:
            result.status = "failed"
            result.narrative = f"Walk-forward failed: {str(e)}"

        return result

    def _run_feature_ablation(
        self,
        design: ExperimentDesign,
        trades: List[Dict],
        result: ExperimentResult,
    ) -> ExperimentResult:
        """Retrain with/without suspected drifted feature."""
        if len(trades) < 20:
            result.status = "insufficient_data"
            result.narrative = f"Need 20+ trades for ablation, have {len(trades)}"
            return result

        try:
            from .model_arena import ModelArena

            feature_names = FeatureVector.feature_names()
            target_feature = design.independent_variable

            # Find feature index
            if target_feature not in feature_names:
                result.status = "failed"
                result.narrative = f"Feature {target_feature} not found"
                return result

            feature_idx = feature_names.index(target_feature)

            # Build feature matrix
            X_list, y_list = [], []
            for trade in trades:
                features_raw = trade.get("entry_features", "{}")
                if isinstance(features_raw, str):
                    features_dict = json.loads(features_raw)
                else:
                    features_dict = features_raw
                if not features_dict:
                    continue
                try:
                    fv = FeatureVector(
                        symbol=trade["symbol"],
                        timestamp=trade.get("entry_date", ""),
                        **{k: v for k, v in features_dict.items()
                           if k in FeatureVector.__dataclass_fields__
                           and k not in ("symbol", "timestamp")}
                    )
                    X_list.append(fv.to_ml_array())
                    y_list.append(1 if float(trade.get("pnl_pct", 0)) > 0 else 0)
                except Exception:
                    continue

            if len(X_list) < 20:
                result.status = "insufficient_data"
                return result

            X = np.array(X_list)
            y = np.array(y_list)

            arena = ModelArena()

            # With feature (full model)
            full_results = arena.train_and_compare(
                X, y, feature_names, model_types=["gradient_boosting"]
            )

            # Without feature (ablated)
            X_ablated = np.delete(X, feature_idx, axis=1)
            ablated_names = [n for i, n in enumerate(feature_names) if i != feature_idx]
            ablated_results = arena.train_and_compare(
                X_ablated, y, ablated_names, model_types=["gradient_boosting"]
            )

            full_score = full_results.get("best_score", 0) if full_results else 0
            ablated_score = ablated_results.get("best_score", 0) if ablated_results else 0

            score_diff = full_score - ablated_score
            result.effect_size = round(score_diff * 10, 4)  # Scale for readability
            result.is_significant = abs(score_diff) > 0.02  # 2% accuracy difference

            result.control_metrics = {"full_model_accuracy": round(full_score, 4)}
            result.treatment_metrics = {"ablated_model_accuracy": round(ablated_score, 4)}

            if score_diff > 0.02:
                result.narrative = (
                    f"Feature {target_feature} IS useful: removing it drops "
                    f"accuracy by {abs(score_diff)*100:.1f}%"
                )
            elif score_diff < -0.02:
                result.narrative = (
                    f"Feature {target_feature} may be HARMFUL: removing it improves "
                    f"accuracy by {abs(score_diff)*100:.1f}%"
                )
            else:
                result.narrative = (
                    f"Feature {target_feature} has negligible impact "
                    f"(accuracy diff={score_diff*100:.1f}%)"
                )

        except ImportError:
            result.status = "failed"
            result.narrative = "ModelArena not available"
        except Exception as e:
            result.status = "failed"
            result.narrative = f"Feature ablation failed: {str(e)}"

        return result

    # ── Trade Selection ────────────────────────────────────────────

    def _select_trades(
        self, trades: List[Dict], definition: Dict
    ) -> List[Dict]:
        """Filter trades by condition definition."""
        if not definition:
            return trades

        selected = trades

        # Filter by regime
        regime = definition.get("regime")
        if regime:
            selected = [t for t in selected if t.get("regime_at_entry") == regime]

        # Filter by feature condition
        feature = definition.get("feature")
        threshold = definition.get("threshold")
        operator = definition.get("operator", ">=")
        if feature and threshold is not None:
            filtered = []
            for t in selected:
                val = self._get_feature_value(t, feature)
                if val is None:
                    continue
                if operator == ">=" and val >= threshold:
                    filtered.append(t)
                elif operator == "<" and val < threshold:
                    filtered.append(t)
                elif operator == "==" and val == threshold:
                    filtered.append(t)
                elif operator == ">" and val > threshold:
                    filtered.append(t)
                elif operator == "<=" and val <= threshold:
                    filtered.append(t)
            selected = filtered

        # Filter by day of week
        day = definition.get("day_of_week")
        if day is not None:
            selected = [
                t for t in selected
                if self._get_feature_value(t, "day_of_week") == day
            ]

        # Filter by hour bucket
        hour_bucket = definition.get("hour_bucket")
        if hour_bucket:
            filtered = []
            for t in selected:
                hour = self._get_feature_value(t, "hour_of_day")
                if hour is None:
                    continue
                if hour_bucket == "morning" and 9 <= hour < 12:
                    filtered.append(t)
                elif hour_bucket == "afternoon" and 12 <= hour < 16:
                    filtered.append(t)
                elif hour_bucket == "after_hours" and hour >= 16:
                    filtered.append(t)
            selected = filtered

        # Filter by exit reason
        exit_reason = definition.get("exit_reason")
        if exit_reason:
            selected = [t for t in selected if t.get("exit_reason") == exit_reason]

        # Filter by group = "baseline" (complement of treatment)
        if definition.get("group") == "baseline":
            # Return all trades (control is full dataset)
            return trades

        # NOT filter (complement)
        not_regime = definition.get("not_regime")
        if not_regime:
            selected = [t for t in selected if t.get("regime_at_entry") != not_regime]

        not_day = definition.get("not_day_of_week")
        if not_day is not None:
            selected = [
                t for t in selected
                if self._get_feature_value(t, "day_of_week") != not_day
            ]

        return selected

    def _get_feature_value(self, trade: Dict, feature: str) -> Optional[float]:
        """Extract feature value from trade's entry_features."""
        features = trade.get("entry_features", {})
        if isinstance(features, str):
            try:
                features = json.loads(features)
            except (json.JSONDecodeError, TypeError):
                return None
        val = features.get(feature)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return None

    # ── Group Metrics ──────────────────────────────────────────────

    def _compute_group_metrics(self, trades: List[Dict]) -> GroupMetrics:
        """Compute aggregate metrics for a group of trades."""
        if not trades:
            return GroupMetrics()

        pnls = [float(t.get("pnl_pct", 0)) for t in trades]
        pnl_arr = np.array(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        return GroupMetrics(
            n=len(trades),
            mean_pnl=round(float(np.mean(pnl_arr)), 4),
            median_pnl=round(float(np.median(pnl_arr)), 4),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0,
            profit_factor=round(total_wins / total_losses, 4) if total_losses > 0 else float('inf') if total_wins > 0 else 0,
            sharpe=round(float(np.mean(pnl_arr) / np.std(pnl_arr)), 4) if np.std(pnl_arr) > 0 else 0,
            total_pnl=round(float(np.sum(pnl_arr)), 4),
        )

    # ── Confidence Labeling ────────────────────────────────────────

    def _confidence_label(self, result: ExperimentResult) -> str:
        """Assign confidence level based on p-value and effect size."""
        if result.status != "completed":
            return "inconclusive"

        p = result.p_value
        d = abs(result.effect_size)

        if p < 0.05 and d >= 0.8:
            return "high"
        elif p < ALPHA and d >= MIN_EFFECT_SIZE:
            return "medium"
        elif p < 0.15 and d >= 0.2:
            return "low"
        return "inconclusive"

    # ── FDR Correction ─────────────────────────────────────────────

    @staticmethod
    def apply_fdr_correction(
        results: List[ExperimentResult],
    ) -> List[ExperimentResult]:
        """Apply Benjamini-Hochberg FDR correction at q=0.15."""
        if not results:
            return results

        completed = [r for r in results if r.status == "completed"]
        if not completed:
            return results

        # Sort by p-value
        sorted_results = sorted(completed, key=lambda r: r.p_value)
        m = len(sorted_results)

        for i, r in enumerate(sorted_results):
            rank = i + 1
            threshold = (rank / m) * FDR_Q
            if r.p_value > threshold:
                # This and all higher p-values are not significant after FDR
                r.is_significant = False
                r.confidence_level = "inconclusive"

        return results

    # ── Comparison & Reporting ─────────────────────────────────────

    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """Side-by-side comparison of multiple experiments."""
        experiments = []
        for eid in experiment_ids:
            exps = self.db.get_experiments(limit=100)
            for e in exps:
                if e.get("experiment_id") == eid:
                    experiments.append(e)
                    break

        if not experiments:
            return {"error": "No experiments found"}

        return {
            "experiments": [
                {
                    "id": e.get("experiment_id"),
                    "name": e.get("name"),
                    "type": e.get("experiment_type"),
                    "effect_size": e.get("effect_size"),
                    "p_value": e.get("p_value"),
                    "is_significant": e.get("is_significant"),
                    "best_model": e.get("best_model_type"),
                }
                for e in experiments
            ],
            "count": len(experiments),
        }

    def generate_report(self, days: int = 30) -> Dict:
        """Automated experiment summary for the last N days."""
        experiments = self.db.get_experiments(limit=200)

        # Filter by date (approximate — Supabase returns newest first)
        total = len(experiments)
        completed = [e for e in experiments if e.get("status") == "completed"]
        significant = [e for e in completed if e.get("is_significant")]

        return {
            "total_experiments": total,
            "completed": len(completed),
            "significant_findings": len(significant),
            "significant_rate": round(len(significant) / len(completed), 3) if completed else 0,
            "top_findings": [
                {
                    "name": e.get("name"),
                    "effect_size": e.get("effect_size"),
                    "p_value": e.get("p_value"),
                    "narrative": e.get("narrative"),
                }
                for e in significant[:5]
            ],
        }

    # ── Persistence ────────────────────────────────────────────────

    def _save_result(
        self, design: ExperimentDesign, result: ExperimentResult
    ):
        """Persist experiment design and result to DB."""
        try:
            row = {
                "experiment_id": design.experiment_id,
                "hypothesis_id": design.hypothesis_id,
                "name": design.name,
                "experiment_type": design.experiment_type,
                "independent_variable": design.independent_variable,
                "dependent_variable": design.dependent_variable,
                "control_group": json.dumps(design.control_definition, default=str),
                "treatment_group": json.dumps(design.treatment_definition, default=str),
                "min_sample_size": design.min_sample_size,
                "significance_level": design.significance_level,
                "status": result.status,
                "results": json.dumps({
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "confidence_level": result.confidence_level,
                }, default=str),
                "control_metrics": json.dumps(result.control_metrics, default=str),
                "treatment_metrics": json.dumps(result.treatment_metrics, default=str),
                "effect_size": result.effect_size,
                "p_value": result.p_value,
                "is_significant": result.is_significant,
                "model_results": json.dumps(result.model_results, default=str),
                "best_model_type": result.best_model_type,
                "runtime_seconds": result.runtime_seconds,
                "narrative": result.narrative,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat()
                if result.status == "completed" else None,
            }
            self.db.save_experiment(row)
        except Exception as e:
            logger.error(f"Failed to save experiment result: {e}")
