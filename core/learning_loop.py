#!/usr/bin/env python3
"""
LEARNING LOOP — Self-improving orchestrator.

Ties the hypothesis engine, experiment tracker, and model arena together
in a nightly cycle: hypothesize → experiment → validate → incorporate → repeat.

Feature-gated via config.feature_flags.learning_loop_enabled.
"""

import json
import logging
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .config import get_config, get_feature_flags
from .db import get_db
from .feature_engine import FeatureVector
from .hypothesis_engine import HypothesisEngine, Hypothesis
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
MAX_HYPOTHESES_PER_CYCLE = 5
DRIFT_PSI_THRESHOLD = 0.2


@dataclass
class LearningCycleReport:
    """Summary of a complete learning cycle."""
    cycle_id: str = ""
    hypotheses_generated: int = 0
    hypotheses_tested: int = 0
    hypotheses_validated: int = 0
    experiments_run: int = 0
    actions_taken: List[str] = field(default_factory=list)
    features_drifted: List[str] = field(default_factory=list)
    retrain_triggered: bool = False
    executive_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0


class LearningLoop:
    """Self-improving orchestrator — runs the full learning cycle."""

    def __init__(self):
        self.db = get_db()
        self.hypothesis_engine = HypothesisEngine()
        self.experiment_tracker = ExperimentTracker()

    def run_cycle(self) -> LearningCycleReport:
        """Execute a complete learning cycle.

        Steps:
        1. Fetch trades from Supabase
        2. Generate & rank hypotheses (max 5)
        3. Design & run experiments for top hypotheses
        4. Incorporate validated findings
        5. Check feature drift
        6. Generate report
        7. Persist everything
        """
        t0 = time.time()
        report = LearningCycleReport(
            cycle_id=f"cycle_{uuid.uuid4().hex[:12]}"
        )

        logger.info(f"Learning cycle {report.cycle_id} starting...")

        try:
            # Step 1: Fetch trades
            trades = self.db.get_trades_with_features(limit=500)
            if len(trades) < 10:
                report.executive_summary = (
                    f"Insufficient trade data ({len(trades)} trades). "
                    f"Need at least 10 trades with features for learning."
                )
                logger.info(report.executive_summary)
                report.runtime_seconds = round(time.time() - t0, 2)
                return report

            logger.info(f"Fetched {len(trades)} trades with features")

            # Step 2: Generate hypotheses
            hypotheses = self.hypothesis_engine.generate_all(trades)
            report.hypotheses_generated = len(hypotheses)
            logger.info(f"Generated {len(hypotheses)} hypotheses")

            # Take top N
            top_hypotheses = hypotheses[:MAX_HYPOTHESES_PER_CYCLE]

            # Persist hypotheses
            for h in hypotheses:
                self.db.save_hypothesis({
                    "hypothesis_id": h.id,
                    "category": h.category,
                    "statement": h.statement,
                    "source": h.source,
                    "priority_score": h.priority_score,
                    "status": "pending" if h not in top_hypotheses else "testing",
                    "supporting_evidence": json.dumps(h.supporting_evidence, default=str),
                    "effect_size": h.effect_size,
                    "sample_size": h.sample_size,
                    "narrative": h.narrative,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

            # Step 3: Design & run experiments
            results = []
            for h in top_hypotheses:
                design = self.experiment_tracker.design_experiment(h)
                if design is None:
                    continue

                logger.info(f"Running experiment: {design.name}")
                result = self.experiment_tracker.run_experiment(design, trades)
                results.append((h, result))
                report.experiments_run += 1

                # Update hypothesis with experiment result
                self.db.save_hypothesis({
                    "hypothesis_id": h.id,
                    "status": "validated" if result.is_significant else "rejected",
                    "experiment_id": result.experiment_id,
                    "validation_result": json.dumps({
                        "effect_size": result.effect_size,
                        "p_value": result.p_value,
                        "is_significant": result.is_significant,
                        "confidence_level": result.confidence_level,
                    }, default=str),
                    "confidence_level": result.confidence_level,
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                })

            report.hypotheses_tested = len(results)

            # Apply FDR correction across all results
            all_results = [r for _, r in results]
            ExperimentTracker.apply_fdr_correction(all_results)

            # Step 4: Incorporate validated findings
            validated = [
                (h, r) for h, r in results
                if r.is_significant and r.status == "completed"
            ]
            report.hypotheses_validated = len(validated)

            if validated:
                actions = self._step_incorporate_findings(validated)
                report.actions_taken = actions
                logger.info(f"Incorporated {len(actions)} findings")

            # Step 5: Check feature drift
            drifted = self._step_check_drift(trades)
            report.features_drifted = drifted

            if drifted:
                logger.warning(f"Feature drift detected: {drifted}")
                # Trigger retrain if significant drift in top features
                model_info = self.db.get_active_model("signal_quality")
                top_features = []
                if model_info and model_info.get("feature_importance"):
                    fi = model_info["feature_importance"]
                    if isinstance(fi, str):
                        fi = json.loads(fi)
                    top_features = sorted(fi.keys(), key=lambda k: fi[k], reverse=True)[:20]

                drifted_top = [f for f in drifted if f in top_features]
                if drifted_top:
                    report.retrain_triggered = True
                    try:
                        from .ml_pipeline import get_ml_pipeline
                        pipeline = get_ml_pipeline()
                        pipeline.train(force=True)
                        logger.info("ML retrain triggered due to feature drift")
                        report.actions_taken.append(
                            f"ML retrain triggered (drifted features: {drifted_top})"
                        )
                    except Exception as e:
                        logger.error(f"ML retrain failed: {e}")

            # Step 6: Generate summary
            report.executive_summary = self._build_summary(report, trades)
            report.recommendations = self._build_recommendations(validated, drifted)

        except Exception as e:
            logger.error(f"Learning cycle failed: {e}", exc_info=True)
            report.executive_summary = f"Learning cycle failed: {str(e)}"

        report.runtime_seconds = round(time.time() - t0, 2)
        logger.info(
            f"Learning cycle {report.cycle_id} complete in {report.runtime_seconds}s: "
            f"{report.hypotheses_generated} hypotheses, "
            f"{report.experiments_run} experiments, "
            f"{report.hypotheses_validated} validated"
        )

        # Persist the cycle report as a learning action
        self.db.save_learning_action({
            "action_type": "cycle_complete",
            "description": report.executive_summary,
            "before_state": json.dumps({
                "hypotheses_generated": report.hypotheses_generated,
                "experiments_run": report.experiments_run,
            }, default=str),
            "after_state": json.dumps({
                "hypotheses_validated": report.hypotheses_validated,
                "actions_taken": report.actions_taken,
                "features_drifted": report.features_drifted,
                "retrain_triggered": report.retrain_triggered,
                "recommendations": report.recommendations,
            }, default=str),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        return report

    # ── Step 4: Incorporate Findings ───────────────────────────────

    def _step_incorporate_findings(
        self, validated: List[tuple]
    ) -> List[str]:
        """Apply validated results to the live system."""
        actions = []

        for hypothesis, result in validated:
            try:
                if hypothesis.category in ("regime_conditional", "feature_interaction"):
                    action = self._incorporate_regime_finding(hypothesis, result)
                    if action:
                        actions.append(action)

                elif hypothesis.category == "temporal_pattern":
                    action = self._incorporate_temporal_finding(hypothesis, result)
                    if action:
                        actions.append(action)

                elif hypothesis.category == "feature_drift":
                    actions.append(
                        f"Feature drift flagged: {hypothesis.source} "
                        f"(PSI={result.effect_size:.3f})"
                    )

            except Exception as e:
                logger.error(f"Failed to incorporate finding: {e}")

        return actions

    def _incorporate_regime_finding(
        self, hypothesis: Hypothesis, result
    ) -> Optional[str]:
        """Store regime-specific weight overrides."""
        evidence = hypothesis.supporting_evidence
        regime = evidence.get("regime")
        if not regime:
            return None

        # Determine weight adjustment based on effect
        if result.effect_size > 0:
            multiplier = min(1.5, 1.0 + abs(result.effect_size) * 0.2)
        else:
            multiplier = max(0.5, 1.0 - abs(result.effect_size) * 0.2)

        try:
            self.db._client.table("ensemble_weight_overrides").upsert({
                "regime": regime,
                "signal_type": evidence.get("signal_type", "all"),
                "weight_multiplier": round(multiplier, 3),
                "source_experiment_id": result.experiment_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="regime,signal_type").execute()

            self.db.save_learning_action({
                "action_type": "weight_override",
                "source_experiment_id": result.experiment_id,
                "source_hypothesis_id": hypothesis.id,
                "description": (
                    f"Set weight multiplier {multiplier:.3f} for "
                    f"regime={regime} (d={result.effect_size:.2f})"
                ),
                "before_state": json.dumps({"multiplier": 1.0}),
                "after_state": json.dumps({"multiplier": multiplier}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

            return (
                f"Weight override: regime={regime}, multiplier={multiplier:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to save weight override: {e}")
            return None

    def _incorporate_temporal_finding(
        self, hypothesis: Hypothesis, result
    ) -> Optional[str]:
        """Store temporal sizing adjustments."""
        evidence = hypothesis.supporting_evidence
        dimension = evidence.get("dimension")  # e.g., "day_of_week", "hour_bucket"
        value = evidence.get("value")  # e.g., 0 (Monday), "morning"

        if dimension is None or value is None:
            return None

        # Determine sizing adjustment
        if result.effect_size > 0:
            multiplier = min(1.3, 1.0 + abs(result.effect_size) * 0.15)
        else:
            multiplier = max(0.7, 1.0 - abs(result.effect_size) * 0.15)

        try:
            value_int = int(value) if isinstance(value, (int, float)) else hash(str(value)) % 100

            self.db._client.table("temporal_adjustments").upsert({
                "dimension": str(dimension),
                "value": value_int,
                "size_multiplier": round(multiplier, 3),
                "source_experiment_id": result.experiment_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="dimension,value").execute()

            self.db.save_learning_action({
                "action_type": "temporal_adjustment",
                "source_experiment_id": result.experiment_id,
                "source_hypothesis_id": hypothesis.id,
                "description": (
                    f"Set size multiplier {multiplier:.3f} for "
                    f"{dimension}={value} (d={result.effect_size:.2f})"
                ),
                "before_state": json.dumps({"multiplier": 1.0}),
                "after_state": json.dumps({"multiplier": multiplier}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

            return (
                f"Temporal adjustment: {dimension}={value}, "
                f"size_multiplier={multiplier:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to save temporal adjustment: {e}")
            return None

    # ── Step 5: Check Drift ────────────────────────────────────────

    def _step_check_drift(self, trades: List[Dict]) -> List[str]:
        """Check PSI per feature, flag if > threshold for top features."""
        if len(trades) < 20:
            return []

        drifted_features = []
        feature_names = FeatureVector.feature_names()

        # Split into first half (training proxy) and recent half
        mid = len(trades) // 2
        old_trades = trades[mid:]  # older (higher index = older since sorted desc)
        new_trades = trades[:mid]  # newer

        for fname in feature_names:
            try:
                old_vals = []
                new_vals = []

                for t in old_trades:
                    val = self._extract_feature(t, fname)
                    if val is not None:
                        old_vals.append(val)

                for t in new_trades:
                    val = self._extract_feature(t, fname)
                    if val is not None:
                        new_vals.append(val)

                if len(old_vals) < 5 or len(new_vals) < 5:
                    continue

                old_arr = np.array(old_vals)
                new_arr = np.array(new_vals)

                psi = self._psi(old_arr, new_arr)
                if psi > DRIFT_PSI_THRESHOLD:
                    drifted_features.append(fname)
                    self.db.save_feature_drift({
                        "feature_name": fname,
                        "drift_type": "psi",
                        "current_mean": round(float(np.mean(new_arr)), 6),
                        "training_mean": round(float(np.mean(old_arr)), 6),
                        "current_std": round(float(np.std(new_arr)), 6),
                        "training_std": round(float(np.std(old_arr)), 6),
                        "drift_magnitude": round(psi, 6),
                        "is_significant": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    })

            except Exception:
                continue

        return drifted_features

    def _extract_feature(self, trade: Dict, feature: str) -> Optional[float]:
        """Extract feature value from trade."""
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

    def _psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 5
    ) -> float:
        """Population Stability Index (5 bins for small samples)."""
        try:
            # Use quantile-based bins from the expected distribution
            breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 3:
                return 0.0

            expected_counts = np.histogram(expected, bins=breakpoints)[0]
            actual_counts = np.histogram(actual, bins=breakpoints)[0]

            # Convert to proportions with Laplace smoothing
            expected_pct = (expected_counts + 0.5) / (len(expected) + 0.5 * len(expected_counts))
            actual_pct = (actual_counts + 0.5) / (len(actual) + 0.5 * len(actual_counts))

            psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
            return max(0.0, psi)
        except Exception:
            return 0.0

    # ── Report Building ────────────────────────────────────────────

    def _build_summary(
        self, report: LearningCycleReport, trades: List[Dict]
    ) -> str:
        """Build executive summary."""
        parts = [
            f"Learning cycle analyzed {len(trades)} trades.",
            f"Generated {report.hypotheses_generated} hypotheses, "
            f"tested {report.hypotheses_tested}, "
            f"validated {report.hypotheses_validated}.",
        ]

        if report.actions_taken:
            parts.append(
                f"Took {len(report.actions_taken)} actions: "
                + "; ".join(report.actions_taken[:3])
            )

        if report.features_drifted:
            parts.append(
                f"Drift detected in {len(report.features_drifted)} features."
            )

        if report.retrain_triggered:
            parts.append("ML model retrain was triggered.")

        return " ".join(parts)

    def _build_recommendations(
        self,
        validated: List[tuple],
        drifted: List[str],
    ) -> List[str]:
        """Build actionable recommendations."""
        recs = []

        for hypothesis, result in validated:
            if result.confidence_level in ("high", "medium"):
                recs.append(
                    f"[{result.confidence_level.upper()}] {hypothesis.statement} "
                    f"(d={result.effect_size:.2f}, p={result.p_value:.3f})"
                )

        if len(drifted) > 3:
            recs.append(
                f"Significant feature drift in {len(drifted)} features. "
                f"Consider investigating: {', '.join(drifted[:5])}"
            )

        if not recs:
            recs.append("No actionable findings this cycle. Continue collecting data.")

        return recs

    # ── Cumulative Insights ────────────────────────────────────────

    def get_cumulative_insights(self) -> Dict:
        """Aggregate all validated findings across cycles."""
        validated = self.db.get_hypotheses(status="validated", limit=200)
        experiments = self.db.get_experiments(limit=200)
        overrides = self.db.get_ensemble_weight_overrides()
        adjustments = self.db.get_temporal_adjustments()

        return {
            "total_validated_hypotheses": len(validated),
            "total_experiments": len(experiments),
            "active_weight_overrides": len(overrides),
            "active_temporal_adjustments": len(adjustments),
            "weight_overrides": overrides,
            "temporal_adjustments": adjustments,
            "recent_validated": [
                {
                    "statement": h.get("statement"),
                    "effect_size": h.get("effect_size"),
                    "confidence_level": h.get("confidence_level"),
                }
                for h in validated[:10]
            ],
        }
