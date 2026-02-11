#!/usr/bin/env python3
"""
HYPOTHESIS ENGINE — Auto-generates testable hypotheses from trade data.

Scans trade history for patterns using small-sample-safe statistics
(bootstrap CI, Cohen's d, PSI). Generates ranked hypotheses for
the experiment tracker to validate.

Designed for paper trading phase (~30 trades). All statistical methods
handle small samples gracefully: bootstrap CI instead of t-tests,
conservative bin counts for PSI, minimum subgroup sizes enforced.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .db import get_db
from .feature_engine import FeatureVector

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
MIN_SUBGROUP_SIZE = 5
MIN_EFFECT_SIZE = 0.3  # Cohen's d threshold


# ── Dataclass ────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """A testable hypothesis discovered from trade data."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = ""
    statement: str = ""
    source: str = ""
    priority_score: float = 0.0
    effect_size: float = 0.0
    sample_size: int = 0
    status: str = "pending"
    narrative: str = ""
    supporting_evidence: dict = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_feature_value(trade: Dict, feature_name: str) -> Optional[float]:
    """Extract a numeric feature value from a trade's entry_features dict.

    Handles cases where entry_features is already a dict or still a JSON string.
    Returns None if the feature is missing, non-numeric, or extraction fails.
    """
    try:
        features = trade.get("entry_features")
        if features is None:
            return None

        # Parse JSON string if necessary
        if isinstance(features, str):
            try:
                features = json.loads(features)
            except (json.JSONDecodeError, TypeError):
                return None

        if not isinstance(features, dict):
            return None

        val = features.get(feature_name)
        if val is None:
            return None

        return float(val)
    except (ValueError, TypeError):
        return None


# ── Engine ───────────────────────────────────────────────────────────────

class HypothesisEngine:
    """Scans trade history for statistically meaningful patterns
    and generates ranked, testable hypotheses."""

    def __init__(self):
        self.db = get_db()

    # ── Master method ────────────────────────────────────────────────

    def generate_all(self, trades: List[Dict]) -> List[Hypothesis]:
        """Run all scanners, deduplicate, rank, and return top hypotheses.

        Returns an empty list if the trade set is too small to draw
        any meaningful conclusions (fewer than MIN_SUBGROUP_SIZE * 2).
        """
        if not trades or len(trades) < MIN_SUBGROUP_SIZE * 2:
            logger.info(
                "Insufficient trades for hypothesis generation "
                f"(have {len(trades) if trades else 0}, need {MIN_SUBGROUP_SIZE * 2})"
            )
            return []

        all_hypotheses: List[Hypothesis] = []

        scanners = [
            ("regime_interactions", self.scan_regime_interactions),
            ("feature_thresholds", self.scan_feature_thresholds),
            ("temporal_patterns", self.scan_temporal_patterns),
            ("feature_drift", self.scan_feature_drift),
            ("exit_reason_patterns", self.scan_exit_reason_patterns),
            ("signal_accuracy_by_regime", self.scan_signal_accuracy_by_regime),
        ]

        for name, scanner in scanners:
            try:
                results = scanner(trades)
                logger.debug(f"Scanner '{name}' produced {len(results)} hypotheses")
                all_hypotheses.extend(results)
            except Exception as e:
                logger.warning(f"Scanner '{name}' failed: {e}", exc_info=True)

        # Deduplicate by statement (exact match)
        seen_statements: set = set()
        unique: List[Hypothesis] = []
        for h in all_hypotheses:
            if h.statement not in seen_statements:
                seen_statements.add(h.statement)
                unique.append(h)

        ranked = self._rank_hypotheses(unique)
        logger.info(
            f"Hypothesis engine produced {len(ranked)} hypotheses "
            f"from {len(trades)} trades"
        )
        return ranked

    # ── Scanners ─────────────────────────────────────────────────────

    def scan_regime_interactions(self, trades: List[Dict]) -> List[Hypothesis]:
        """Compare PnL across market regimes.

        Groups trades by regime_at_entry. For each regime with enough trades,
        bootstrap the mean PnL difference against all other trades and
        compute Cohen's d. Generates a hypothesis if the effect is large enough.
        """
        try:
            # Group by regime
            regime_groups: Dict[str, List[float]] = {}
            for t in trades:
                regime = t.get("regime_at_entry")
                pnl = t.get("pnl_pct")
                if regime and pnl is not None:
                    regime_groups.setdefault(regime, []).append(float(pnl))

            hypotheses: List[Hypothesis] = []
            all_pnl = [float(t["pnl_pct"]) for t in trades if t.get("pnl_pct") is not None]
            if len(all_pnl) < MIN_SUBGROUP_SIZE:
                return []

            for regime, pnl_list in regime_groups.items():
                if len(pnl_list) < MIN_SUBGROUP_SIZE:
                    continue

                # Other trades = everything not in this regime
                other_pnl = [p for t_regime, pnl_group in regime_groups.items()
                             if t_regime != regime for p in pnl_group]
                if len(other_pnl) < MIN_SUBGROUP_SIZE:
                    continue

                a = np.array(pnl_list)
                b = np.array(other_pnl)

                d = self._cohens_d(a, b)
                if abs(d) < MIN_EFFECT_SIZE:
                    continue

                obs_diff, ci_lo, ci_hi = self._bootstrap_mean_diff(a, b)
                direction = "higher" if obs_diff > 0 else "lower"
                mean_pnl = float(np.mean(a))
                n = len(pnl_list)

                hypotheses.append(Hypothesis(
                    category="regime_interaction",
                    statement=(
                        f"Trades in {regime} regime show {direction} returns "
                        f"(d={d:.2f}, n={n})"
                    ),
                    source="scan_regime_interactions",
                    effect_size=abs(d),
                    sample_size=n,
                    narrative=(
                        f"Trades entered during the '{regime}' regime averaged "
                        f"{mean_pnl:.2f}% PnL vs {float(np.mean(b)):.2f}% for other regimes. "
                        f"Bootstrap 90% CI of the difference: [{ci_lo:.3f}%, {ci_hi:.3f}%]."
                    ),
                    supporting_evidence={
                        "regime": regime,
                        "mean_pnl_regime": round(mean_pnl, 4),
                        "mean_pnl_other": round(float(np.mean(b)), 4),
                        "cohens_d": round(d, 4),
                        "bootstrap_ci_90": [round(ci_lo, 4), round(ci_hi, 4)],
                        "n_regime": n,
                        "n_other": len(other_pnl),
                    },
                ))

            return hypotheses

        except Exception as e:
            logger.warning(f"scan_regime_interactions failed: {e}", exc_info=True)
            return []

    def scan_feature_thresholds(self, trades: List[Dict]) -> List[Hypothesis]:
        """Find features whose median split correlates with PnL differences.

        Prioritises the top-20 features by model importance when an active
        model is available. Falls back to scanning all features.
        """
        try:
            # Determine which features to scan
            feature_names = self._get_priority_features()

            categories = FeatureVector.feature_categories()
            # Build reverse lookup: feature_name -> category
            feature_to_category: Dict[str, str] = {}
            for cat, names in categories.items():
                for name in names:
                    feature_to_category[name] = cat

            hypotheses: List[Hypothesis] = []

            for feat in feature_names:
                # Extract (feature_value, pnl) pairs
                pairs: List[Tuple[float, float]] = []
                for t in trades:
                    val = _extract_feature_value(t, feat)
                    pnl = t.get("pnl_pct")
                    if val is not None and pnl is not None:
                        pairs.append((float(val), float(pnl)))

                if len(pairs) < MIN_SUBGROUP_SIZE * 2:
                    continue

                values = np.array([p[0] for p in pairs])
                pnls = np.array([p[1] for p in pairs])

                median_val = float(np.median(values))
                above_mask = values >= median_val
                below_mask = ~above_mask

                above_pnl = pnls[above_mask]
                below_pnl = pnls[below_mask]

                if len(above_pnl) < MIN_SUBGROUP_SIZE or len(below_pnl) < MIN_SUBGROUP_SIZE:
                    continue

                d = self._cohens_d(above_pnl, below_pnl)
                if abs(d) < MIN_EFFECT_SIZE:
                    continue

                obs_diff, ci_lo, ci_hi = self._bootstrap_mean_diff(above_pnl, below_pnl)
                direction = "better" if obs_diff > 0 else "worse"
                cat = feature_to_category.get(feat, "unknown")

                hypotheses.append(Hypothesis(
                    category="feature_threshold",
                    statement=(
                        f"Trades with {feat} above median ({median_val:.3f}) "
                        f"perform {direction} (d={d:.2f}, n={len(pairs)})"
                    ),
                    source="scan_feature_thresholds",
                    effect_size=abs(d),
                    sample_size=len(pairs),
                    narrative=(
                        f"Splitting trades at the median of '{feat}' ({cat} category) "
                        f"reveals a Cohen's d of {d:.2f}. Above-median trades averaged "
                        f"{float(np.mean(above_pnl)):.2f}% PnL vs "
                        f"{float(np.mean(below_pnl)):.2f}% below-median. "
                        f"Bootstrap 90% CI: [{ci_lo:.3f}%, {ci_hi:.3f}%]."
                    ),
                    supporting_evidence={
                        "feature": feat,
                        "feature_category": cat,
                        "median_threshold": round(median_val, 4),
                        "mean_pnl_above": round(float(np.mean(above_pnl)), 4),
                        "mean_pnl_below": round(float(np.mean(below_pnl)), 4),
                        "cohens_d": round(d, 4),
                        "bootstrap_ci_90": [round(ci_lo, 4), round(ci_hi, 4)],
                        "n_above": int(np.sum(above_mask)),
                        "n_below": int(np.sum(below_mask)),
                    },
                ))

            return hypotheses

        except Exception as e:
            logger.warning(f"scan_feature_thresholds failed: {e}", exc_info=True)
            return []

    def scan_temporal_patterns(self, trades: List[Dict]) -> List[Hypothesis]:
        """Detect day-of-week and time-of-day PnL patterns.

        Groups trades by the day_of_week and hour_of_day stored in
        entry_features, then compares each group against the rest.
        """
        try:
            hypotheses: List[Hypothesis] = []

            # ── Day of week ──────────────────────────────────────
            day_groups: Dict[int, List[float]] = {}
            for t in trades:
                day = _extract_feature_value(t, "day_of_week")
                pnl = t.get("pnl_pct")
                if day is not None and pnl is not None:
                    day_int = int(day)
                    if 0 <= day_int <= 4:
                        day_groups.setdefault(day_int, []).append(float(pnl))

            day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                         3: "Thursday", 4: "Friday"}

            all_day_pnl = [p for group in day_groups.values() for p in group]
            if len(all_day_pnl) >= MIN_SUBGROUP_SIZE:
                for day_int, pnl_list in day_groups.items():
                    if len(pnl_list) < MIN_SUBGROUP_SIZE:
                        continue
                    other = [p for d, group in day_groups.items()
                             if d != day_int for p in group]
                    if len(other) < MIN_SUBGROUP_SIZE:
                        continue

                    a = np.array(pnl_list)
                    b = np.array(other)
                    d = self._cohens_d(a, b)
                    if abs(d) < MIN_EFFECT_SIZE:
                        continue

                    obs_diff, ci_lo, ci_hi = self._bootstrap_mean_diff(a, b)
                    direction = "outperform" if obs_diff > 0 else "underperform"
                    name = day_names.get(day_int, f"Day {day_int}")

                    hypotheses.append(Hypothesis(
                        category="temporal_pattern",
                        statement=(
                            f"Trades entered on {name} {direction} "
                            f"(d={d:.2f}, n={len(pnl_list)})"
                        ),
                        source="scan_temporal_patterns",
                        effect_size=abs(d),
                        sample_size=len(pnl_list),
                        narrative=(
                            f"{name} trades average {float(np.mean(a)):.2f}% PnL "
                            f"vs {float(np.mean(b)):.2f}% on other days. "
                            f"Bootstrap 90% CI: [{ci_lo:.3f}%, {ci_hi:.3f}%]."
                        ),
                        supporting_evidence={
                            "day_of_week": day_int,
                            "day_name": name,
                            "mean_pnl_day": round(float(np.mean(a)), 4),
                            "mean_pnl_other": round(float(np.mean(b)), 4),
                            "cohens_d": round(d, 4),
                            "n_day": len(pnl_list),
                        },
                    ))

            # ── Hour bucket ──────────────────────────────────────
            bucket_groups: Dict[str, List[float]] = {}
            bucket_defs = {
                "morning": (9, 12),
                "afternoon": (12, 16),
                "after_hours": (16, 24),
            }

            for t in trades:
                hour = _extract_feature_value(t, "hour_of_day")
                pnl = t.get("pnl_pct")
                if hour is not None and pnl is not None:
                    hour_int = int(hour)
                    for bucket_name, (lo, hi) in bucket_defs.items():
                        if lo <= hour_int < hi:
                            bucket_groups.setdefault(bucket_name, []).append(float(pnl))
                            break

            all_bucket_pnl = [p for group in bucket_groups.values() for p in group]
            if len(all_bucket_pnl) >= MIN_SUBGROUP_SIZE:
                for bucket, pnl_list in bucket_groups.items():
                    if len(pnl_list) < MIN_SUBGROUP_SIZE:
                        continue
                    other = [p for b, group in bucket_groups.items()
                             if b != bucket for p in group]
                    if len(other) < MIN_SUBGROUP_SIZE:
                        continue

                    a = np.array(pnl_list)
                    b = np.array(other)
                    d = self._cohens_d(a, b)
                    if abs(d) < MIN_EFFECT_SIZE:
                        continue

                    obs_diff, ci_lo, ci_hi = self._bootstrap_mean_diff(a, b)
                    direction = "outperform" if obs_diff > 0 else "underperform"
                    label = bucket.replace("_", " ").title()

                    hypotheses.append(Hypothesis(
                        category="temporal_pattern",
                        statement=(
                            f"Trades entered during {label} hours {direction} "
                            f"(d={d:.2f}, n={len(pnl_list)})"
                        ),
                        source="scan_temporal_patterns",
                        effect_size=abs(d),
                        sample_size=len(pnl_list),
                        narrative=(
                            f"{label} trades average {float(np.mean(a)):.2f}% PnL "
                            f"vs {float(np.mean(b)):.2f}% at other times. "
                            f"Bootstrap 90% CI: [{ci_lo:.3f}%, {ci_hi:.3f}%]."
                        ),
                        supporting_evidence={
                            "hour_bucket": bucket,
                            "mean_pnl_bucket": round(float(np.mean(a)), 4),
                            "mean_pnl_other": round(float(np.mean(b)), 4),
                            "cohens_d": round(d, 4),
                            "n_bucket": len(pnl_list),
                        },
                    ))

            return hypotheses

        except Exception as e:
            logger.warning(f"scan_temporal_patterns failed: {e}", exc_info=True)
            return []

    def scan_feature_drift(self, trades: List[Dict]) -> List[Hypothesis]:
        """Detect features whose distribution has shifted over time.

        Splits trades chronologically into first and second halves, then
        computes Population Stability Index (PSI) for each feature. A PSI
        above 0.2 indicates significant distributional drift.
        """
        try:
            if len(trades) < MIN_SUBGROUP_SIZE * 2:
                return []

            # Sort chronologically (oldest first)
            sorted_trades = sorted(
                trades,
                key=lambda t: t.get("entry_date") or t.get("exit_date") or ""
            )
            mid = len(sorted_trades) // 2
            first_half = sorted_trades[:mid]
            second_half = sorted_trades[mid:]

            feature_names = FeatureVector.feature_names()
            hypotheses: List[Hypothesis] = []

            for feat in feature_names:
                first_vals = []
                second_vals = []

                for t in first_half:
                    v = _extract_feature_value(t, feat)
                    if v is not None:
                        first_vals.append(v)

                for t in second_half:
                    v = _extract_feature_value(t, feat)
                    if v is not None:
                        second_vals.append(v)

                if len(first_vals) < MIN_SUBGROUP_SIZE or len(second_vals) < MIN_SUBGROUP_SIZE:
                    continue

                expected = np.array(first_vals)
                actual = np.array(second_vals)

                psi = self._population_stability_index(expected, actual)
                if psi <= 0.2:
                    continue

                hypotheses.append(Hypothesis(
                    category="feature_drift",
                    statement=(
                        f"Feature {feat} has drifted significantly (PSI={psi:.3f})"
                    ),
                    source="scan_feature_drift",
                    effect_size=psi,
                    sample_size=len(first_vals) + len(second_vals),
                    narrative=(
                        f"The distribution of '{feat}' has shifted between the first "
                        f"{len(first_vals)} and last {len(second_vals)} trades "
                        f"(PSI={psi:.3f}). PSI > 0.2 indicates the model's training "
                        f"distribution may no longer match live conditions. "
                        f"First half mean: {float(np.mean(expected)):.3f}, "
                        f"second half mean: {float(np.mean(actual)):.3f}."
                    ),
                    supporting_evidence={
                        "feature": feat,
                        "psi": round(psi, 4),
                        "mean_first_half": round(float(np.mean(expected)), 4),
                        "mean_second_half": round(float(np.mean(actual)), 4),
                        "std_first_half": round(float(np.std(expected)), 4),
                        "std_second_half": round(float(np.std(actual)), 4),
                        "n_first": len(first_vals),
                        "n_second": len(second_vals),
                    },
                ))

            return hypotheses

        except Exception as e:
            logger.warning(f"scan_feature_drift failed: {e}", exc_info=True)
            return []

    def scan_exit_reason_patterns(self, trades: List[Dict]) -> List[Hypothesis]:
        """Identify entry conditions that predict specific exit types.

        For each exit_reason category (stop_loss, take_profit, etc.),
        compares entry feature distributions against trades with other
        exit reasons using Cohen's d.
        """
        try:
            # Group trades by exit reason
            exit_groups: Dict[str, List[Dict]] = {}
            for t in trades:
                reason = t.get("exit_reason")
                if reason:
                    exit_groups.setdefault(reason, []).append(t)

            hypotheses: List[Hypothesis] = []
            feature_names = self._get_priority_features()

            for reason, group_trades in exit_groups.items():
                if len(group_trades) < MIN_SUBGROUP_SIZE:
                    continue

                other_trades = [t for r, grp in exit_groups.items()
                                if r != reason for t in grp]
                if len(other_trades) < MIN_SUBGROUP_SIZE:
                    continue

                # Compare entry features between this exit reason and others
                for feat in feature_names:
                    group_vals = []
                    for t in group_trades:
                        v = _extract_feature_value(t, feat)
                        if v is not None:
                            group_vals.append(v)

                    other_vals = []
                    for t in other_trades:
                        v = _extract_feature_value(t, feat)
                        if v is not None:
                            other_vals.append(v)

                    if len(group_vals) < MIN_SUBGROUP_SIZE or len(other_vals) < MIN_SUBGROUP_SIZE:
                        continue

                    a = np.array(group_vals)
                    b = np.array(other_vals)
                    d = self._cohens_d(a, b)

                    if abs(d) < MIN_EFFECT_SIZE:
                        continue

                    direction = "higher" if d > 0 else "lower"
                    reason_label = reason.replace("_", " ")

                    hypotheses.append(Hypothesis(
                        category="exit_reason_pattern",
                        statement=(
                            f"Trades exiting via {reason_label} have {direction} "
                            f"entry {feat} (d={d:.2f}, n={len(group_vals)})"
                        ),
                        source="scan_exit_reason_patterns",
                        effect_size=abs(d),
                        sample_size=len(group_vals),
                        narrative=(
                            f"Trades that exit via '{reason_label}' show a distinctly "
                            f"{direction} '{feat}' at entry (mean={float(np.mean(a)):.3f}) "
                            f"compared to other exits (mean={float(np.mean(b)):.3f}). "
                            f"Cohen's d = {d:.2f}. This suggests '{feat}' may predict "
                            f"'{reason_label}' exits."
                        ),
                        supporting_evidence={
                            "exit_reason": reason,
                            "feature": feat,
                            "mean_group": round(float(np.mean(a)), 4),
                            "mean_other": round(float(np.mean(b)), 4),
                            "cohens_d": round(d, 4),
                            "n_group": len(group_vals),
                            "n_other": len(other_vals),
                        },
                    ))

            return hypotheses

        except Exception as e:
            logger.warning(f"scan_exit_reason_patterns failed: {e}", exc_info=True)
            return []

    def scan_signal_accuracy_by_regime(self, trades: List[Dict]) -> List[Hypothesis]:
        """Detect regimes where signal accuracy diverges from baseline.

        Attempts to read signal_evaluations from the DB for pre-computed
        accuracy data. Falls back to computing win rate per regime from
        trade outcomes.
        """
        try:
            hypotheses: List[Hypothesis] = []

            # Try to read signal evaluations from DB
            evaluations = None
            try:
                if self.db.connected and self.db._client:
                    resp = (self.db._client.table("signal_evaluations")
                            .select("*")
                            .limit(500)
                            .execute())
                    evaluations = resp.data if resp.data else None
            except Exception:
                # Table may not exist — that's fine
                evaluations = None

            if evaluations and len(evaluations) >= MIN_SUBGROUP_SIZE:
                # Use pre-computed signal evaluations
                regime_acc: Dict[str, List[float]] = {}
                for ev in evaluations:
                    regime = ev.get("regime")
                    accuracy = ev.get("accuracy")
                    if regime and accuracy is not None:
                        regime_acc.setdefault(regime, []).append(float(accuracy))

                all_acc = [a for group in regime_acc.values() for a in group]
                if len(all_acc) >= MIN_SUBGROUP_SIZE:
                    overall_acc = float(np.mean(all_acc))

                    for regime, acc_list in regime_acc.items():
                        if len(acc_list) < MIN_SUBGROUP_SIZE:
                            continue
                        other_acc = [a for r, g in regime_acc.items()
                                     if r != regime for a in g]
                        if len(other_acc) < MIN_SUBGROUP_SIZE:
                            continue

                        a = np.array(acc_list)
                        b = np.array(other_acc)
                        d = self._cohens_d(a, b)
                        if abs(d) < MIN_EFFECT_SIZE:
                            continue

                        regime_mean = float(np.mean(a))
                        direction = "higher" if regime_mean > overall_acc else "lower"

                        hypotheses.append(Hypothesis(
                            category="signal_accuracy",
                            statement=(
                                f"Signal accuracy in {regime} regime is {direction} "
                                f"than average (d={d:.2f}, n={len(acc_list)})"
                            ),
                            source="scan_signal_accuracy_by_regime",
                            effect_size=abs(d),
                            sample_size=len(acc_list),
                            narrative=(
                                f"Signal accuracy in the '{regime}' regime averages "
                                f"{regime_mean:.1%} vs {overall_acc:.1%} overall. "
                                f"Cohen's d = {d:.2f}."
                            ),
                            supporting_evidence={
                                "regime": regime,
                                "mean_accuracy_regime": round(regime_mean, 4),
                                "mean_accuracy_overall": round(overall_acc, 4),
                                "cohens_d": round(d, 4),
                                "n_evaluations": len(acc_list),
                                "data_source": "signal_evaluations",
                            },
                        ))

                return hypotheses

            # Fallback: compute win rate per regime from trade outcomes
            regime_wins: Dict[str, List[int]] = {}
            for t in trades:
                regime = t.get("regime_at_entry")
                pnl = t.get("pnl_pct")
                if regime and pnl is not None:
                    win = 1 if float(pnl) > 0 else 0
                    regime_wins.setdefault(regime, []).append(win)

            all_wins = [w for group in regime_wins.values() for w in group]
            if len(all_wins) < MIN_SUBGROUP_SIZE:
                return []
            overall_wr = float(np.mean(all_wins))

            for regime, wins_list in regime_wins.items():
                if len(wins_list) < MIN_SUBGROUP_SIZE:
                    continue

                regime_wr = float(np.mean(wins_list))
                diff = regime_wr - overall_wr

                # Use Cohen's h for proportions as an effect size proxy
                # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
                p1 = np.clip(regime_wr, 0.001, 0.999)
                p2 = np.clip(overall_wr, 0.001, 0.999)
                h = float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))

                if abs(h) < MIN_EFFECT_SIZE:
                    continue

                direction = "higher" if diff > 0 else "lower"
                n = len(wins_list)

                hypotheses.append(Hypothesis(
                    category="signal_accuracy",
                    statement=(
                        f"Win rate in {regime} regime is {direction} "
                        f"than average (h={h:.2f}, n={n})"
                    ),
                    source="scan_signal_accuracy_by_regime",
                    effect_size=abs(h),
                    sample_size=n,
                    narrative=(
                        f"Win rate in the '{regime}' regime is "
                        f"{regime_wr:.0%} ({n} trades) vs {overall_wr:.0%} overall. "
                        f"Cohen's h = {h:.2f}."
                    ),
                    supporting_evidence={
                        "regime": regime,
                        "win_rate_regime": round(regime_wr, 4),
                        "win_rate_overall": round(overall_wr, 4),
                        "cohens_h": round(h, 4),
                        "n_trades": n,
                        "data_source": "trade_outcomes",
                    },
                ))

            return hypotheses

        except Exception as e:
            logger.warning(f"scan_signal_accuracy_by_regime failed: {e}", exc_info=True)
            return []

    # ── Ranking ──────────────────────────────────────────────────────

    def _rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Score and sort hypotheses by actionability.

        Scoring formula:
            priority = impact (0.4) + testability (0.3) + novelty (0.3)

        - impact: proportional to effect_size (capped at d=2.0)
        - testability: proportional to sample_size (saturates at n=30)
        - novelty: 1.0 for now (future: penalise previously tested)
        """
        for h in hypotheses:
            impact = min(h.effect_size / 2.0, 1.0) * 0.4
            sample_size_factor = min(1.0, h.sample_size / 30.0)
            testability = sample_size_factor * 0.3
            novelty = 1.0 * 0.3  # Placeholder — future: check DB for prior tests
            h.priority_score = round(impact + testability + novelty, 4)

        hypotheses.sort(key=lambda h: h.priority_score, reverse=True)
        return hypotheses

    # ── Statistical methods ──────────────────────────────────────────

    def _bootstrap_mean_diff(
        self,
        a: np.ndarray,
        b: np.ndarray,
        n_iterations: int = 1000,
    ) -> Tuple[float, float, float]:
        """Bootstrap the difference in means between two samples.

        Returns:
            (observed_diff, ci_lower, ci_upper) where CI is the 90%
            confidence interval computed from the bootstrap distribution.
        """
        rng = np.random.default_rng()
        observed_diff = float(np.mean(a) - np.mean(b))
        diffs = np.empty(n_iterations)

        for i in range(n_iterations):
            a_boot = rng.choice(a, size=len(a), replace=True)
            b_boot = rng.choice(b, size=len(b), replace=True)
            diffs[i] = np.mean(a_boot) - np.mean(b_boot)

        ci_lower = float(np.percentile(diffs, 5))
        ci_upper = float(np.percentile(diffs, 95))
        return observed_diff, ci_lower, ci_upper

    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d effect size using pooled standard deviation.

        Returns 0.0 when the pooled std is zero (i.e. both samples are
        constant), which avoids division-by-zero.
        """
        n1 = len(a)
        n2 = len(b)
        if n1 < 2 or n2 < 2:
            return 0.0

        var1 = float(np.var(a, ddof=1))
        var2 = float(np.var(b, ddof=1))
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return float((np.mean(a) - np.mean(b)) / pooled_std)

    def _population_stability_index(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 5,
    ) -> float:
        """Population Stability Index — measures distributional drift.

        Uses 5 bins by default (suitable for small samples). Applies
        Laplace smoothing to avoid log(0) when a bin is empty.

        PSI interpretation:
            < 0.1  — no significant shift
            0.1-0.2 — moderate shift, monitor
            > 0.2  — significant shift, investigate

        Returns 0.0 if inputs are degenerate (e.g. constant arrays).
        """
        # Compute bin edges from the expected distribution
        combined = np.concatenate([expected, actual])
        unique_vals = np.unique(combined)

        if len(unique_vals) < 2:
            return 0.0

        # Use quantile-based bins for robustness with small samples
        try:
            percentiles = np.linspace(0, 100, bins + 1)
            bin_edges = np.percentile(expected, percentiles)
            # Ensure edges are unique (can collapse with small samples)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                return 0.0
        except Exception:
            return 0.0

        n_bins = len(bin_edges) - 1

        # Digitize into bins
        exp_counts = np.histogram(expected, bins=bin_edges)[0].astype(float)
        act_counts = np.histogram(actual, bins=bin_edges)[0].astype(float)

        # Laplace smoothing — add a small count to avoid zeros
        smoothing = 0.5
        exp_counts += smoothing
        act_counts += smoothing

        # Convert to proportions
        exp_pct = exp_counts / exp_counts.sum()
        act_pct = act_counts / act_counts.sum()

        # PSI = sum( (actual% - expected%) * ln(actual% / expected%) )
        psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
        return max(0.0, psi)

    # ── Internal helpers ─────────────────────────────────────────────

    def _get_priority_features(self, top_n: int = 20) -> List[str]:
        """Return the top-N features by importance from the active model.

        Falls back to the full feature list if no model is available or
        if feature_importance is missing/empty.
        """
        all_features = FeatureVector.feature_names()

        try:
            model_info = self.db.get_active_model("signal_model")
            if model_info:
                importance_raw = model_info.get("feature_importance")
                if importance_raw:
                    if isinstance(importance_raw, str):
                        importance = json.loads(importance_raw)
                    else:
                        importance = importance_raw

                    if isinstance(importance, dict) and importance:
                        # Sort by importance descending, take top N
                        sorted_feats = sorted(
                            importance.items(),
                            key=lambda x: float(x[1]),
                            reverse=True,
                        )
                        top_features = [f for f, _ in sorted_feats[:top_n]]
                        if top_features:
                            logger.debug(
                                f"Using top-{len(top_features)} features from model importance"
                            )
                            return top_features
        except Exception as e:
            logger.debug(f"Could not load feature importance from model: {e}")

        return all_features
