#!/usr/bin/env python3
"""
ADVANCED OPPORTUNITY SCORER - Multi-Factor Conviction Scoring

Inspired by both the Phantom Forecast Tool's opportunity scoring and
the Congressional Trading System's conviction scoring algorithm.

Combines ALL intelligence sources into a single 0-100 conviction score:
1. Technical Score (25%) - Price action, momentum, volume
2. Sentiment Score (15%) - Fear/greed, social, news
3. Smart Money Score (20%) - Options flow, congressional trades, insider
4. Council Score (20%) - Phantom Council consensus
5. Macro Score (10%) - Regime alignment, VIX, macro triggers
6. Quality Score (10%) - Fundamentals, earnings, moat

The resulting score determines position sizing and priority:
- 80-100: Maximum conviction - full position size
- 60-79: High conviction - standard position
- 40-59: Moderate - half position
- 20-39: Low - paper trade only
- 0-19: No trade
"""

import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCORES_LOG = BASE_DIR / "opportunity_scores.json"


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of each scoring dimension"""
    technical: float = 0  # 0-100 raw score
    sentiment: float = 0
    smart_money: float = 0
    council: float = 0
    macro: float = 0
    quality: float = 0


@dataclass
class OpportunityScore:
    """Complete opportunity score for a symbol"""
    symbol: str
    timestamp: str
    total_score: float  # 0-100 weighted composite
    conviction_label: str  # "Maximum", "High", "Moderate", "Low", "No Trade"
    direction: str  # "long", "short", "neutral"
    breakdown: ScoreBreakdown
    weighted_breakdown: Dict  # Shows weighted contribution of each factor
    key_drivers: List[str]  # Top 3 drivers
    key_risks: List[str]  # Top 3 risks
    position_size_pct: float  # Suggested position size as % of portfolio
    edge_estimate: float  # Estimated edge in %


# Weights for each scoring dimension
WEIGHTS = {
    "technical": 0.25,
    "sentiment": 0.15,
    "smart_money": 0.20,
    "council": 0.20,
    "macro": 0.10,
    "quality": 0.10,
}


class OpportunityScorer:
    """
    Combines all intelligence sources into a single conviction score.
    """

    def __init__(self, weights: Dict = None):
        self.weights = weights or WEIGHTS

    def score(self, symbol: str,
              recommendation: Optional[Dict] = None,
              council_decision: Optional[Dict] = None,
              congress_intel: Optional[Dict] = None,
              macro_report: Optional[Dict] = None,
              market_data: Optional[Dict] = None) -> OpportunityScore:
        """
        Score an opportunity using all available data sources.
        """
        breakdown = ScoreBreakdown()
        drivers = []
        risks = []

        # 1. Technical Score
        breakdown.technical, tech_notes = self._score_technical(
            recommendation, market_data
        )
        drivers.extend(tech_notes.get("drivers", []))
        risks.extend(tech_notes.get("risks", []))

        # 2. Sentiment Score
        breakdown.sentiment, sent_notes = self._score_sentiment(
            market_data
        )
        drivers.extend(sent_notes.get("drivers", []))
        risks.extend(sent_notes.get("risks", []))

        # 3. Smart Money Score
        breakdown.smart_money, sm_notes = self._score_smart_money(
            congress_intel, market_data
        )
        drivers.extend(sm_notes.get("drivers", []))
        risks.extend(sm_notes.get("risks", []))

        # 4. Council Score
        breakdown.council, council_notes = self._score_council(
            council_decision
        )
        drivers.extend(council_notes.get("drivers", []))
        risks.extend(council_notes.get("risks", []))

        # 5. Macro Score
        breakdown.macro, macro_notes = self._score_macro(
            macro_report, market_data
        )
        drivers.extend(macro_notes.get("drivers", []))
        risks.extend(macro_notes.get("risks", []))

        # 6. Quality Score
        breakdown.quality, quality_notes = self._score_quality(
            market_data
        )
        drivers.extend(quality_notes.get("drivers", []))
        risks.extend(quality_notes.get("risks", []))

        # Calculate weighted total
        total = (
            breakdown.technical * self.weights["technical"] +
            breakdown.sentiment * self.weights["sentiment"] +
            breakdown.smart_money * self.weights["smart_money"] +
            breakdown.council * self.weights["council"] +
            breakdown.macro * self.weights["macro"] +
            breakdown.quality * self.weights["quality"]
        )

        # Weighted breakdown for UI
        weighted = {
            "technical": round(breakdown.technical * self.weights["technical"], 1),
            "sentiment": round(breakdown.sentiment * self.weights["sentiment"], 1),
            "smart_money": round(breakdown.smart_money * self.weights["smart_money"], 1),
            "council": round(breakdown.council * self.weights["council"], 1),
            "macro": round(breakdown.macro * self.weights["macro"], 1),
            "quality": round(breakdown.quality * self.weights["quality"], 1),
        }

        # Conviction label
        if total >= 80:
            label = "Maximum"
        elif total >= 60:
            label = "High"
        elif total >= 40:
            label = "Moderate"
        elif total >= 20:
            label = "Low"
        else:
            label = "No Trade"

        # Direction
        action = recommendation.get("action", "HOLD") if recommendation else "HOLD"
        if action in ("BUY", "STRONG_BUY"):
            direction = "long"
        elif action in ("SELL", "STRONG_SELL"):
            direction = "short"
        else:
            direction = "neutral"

        # Position size
        if label == "Maximum":
            size_pct = 5.0
        elif label == "High":
            size_pct = 3.0
        elif label == "Moderate":
            size_pct = 1.5
        elif label == "Low":
            size_pct = 0.5
        else:
            size_pct = 0.0

        # Edge estimate (rough)
        edge = max(0, (total - 50) * 0.1)  # 0-5% edge

        # Deduplicate and trim
        key_drivers = list(dict.fromkeys(drivers))[:3]
        key_risks = list(dict.fromkeys(risks))[:3]

        score = OpportunityScore(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_score=round(total, 1),
            conviction_label=label,
            direction=direction,
            breakdown=breakdown,
            weighted_breakdown=weighted,
            key_drivers=key_drivers,
            key_risks=key_risks,
            position_size_pct=size_pct,
            edge_estimate=round(edge, 2),
        )

        self._log_score(score)
        return score

    def _score_technical(self, rec: Optional[Dict],
                         data: Optional[Dict]) -> tuple:
        """Score technical factors (0-100)"""
        score = 50
        notes = {"drivers": [], "risks": []}

        if not rec:
            return score, notes

        # Signal confidence
        confidence = rec.get("confidence", 0)
        score = confidence * 100

        # Adjust based on signal strength
        action = rec.get("action", "HOLD")
        if action == "STRONG_BUY":
            score = min(100, score + 15)
            notes["drivers"].append("Strong bullish technical setup")
        elif action == "BUY":
            score = min(100, score + 5)
            notes["drivers"].append("Bullish technical signals")
        elif action == "STRONG_SELL":
            score = min(100, score + 15)
            notes["drivers"].append("Strong bearish technical setup")
        elif action == "SELL":
            score = min(100, score + 5)
        elif action == "HOLD":
            score = max(20, score - 20)
            notes["risks"].append("No clear technical direction")

        # Risk/reward
        rr = rec.get("risk_reward", 0)
        if rr > 3:
            score = min(100, score + 10)
            notes["drivers"].append(f"Excellent risk/reward ({rr:.1f}:1)")
        elif rr < 1:
            score = max(0, score - 15)
            notes["risks"].append(f"Poor risk/reward ({rr:.1f}:1)")

        # Technical data
        technicals = data.get("technicals", {}) if data else {}
        rsi = technicals.get("rsi")
        if rsi and (rsi < 25 or rsi > 75):
            notes["drivers"].append(f"RSI extreme ({rsi:.0f})")

        return max(0, min(100, score)), notes

    def _score_sentiment(self, data: Optional[Dict]) -> tuple:
        """Score sentiment factors (0-100)"""
        score = 50
        notes = {"drivers": [], "risks": []}

        if not data:
            return score, notes

        sentiment = data.get("sentiment", {})
        fg = sentiment.get("fear_greed")

        if fg is not None:
            if fg < 20:
                score = 85
                notes["drivers"].append(f"Extreme fear ({fg}) - contrarian buy zone")
            elif fg < 35:
                score = 65
                notes["drivers"].append(f"Fear ({fg}) - bullish lean")
            elif fg > 80:
                score = 85  # High score because it's actionable (sell signal)
                notes["risks"].append(f"Extreme greed ({fg}) - reversal risk")
            elif fg > 65:
                score = 35
                notes["risks"].append(f"Greed ({fg}) - caution on new longs")
            else:
                score = 50

        social = sentiment.get("social_mentions", 0)
        if social > 500:
            notes["risks"].append(f"High social attention ({social} mentions)")

        return max(0, min(100, score)), notes

    def _score_smart_money(self, congress: Optional[Dict],
                           data: Optional[Dict]) -> tuple:
        """Score smart money factors (0-100)"""
        score = 50
        notes = {"drivers": [], "risks": []}

        # Congressional intelligence
        if congress:
            buying = congress.get("members_buying", 0)
            selling = congress.get("members_selling", 0)
            conviction = congress.get("conviction", 0)
            notable = congress.get("notable_traders", [])

            if buying > 2 and selling == 0:
                score += 25
                notes["drivers"].append(f"Congressional cluster buy ({buying} members)")
            elif buying > 0:
                score += 10
            elif selling > 2 and buying == 0:
                score -= 15
                notes["risks"].append(f"Congress selling ({selling} members)")

            if conviction > 70:
                score += 10
                notes["drivers"].append(f"High conviction congressional trade ({conviction:.0f})")

            if notable:
                score += 5
                notes["drivers"].append(f"Notable traders: {', '.join(notable[:2])}")

        # Options flow
        flow = data.get("flow", {}).get("options", {}) if data else {}
        if flow:
            pcr = flow.get("put_call_ratio")
            if pcr is not None:
                if pcr < 0.5:
                    score += 15
                    notes["drivers"].append("Bullish options flow")
                elif pcr > 1.5:
                    score -= 10
                    notes["risks"].append("Bearish options flow")

            if flow.get("unusual_activity"):
                score += 10
                notes["drivers"].append("Unusual options activity")

        return max(0, min(100, score)), notes

    def _score_council(self, council: Optional[Dict]) -> tuple:
        """Score Phantom Council consensus (0-100)"""
        score = 50
        notes = {"drivers": [], "risks": []}

        if not council:
            return score, notes

        consensus = council.get("consensus_score", 0)
        conviction = council.get("conviction_level", "")
        bulls = council.get("bulls", 0)
        bears = council.get("bears", 0)

        # Map consensus score (-100 to +100) to (0 to 100)
        score = (consensus + 100) / 2

        if conviction == "Unanimous":
            score = min(100, score + 10)
            notes["drivers"].append(f"Unanimous council ({bulls}B/{bears}S)")
        elif conviction == "Strong Majority":
            score = min(100, score + 5)
            notes["drivers"].append(f"Strong council majority ({bulls}B/{bears}S)")
        elif conviction == "Contested":
            score = max(0, score - 10)
            notes["risks"].append(f"Council split ({bulls}B/{bears}S)")
        elif conviction == "Split":
            notes["risks"].append("Council opinion divided")

        return max(0, min(100, score)), notes

    def _score_macro(self, macro: Optional[Dict],
                     data: Optional[Dict]) -> tuple:
        """Score macro alignment (0-100)"""
        score = 50
        notes = {"drivers": [], "risks": []}

        if macro:
            regime = macro.get("regime", {})
            regime_name = regime.get("primary_regime", "neutral") if isinstance(regime, dict) else "neutral"

            if "risk_on" in regime_name:
                score += 20
                notes["drivers"].append("Risk-on macro regime")
            elif "risk_off" in regime_name:
                score -= 20
                notes["risks"].append("Risk-off macro regime")

            risk_score = macro.get("risk_score", 50)
            if risk_score > 70:
                score -= 15
                notes["risks"].append(f"Elevated macro risk ({risk_score:.0f}/100)")
            elif risk_score < 30:
                score += 10

            # Active triggers
            triggers = macro.get("active_triggers", [])
            critical = [t for t in triggers if isinstance(t, dict) and t.get("severity") == "critical"]
            if critical:
                score -= 20
                notes["risks"].append(f"{len(critical)} critical macro trigger(s)")

        # VIX from market data
        if data:
            vix = data.get("macro", {}).get("vix")
            if vix and vix > 30:
                score -= 15
                notes["risks"].append(f"VIX at {vix:.0f}")
            elif vix and vix < 15:
                score += 5

        return max(0, min(100, score)), notes

    def _score_quality(self, data: Optional[Dict]) -> tuple:
        """Score fundamental quality (0-100)"""
        score = 50
        notes = {"drivers": [], "risks": []}

        if not data:
            return score, notes

        fundamentals = data.get("fundamentals", {})

        pe = fundamentals.get("pe_ratio")
        if pe is not None:
            if 5 < pe < 20:
                score += 15
                notes["drivers"].append(f"Reasonable valuation (P/E {pe:.0f})")
            elif pe > 50:
                score -= 10
                notes["risks"].append(f"Expensive (P/E {pe:.0f})")
            elif pe < 0:
                score -= 10
                notes["risks"].append("No earnings")

        margin = fundamentals.get("profit_margin")
        if margin and margin > 0.20:
            score += 10
            notes["drivers"].append(f"High profit margin ({margin:.0%})")
        elif margin and margin < 0:
            score -= 10

        growth = fundamentals.get("revenue_growth")
        if growth and growth > 0.20:
            score += 10
            notes["drivers"].append(f"Strong revenue growth ({growth:.0%})")
        elif growth and growth < -0.10:
            score -= 10
            notes["risks"].append("Revenue declining")

        return max(0, min(100, score)), notes

    def _log_score(self, score: OpportunityScore):
        """Log score for tracking"""
        try:
            existing = []
            if SCORES_LOG.exists():
                existing = json.loads(SCORES_LOG.read_text())

            entry = {
                "symbol": score.symbol,
                "timestamp": score.timestamp,
                "total": score.total_score,
                "label": score.conviction_label,
                "direction": score.direction,
            }
            existing.append(entry)
            existing = existing[-500:]
            SCORES_LOG.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    def score_universe(self, symbols: List[str],
                       recommendations: Dict = None,
                       council_decisions: Dict = None,
                       congress_signals: Dict = None,
                       macro_report: Optional[Dict] = None,
                       market_data_map: Dict = None) -> List[OpportunityScore]:
        """Score entire universe and return sorted results"""
        scores = []
        for symbol in symbols:
            rec = recommendations.get(symbol) if recommendations else None
            council = council_decisions.get(symbol) if council_decisions else None
            congress = congress_signals.get(symbol) if congress_signals else None
            mdata = market_data_map.get(symbol) if market_data_map else None

            opp = self.score(
                symbol=symbol,
                recommendation=rec,
                council_decision=council,
                congress_intel=congress,
                macro_report=macro_report,
                market_data=mdata,
            )
            scores.append(opp)

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores
