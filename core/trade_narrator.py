#!/usr/bin/env python3
"""
TRADE NARRATOR - AI-Powered Narrative Explanations for Every Trade

Inspired by the Congressional Trading System's LLM story generation.
Turns raw data into compelling, human-readable trade narratives.

Features:
- Generates trade thesis narratives explaining WHY to trade
- Creates risk briefings for each position
- Produces daily market digest summaries
- Formats output for dashboard display, alerts, and social sharing

Works with or without an LLM API key:
- With API key: Uses Claude/OpenAI for rich narratives
- Without: Uses template-based narrative generation (still good!)
"""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


@dataclass
class TradeNarrative:
    """A narrative explanation for a trade recommendation"""
    symbol: str
    headline: str  # Short punchy headline
    thesis: str  # 2-3 sentence trade thesis
    bull_case: str  # Best case scenario
    bear_case: str  # Worst case scenario
    risk_briefing: str  # Key risks to monitor
    smart_money_note: str  # What institutions/insiders are doing
    timing_note: str  # Why now?
    confidence_label: str  # "High Conviction", "Moderate", "Speculative"
    tags: List[str]  # e.g. ["momentum", "value", "earnings_play"]


@dataclass
class MarketDigest:
    """Daily market digest narrative"""
    headline: str
    market_mood: str
    regime_narrative: str
    top_opportunities: List[str]
    risk_warnings: List[str]
    sector_story: str
    smart_money_summary: str
    closing_thought: str


class TradeNarratorEngine:
    """
    Generates narratives for trades and market conditions.
    Template-based by default, LLM-enhanced when API key available.
    """

    def __init__(self):
        self.use_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or
                           os.environ.get("OPENAI_API_KEY"))

    def narrate_trade(self, symbol: str, recommendation: Dict,
                      council_decision: Optional[Dict] = None,
                      congress_intel: Optional[Dict] = None,
                      market_context: Optional[Dict] = None) -> TradeNarrative:
        """
        Generate a narrative for a trade recommendation.

        Args:
            recommendation: TradeRecommendation as dict
            council_decision: PhantomCouncil decision as dict (optional)
            congress_intel: Congressional intelligence signal (optional)
            market_context: Current market context dict (optional)
        """
        action = recommendation.get("action", "HOLD")
        confidence = recommendation.get("confidence", 0)
        reasons = recommendation.get("reasons", [])
        regime = recommendation.get("regime", "unknown")
        entry = recommendation.get("entry_price", 0)
        stop = recommendation.get("stop_loss", 0)
        target = recommendation.get("take_profit", 0)
        rr = recommendation.get("risk_reward", 0)

        # Generate headline
        headline = self._generate_headline(symbol, action, confidence, reasons)

        # Generate thesis
        thesis = self._generate_thesis(symbol, action, confidence, reasons, regime)

        # Bull / Bear cases
        bull_case = self._generate_bull_case(symbol, target, entry, reasons)
        bear_case = self._generate_bear_case(symbol, stop, entry, reasons)

        # Risk briefing
        risk_briefing = self._generate_risk_briefing(
            symbol, regime, confidence, rr, market_context
        )

        # Smart money note
        smart_money = self._generate_smart_money_note(
            symbol, council_decision, congress_intel
        )

        # Timing
        timing = self._generate_timing_note(symbol, reasons, regime, market_context)

        # Confidence label
        if confidence > 0.75:
            conf_label = "High Conviction"
        elif confidence > 0.5:
            conf_label = "Moderate Conviction"
        elif confidence > 0.3:
            conf_label = "Low Conviction"
        else:
            conf_label = "Speculative"

        # Tags
        tags = self._extract_tags(action, reasons, regime)

        return TradeNarrative(
            symbol=symbol,
            headline=headline,
            thesis=thesis,
            bull_case=bull_case,
            bear_case=bear_case,
            risk_briefing=risk_briefing,
            smart_money_note=smart_money,
            timing_note=timing,
            confidence_label=conf_label,
            tags=tags,
        )

    def generate_market_digest(self, state: Dict,
                                signals: List[Dict] = None,
                                congress_report: Optional[Dict] = None) -> MarketDigest:
        """Generate daily market digest narrative"""
        regime = state.get("market_regime", "unknown")
        fg = state.get("fear_greed") or 50
        vix = state.get("vix") or 20
        recs = state.get("recommendations", [])

        headline = self._digest_headline(regime, fg, vix, recs)
        mood = self._market_mood(fg, vix)
        regime_narrative = self._regime_narrative(regime, vix)

        top_opps = []
        for rec in recs[:3]:
            sym = rec.get("symbol", "???")
            act = rec.get("action", "HOLD")
            conf = rec.get("confidence", 0)
            top_opps.append(
                f"{sym}: {act} signal at {conf:.0%} confidence"
            )

        risk_warnings = self._risk_warnings(regime, vix, fg)
        sector_story = self._sector_narrative(state)
        smart_money = self._smart_money_digest(congress_report)
        closing = self._closing_thought(regime, fg, len(recs))

        return MarketDigest(
            headline=headline,
            market_mood=mood,
            regime_narrative=regime_narrative,
            top_opportunities=top_opps,
            risk_warnings=risk_warnings,
            sector_story=sector_story,
            smart_money_summary=smart_money,
            closing_thought=closing,
        )

    # === HEADLINE GENERATORS ===

    def _generate_headline(self, symbol: str, action: str, confidence: float,
                           reasons: List[str]) -> str:
        if action in ("STRONG_BUY", "BUY"):
            if confidence > 0.75:
                return f"{symbol}: Strong Buy Signal Flashing"
            elif "Extreme Fear" in " ".join(reasons):
                return f"{symbol}: Fear Creates Opportunity"
            elif any("momentum" in r.lower() for r in reasons):
                return f"{symbol}: Momentum Building"
            else:
                return f"{symbol}: Bullish Setup Forming"
        elif action in ("STRONG_SELL", "SELL"):
            if confidence > 0.75:
                return f"{symbol}: Exit Signal Triggered"
            elif "Extreme Greed" in " ".join(reasons):
                return f"{symbol}: Greed Warning - Take Profits"
            else:
                return f"{symbol}: Bearish Pressure Mounting"
        else:
            return f"{symbol}: Waiting for Clarity"

    def _generate_thesis(self, symbol: str, action: str, confidence: float,
                         reasons: List[str], regime: str) -> str:
        reason_text = reasons[0] if reasons else "Multiple factors"

        if action in ("STRONG_BUY", "BUY"):
            return (
                f"{symbol} is showing a {action.lower().replace('_', ' ')} signal "
                f"with {confidence:.0%} confidence. {reason_text}. "
                f"The current {regime} regime supports this thesis."
            )
        elif action in ("STRONG_SELL", "SELL"):
            return (
                f"{symbol} is flashing warning signs with a {action.lower().replace('_', ' ')} "
                f"signal at {confidence:.0%} confidence. {reason_text}. "
                f"Consider reducing exposure in this {regime} environment."
            )
        else:
            return (
                f"{symbol} is in a neutral zone with no clear directional bias. "
                f"{reason_text}. "
                f"Wait for a clearer setup before committing capital."
            )

    def _generate_bull_case(self, symbol: str, target: float, entry: float,
                            reasons: List[str]) -> str:
        if target and entry and target > entry:
            upside = ((target - entry) / entry) * 100
            return (
                f"If the thesis plays out, {symbol} could reach ${target:.2f} "
                f"(+{upside:.1f}% from entry). Key drivers: "
                f"{', '.join(reasons[:2]) if reasons else 'technical momentum'}."
            )
        return f"Bullish scenario sees {symbol} breaking higher on continued momentum."

    def _generate_bear_case(self, symbol: str, stop: float, entry: float,
                            reasons: List[str]) -> str:
        if stop and entry and stop < entry:
            downside = ((entry - stop) / entry) * 100
            return (
                f"Risk is defined at ${stop:.2f} ({downside:.1f}% below entry). "
                f"A break below this level invalidates the thesis. "
                f"Key risk: macro deterioration or sector rotation."
            )
        return f"Bearish scenario sees {symbol} failing to hold support, triggering stop loss."

    def _generate_risk_briefing(self, symbol: str, regime: str, confidence: float,
                                rr: float, context: Optional[Dict]) -> str:
        risks = []
        if regime in ("high_vol", "crisis"):
            risks.append(f"elevated volatility regime ({regime})")
        if confidence < 0.5:
            risks.append("below-average conviction")
        if rr and rr < 1.5:
            risks.append(f"tight risk/reward ({rr:.1f}:1)")

        vix = (context.get("vix") or 20) if context else 20
        if vix > 25:
            risks.append(f"VIX at {vix:.0f}")

        if risks:
            return f"Monitor: {', '.join(risks)}. Size positions accordingly."
        return "Standard risk parameters apply. Follow position sizing rules."

    def _generate_smart_money_note(self, symbol: str,
                                    council: Optional[Dict],
                                    congress: Optional[Dict]) -> str:
        parts = []

        if council:
            conv = council.get("conviction_level", "")
            score = council.get("consensus_score", 0)
            bulls = council.get("bulls", 0)
            bears = council.get("bears", 0)
            parts.append(
                f"Phantom Council: {conv} ({bulls} bulls, {bears} bears, score: {score:+.0f})"
            )

        if congress:
            buying = congress.get("members_buying", 0)
            selling = congress.get("members_selling", 0)
            notable = congress.get("notable_traders", [])
            if buying > 0 or selling > 0:
                parts.append(
                    f"Congress: {buying} buying, {selling} selling"
                )
            if notable:
                parts.append(f"Notable: {', '.join(notable[:2])}")

        if parts:
            return " | ".join(parts)
        return "No significant smart money signals detected."

    def _generate_timing_note(self, symbol: str, reasons: List[str],
                              regime: str, context: Optional[Dict]) -> str:
        fg = (context.get("fear_greed") or 50) if context else 50

        if fg < 25:
            return "Extreme fear creates entry opportunities. Historical edge is strongest here."
        elif fg > 75:
            return "Extreme greed suggests caution. Consider taking partial profits."

        if regime == "trending_up":
            return "Uptrend regime favors buying dips and riding momentum."
        elif regime in ("high_vol", "crisis"):
            return "High volatility regime - reduce position sizes and widen stops."
        elif regime == "mean_reverting":
            return "Range-bound market - fade extremes and take quick profits."

        return "Standard market conditions. Follow the signals and manage risk."

    def _extract_tags(self, action: str, reasons: List[str], regime: str) -> List[str]:
        tags = []
        reasons_text = " ".join(reasons).lower()

        if "momentum" in reasons_text or "trend" in reasons_text:
            tags.append("momentum")
        if "oversold" in reasons_text or "fear" in reasons_text:
            tags.append("mean_reversion")
        if "volume" in reasons_text:
            tags.append("volume_confirm")
        if "options" in reasons_text or "flow" in reasons_text:
            tags.append("smart_money")
        if "congress" in reasons_text:
            tags.append("congressional")
        if "bollinger" in reasons_text:
            tags.append("technical")
        if "rsi" in reasons_text:
            tags.append("osc_signal")

        if action in ("STRONG_BUY", "STRONG_SELL"):
            tags.append("high_conviction")

        if regime in ("crisis", "high_vol"):
            tags.append("volatile")

        return tags or ["standard"]

    # === MARKET DIGEST HELPERS ===

    def _digest_headline(self, regime: str, fg: int, vix: float,
                         recs: List[Dict]) -> str:
        actionable = len([r for r in recs if r.get("action") != "HOLD"])

        if fg < 20:
            return f"Blood in the Streets: {actionable} Opportunities in Extreme Fear"
        elif fg > 80:
            return f"Peak Greed: {actionable} Signals Fire as Market Euphoria Builds"
        elif vix > 30:
            return f"Volatility Spike: Navigating {regime.replace('_', ' ').title()} Markets"
        elif actionable > 5:
            return f"Active Session: {actionable} Signals Detected Across Markets"
        elif actionable > 0:
            return f"Selective Setup: {actionable} Opportunity in Today's {regime.replace('_', ' ').title()} Market"
        else:
            return "Quiet Markets: No Clear Setups - Cash is a Position"

    def _market_mood(self, fg: int, vix: float) -> str:
        if fg < 20:
            mood = "Extreme Fear"
            note = "Markets are in panic mode. Historically, this is where the best entries happen."
        elif fg < 35:
            mood = "Fearful"
            note = "Sentiment is negative but not extreme. Watch for capitulation signals."
        elif fg < 55:
            mood = "Neutral"
            note = "Neither fear nor greed dominating. Follow individual stock setups."
        elif fg < 75:
            mood = "Greedy"
            note = "Optimism is high. Good for existing positions, but be selective on new entries."
        else:
            mood = "Extreme Greed"
            note = "Euphoria in markets. This is where tops form. Tighten stops and take profits."

        vix_note = ""
        if vix > 30:
            vix_note = f" VIX at {vix:.0f} confirms elevated stress."
        elif vix < 13:
            vix_note = f" VIX at {vix:.0f} signals unusual complacency."

        return f"{mood} (F&G: {fg}). {note}{vix_note}"

    def _regime_narrative(self, regime: str, vix: float) -> str:
        narratives = {
            "trending_up": "Markets are in a confirmed uptrend. Favor long positions, buy dips, and ride momentum. This is the easiest regime to trade profitably.",
            "trending_down": "Downtrend in effect. Short opportunities exist but long positions need extra caution. Reduce overall exposure.",
            "mean_reverting": "Range-bound market. Fade moves to extremes, take quick profits, and avoid breakout strategies.",
            "high_vol": "High volatility regime active. Reduce position sizes by 50%, widen stops, and be prepared for sharp reversals.",
            "crisis": "Crisis mode engaged. Capital preservation is priority one. Only trade with extreme conviction and tiny positions.",
            "low_vol": "Low volatility environment. Smaller moves but more predictable. Good for options selling strategies.",
        }
        return narratives.get(regime, f"Current regime: {regime}. Trade with standard parameters.")

    def _risk_warnings(self, regime: str, vix: float, fg: int) -> List[str]:
        warnings = []
        if vix > 25:
            warnings.append(f"VIX elevated at {vix:.0f} - size down and widen stops")
        if fg > 80:
            warnings.append("Extreme greed - high reversal probability")
        if fg < 15:
            warnings.append("Extreme fear can persist - don't go all-in on first signal")
        if regime in ("crisis", "high_vol"):
            warnings.append(f"{regime.replace('_', ' ').title()} regime - halve all position sizes")

        if not warnings:
            warnings.append("No elevated risks detected - trade with normal parameters")
        return warnings

    def _sector_narrative(self, state: Dict) -> str:
        return "Monitor sector rotation for leadership changes. Follow relative strength."

    def _smart_money_digest(self, congress: Optional[Dict]) -> str:
        if not congress:
            return "Congressional trading data not yet loaded. Run congressional scanner for insights."

        signals = congress.get("signals", [])
        clusters = congress.get("recent_cluster_buys", [])

        if clusters:
            top = clusters[0]
            return (
                f"Congressional cluster buy detected: {top.get('symbol', '???')} "
                f"({top.get('members', 0)} members, ~${top.get('volume_est', 0):,} estimated). "
                f"Total {len(signals)} congressional signals active."
            )
        elif signals:
            return f"{len(signals)} congressional trading signals active. Check Congress tab for details."
        return "No significant congressional trading activity detected recently."

    def _closing_thought(self, regime: str, fg: int, num_recs: int) -> str:
        if num_recs == 0:
            return "Sometimes the best trade is no trade. Cash is a position. Wait for high-conviction setups."
        elif fg < 25:
            return "Warren Buffett: 'Be greedy when others are fearful.' The data supports selective buying here."
        elif fg > 75:
            return "As the saying goes: 'The market can stay irrational longer than you can stay solvent.' Protect your gains."
        elif num_recs > 5:
            return "Multiple opportunities doesn't mean take them all. Pick your best 2-3 and execute with discipline."
        else:
            return "Focus on risk management above all. Protect capital first, profits follow naturally."
