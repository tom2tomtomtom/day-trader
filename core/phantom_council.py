#!/usr/bin/env python3
"""
PHANTOM COUNCIL - AI Investor Personas That Debate Every Trade

Inspired by the Phantom Forecast Tool's council concept.
Each persona represents a legendary investor archetype with distinct
analytical frameworks. They independently analyze opportunities,
then we synthesize their views into a conviction score.

Personas:
- Warren (Value): Deep value, margin of safety, moat analysis
- Michael (Contrarian): Short seller mentality, find what's broken
- Cathie (Growth): Disruptive innovation, exponential thinking
- Ray (Macro): All-weather, regime-aware, correlation-focused
- Nancy (Flow): Insider/institutional flow, follow the smart money
- Jesse (Momentum): Pure price action, trend following, tape reading

The council votes on every trade. Unanimous = highest conviction.
Split decisions = reduce size or skip.
"""

import json
import logging
import httpx
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum

from .config import get_config

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
COUNCIL_LOG = BASE_DIR / "council_decisions.json"


class Stance(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class PersonaVerdict:
    """A single persona's analysis of an opportunity"""
    persona_name: str
    persona_style: str
    stance: str  # Stance value
    conviction: float  # 0-100
    reasoning: List[str]
    key_metric: str  # The metric this persona cares most about
    key_value: str  # The value of that metric
    risk_flag: Optional[str] = None


@dataclass
class CouncilDecision:
    """The full council's synthesized decision"""
    symbol: str
    timestamp: str
    verdicts: List[PersonaVerdict]
    consensus_action: str
    consensus_score: float  # -100 to +100
    conviction_level: str  # "Unanimous", "Strong Majority", "Majority", "Split", "Contested"
    bulls: int
    bears: int
    neutrals: int
    debate_summary: str
    position_size_modifier: float  # 0.0 to 1.5
    key_risks: List[str]
    key_catalysts: List[str]


class PhantomCouncil:
    """
    Runs all personas against a symbol and synthesizes their views.
    """

    # System prompts for Claude-powered persona analysis
    PERSONA_PROMPTS = {
        "Warren": (
            "You are Warren, a deep value investor. Focus on P/E, margins, moats, "
            "dividend quality. Be skeptical of hype."
        ),
        "Michael": (
            "You are Michael, a contrarian short seller. Look for overvaluation, "
            "bubbles, sentiment extremes. Be naturally bearish."
        ),
        "Cathie": (
            "You are Cathie, a growth/innovation investor. Focus on revenue growth, "
            "TAM, disruption. Accept high valuations if growth justifies it."
        ),
        "Ray": (
            "You are Ray, a macro strategist. Focus on VIX, yield curve, regime, "
            "correlations. Think about portfolio-level risk."
        ),
        "Nancy": (
            "You are Nancy, a smart money tracker. Focus on options flow, insider "
            "buying, institutional accumulation. Follow the big money."
        ),
        "Jesse": (
            "You are Jesse, a momentum trader. Focus on trend, RSI, volume, "
            "breakouts. Price action is everything."
        ),
    }

    def __init__(self):
        self.config = get_config()
        self.personas = [
            WarrenPersona(),
            MichaelPersona(),
            CathiePersona(),
            RayPersona(),
            NancyPersona(),
            JessePersona(),
        ]

    def convene(self, symbol: str, market_data: Dict) -> CouncilDecision:
        """
        Convene the full council on a symbol.

        market_data should contain:
        - price_data: current price, history, etc.
        - fundamentals: P/E, P/B, margins, growth rates
        - technicals: RSI, MACD, BB, trend, volume
        - sentiment: fear_greed, news_sentiment, social_mentions
        - flow: options flow, institutional buys/sells
        - macro: VIX, yield curve, regime
        """
        # Try AI-powered council first (single Claude call for all 6 personas)
        verdicts = self._convene_with_claude(symbol, market_data)

        # Fall back to rule-based personas
        if verdicts is None:
            verdicts = []
            for persona in self.personas:
                verdict = persona.analyze(symbol, market_data)
                verdicts.append(verdict)

        decision = self._synthesize(symbol, verdicts, market_data)

        self._log_decision(decision)
        return decision

    def _convene_with_claude(self, symbol: str, market_data: Dict) -> Optional[List[PersonaVerdict]]:
        """
        Ask Claude to analyze the symbol as all 6 personas in a single API call.
        Returns a list of 6 PersonaVerdicts, or None if unavailable/error.
        """
        if not self.config.has_anthropic:
            return None

        persona_descriptions = "\n".join(
            f"- **{name}** ({self.personas[i].style}): {prompt}"
            for i, (name, prompt) in enumerate(self.PERSONA_PROMPTS.items())
        )

        system_prompt = (
            "You are the Phantom Council — six legendary investor archetypes who independently "
            "analyze trading opportunities. You must roleplay ALL six personas and provide each "
            "one's independent analysis.\n\n"
            f"The personas are:\n{persona_descriptions}\n\n"
            "For each persona, provide a structured analysis. You MUST respond with ONLY a JSON "
            "array of exactly 6 objects (one per persona, in the order listed above). "
            "Each object must have these fields:\n"
            '- "persona_name": string (Warren, Michael, Cathie, Ray, Nancy, or Jesse)\n'
            '- "persona_style": string (their investing style)\n'
            '- "stance": string (one of: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)\n'
            '- "conviction": integer 0-100\n'
            '- "reasoning": array of 2-4 short strings explaining the analysis\n'
            '- "key_metric": string (the single most important metric for this persona)\n'
            '- "key_value": string (the value of that metric from the data)\n'
            '- "risk_flag": string or null (primary risk concern, if any)\n\n'
            "Respond with ONLY the JSON array. No markdown, no code fences, no explanation."
        )

        user_prompt = (
            f"Analyze {symbol} for a potential trade.\n\n"
            f"Market data:\n{json.dumps(market_data, indent=2, default=str)}"
        )

        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.config.api_keys.anthropic,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning(
                    "Claude API returned status %d: %s", resp.status_code, resp.text[:200]
                )
                return None

            body = resp.json()
            raw_text = body["content"][0]["text"].strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

            parsed = json.loads(raw_text)

            if not isinstance(parsed, list) or len(parsed) != 6:
                logger.warning(
                    "Claude returned %d verdicts instead of 6, falling back",
                    len(parsed) if isinstance(parsed, list) else -1,
                )
                return None

            valid_stances = {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}
            verdicts = []
            for item in parsed:
                stance = item.get("stance", "HOLD")
                if stance not in valid_stances:
                    stance = "HOLD"

                conviction = item.get("conviction", 50)
                if not isinstance(conviction, (int, float)):
                    conviction = 50
                conviction = max(0, min(100, float(conviction)))

                reasoning = item.get("reasoning", [])
                if not isinstance(reasoning, list):
                    reasoning = [str(reasoning)]

                verdicts.append(PersonaVerdict(
                    persona_name=item.get("persona_name", "Unknown"),
                    persona_style=item.get("persona_style", "Unknown"),
                    stance=stance,
                    conviction=conviction,
                    reasoning=reasoning,
                    key_metric=item.get("key_metric", "N/A"),
                    key_value=str(item.get("key_value", "N/A")),
                    risk_flag=item.get("risk_flag"),
                ))

            logger.info("Claude-powered council convened successfully for %s", symbol)
            return verdicts

        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("Claude API call failed: %s", exc)
            return None
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
            logger.warning("Failed to parse Claude response: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected error in Claude council: %s", exc)
            return None

    def _synthesize(self, symbol: str, verdicts: List[PersonaVerdict],
                    market_data: Dict) -> CouncilDecision:
        """Synthesize individual verdicts into a council decision"""

        # Count stances
        stance_scores = {
            "STRONG_BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG_SELL": -2
        }

        scores = []
        bulls = 0
        bears = 0
        neutrals = 0
        risks = []
        catalysts = []

        for v in verdicts:
            score = stance_scores.get(v.stance, 0)
            # Weight by conviction
            weighted = score * (v.conviction / 100)
            scores.append(weighted)

            if score > 0:
                bulls += 1
                catalysts.extend([r for r in v.reasoning if not r.startswith("Risk:")])
            elif score < 0:
                bears += 1
                risks.extend([r for r in v.reasoning if r.startswith("Risk:") or "risk" in r.lower()])
            else:
                neutrals += 1

            if v.risk_flag:
                risks.append(f"{v.persona_name}: {v.risk_flag}")

        # Consensus score (-100 to +100)
        avg_score = np.mean(scores)
        consensus_score = float(np.clip(avg_score * 50, -100, 100))

        # Determine conviction level
        total = len(verdicts)
        max_side = max(bulls, bears)
        if max_side == total:
            conviction_level = "Unanimous"
        elif max_side >= total * 0.8:
            conviction_level = "Strong Majority"
        elif max_side >= total * 0.6:
            conviction_level = "Majority"
        elif bulls == bears:
            conviction_level = "Contested"
        else:
            conviction_level = "Split"

        # Determine action
        if consensus_score > 60:
            action = "STRONG_BUY"
        elif consensus_score > 25:
            action = "BUY"
        elif consensus_score < -60:
            action = "STRONG_SELL"
        elif consensus_score < -25:
            action = "SELL"
        else:
            action = "HOLD"

        # Position size modifier based on conviction
        size_modifiers = {
            "Unanimous": 1.5,
            "Strong Majority": 1.2,
            "Majority": 1.0,
            "Split": 0.5,
            "Contested": 0.3,
        }
        size_mod = size_modifiers.get(conviction_level, 0.5)

        # Generate debate summary
        summary = self._generate_debate_summary(verdicts, conviction_level, bulls, bears)

        # Deduplicate risks/catalysts
        unique_risks = list(dict.fromkeys(risks))[:5]
        unique_catalysts = list(dict.fromkeys(catalysts))[:5]

        return CouncilDecision(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            verdicts=verdicts,
            consensus_action=action,
            consensus_score=round(consensus_score, 1),
            conviction_level=conviction_level,
            bulls=bulls,
            bears=bears,
            neutrals=neutrals,
            debate_summary=summary,
            position_size_modifier=size_mod,
            key_risks=unique_risks,
            key_catalysts=unique_catalysts,
        )

    def _generate_debate_summary(self, verdicts: List[PersonaVerdict],
                                  conviction: str, bulls: int, bears: int) -> str:
        """Generate a human-readable debate summary"""
        bull_names = [v.persona_name for v in verdicts if v.stance in ("BUY", "STRONG_BUY")]
        bear_names = [v.persona_name for v in verdicts if v.stance in ("SELL", "STRONG_SELL")]
        hold_names = [v.persona_name for v in verdicts if v.stance == "HOLD"]

        parts = []
        if bull_names:
            parts.append(f"Bullish: {', '.join(bull_names)}")
        if bear_names:
            parts.append(f"Bearish: {', '.join(bear_names)}")
        if hold_names:
            parts.append(f"Neutral: {', '.join(hold_names)}")

        # Find the strongest opinion
        strongest = max(verdicts, key=lambda v: v.conviction)
        parts.append(
            f"Strongest voice: {strongest.persona_name} "
            f"({strongest.stance}, {strongest.conviction:.0f}% conviction) "
            f"- {strongest.reasoning[0] if strongest.reasoning else 'No reason given'}"
        )

        return " | ".join(parts)

    def _log_decision(self, decision: CouncilDecision):
        """Log decision for analysis"""
        try:
            existing = []
            if COUNCIL_LOG.exists():
                existing = json.loads(COUNCIL_LOG.read_text())

            entry = {
                "symbol": decision.symbol,
                "timestamp": decision.timestamp,
                "action": decision.consensus_action,
                "score": decision.consensus_score,
                "conviction": decision.conviction_level,
                "bulls": decision.bulls,
                "bears": decision.bears,
            }
            existing.append(entry)
            # Keep last 500 entries
            existing = existing[-500:]
            COUNCIL_LOG.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass


# === PERSONA IMPLEMENTATIONS ===

class BasePersona:
    """Base class for all investor personas"""

    name: str = "Base"
    style: str = "Unknown"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        raise NotImplementedError

    def _safe_get(self, data: Dict, *keys, default=None):
        """Safely navigate nested dict"""
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current


class WarrenPersona(BasePersona):
    """
    Warren - The Value Investor
    Focus: Margin of safety, competitive moats, earnings quality
    Hates: Overvaluation, no earnings, hype stocks
    """
    name = "Warren"
    style = "Value Investor"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        score = 50  # Start neutral
        reasons = []
        risk_flag = None

        fundamentals = data.get("fundamentals", {})
        technicals = data.get("technicals", {})
        price_data = data.get("price_data", {})

        # P/E analysis
        pe = fundamentals.get("pe_ratio")
        if pe is not None:
            if pe < 0:
                score -= 20
                reasons.append(f"No earnings (P/E negative) - speculative")
                risk_flag = "No earnings"
            elif pe < 12:
                score += 25
                reasons.append(f"Deep value at P/E {pe:.1f}")
            elif pe < 20:
                score += 10
                reasons.append(f"Reasonable P/E {pe:.1f}")
            elif pe > 40:
                score -= 25
                reasons.append(f"Expensive at P/E {pe:.1f}")
            elif pe > 25:
                score -= 10
                reasons.append(f"Getting pricey at P/E {pe:.1f}")
        else:
            reasons.append("No P/E data - proceed with caution")

        # Margin of safety - price vs 52-week range
        high_52w = price_data.get("high_52w")
        low_52w = price_data.get("low_52w")
        current = price_data.get("current_price", 0)
        if high_52w and low_52w and current:
            range_pct = (current - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
            if range_pct < 0.3:
                score += 15
                reasons.append(f"Near 52-week lows - potential margin of safety")
            elif range_pct > 0.9:
                score -= 10
                reasons.append(f"Near 52-week highs - limited upside")

        # Profit margins
        margin = fundamentals.get("profit_margin")
        if margin is not None:
            if margin > 0.20:
                score += 10
                reasons.append(f"Strong margins ({margin:.0%}) suggest competitive moat")
            elif margin < 0:
                score -= 15
                reasons.append(f"Negative margins - burning cash")

        # Dividend
        div_yield = fundamentals.get("dividend_yield")
        if div_yield and div_yield > 0.02:
            score += 5
            reasons.append(f"Paying {div_yield:.1%} dividend - shareholder friendly")

        # Convert to stance
        if score >= 70:
            stance = "STRONG_BUY"
        elif score >= 55:
            stance = "BUY"
        elif score <= 25:
            stance = "STRONG_SELL"
        elif score <= 40:
            stance = "SELL"
        else:
            stance = "HOLD"

        conviction = min(100, abs(score - 50) * 2 + 30)

        return PersonaVerdict(
            persona_name=self.name,
            persona_style=self.style,
            stance=stance,
            conviction=conviction,
            reasoning=reasons,
            key_metric="P/E Ratio",
            key_value=f"{pe:.1f}" if pe else "N/A",
            risk_flag=risk_flag,
        )


class MichaelPersona(BasePersona):
    """
    Michael - The Contrarian / Short Seller
    Focus: Overvaluation, fraud indicators, bubble dynamics
    Loves: Finding what's broken before the crowd
    """
    name = "Michael"
    style = "Contrarian Short Seller"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        score = 50
        reasons = []
        risk_flag = None

        fundamentals = data.get("fundamentals", {})
        technicals = data.get("technicals", {})
        sentiment = data.get("sentiment", {})
        price_data = data.get("price_data", {})

        # Overvaluation check
        pe = fundamentals.get("pe_ratio")
        if pe is not None and pe > 50:
            score -= 25
            reasons.append(f"Extreme overvaluation at P/E {pe:.0f} - bubble territory")
        elif pe is not None and pe < 8:
            score += 15
            reasons.append(f"Too cheap to ignore at P/E {pe:.0f}")

        # Sentiment extremes (contrarian)
        fg = sentiment.get("fear_greed")
        if fg is not None:
            if fg > 80:
                score -= 20
                reasons.append(f"Extreme greed ({fg}) - market complacency")
                risk_flag = "Extreme greed in market"
            elif fg < 20:
                score += 20
                reasons.append(f"Extreme fear ({fg}) - panic selling creates opportunity")

        # Social media hype (contrarian red flag)
        social = sentiment.get("social_mentions", 0)
        if social > 500:
            score -= 15
            reasons.append(f"Social hype ({social} mentions) - retail mania warning")

        # RSI extremes
        rsi = technicals.get("rsi")
        if rsi is not None:
            if rsi > 80:
                score -= 15
                reasons.append(f"RSI {rsi:.0f} - massively overbought")
            elif rsi < 20:
                score += 15
                reasons.append(f"RSI {rsi:.0f} - panic oversold, contrarian buy")

        # Short interest (if high, others see problems too)
        short_pct = fundamentals.get("short_percent")
        if short_pct and short_pct > 20:
            score -= 10
            reasons.append(f"High short interest ({short_pct:.0f}%) - smart money bearish")

        # Price vs fundamentals disconnect
        pb = fundamentals.get("pb_ratio")
        if pb is not None and pb > 10:
            score -= 15
            reasons.append(f"P/B of {pb:.1f} - completely disconnected from book value")

        if score >= 70:
            stance = "STRONG_BUY"
        elif score >= 55:
            stance = "BUY"
        elif score <= 25:
            stance = "STRONG_SELL"
        elif score <= 40:
            stance = "SELL"
        else:
            stance = "HOLD"

        conviction = min(100, abs(score - 50) * 2 + 25)

        return PersonaVerdict(
            persona_name=self.name,
            persona_style=self.style,
            stance=stance,
            conviction=conviction,
            reasoning=reasons,
            key_metric="Sentiment",
            key_value=f"F&G: {fg}" if fg else "N/A",
            risk_flag=risk_flag,
        )


class CathiePersona(BasePersona):
    """
    Cathie - The Growth/Innovation Investor
    Focus: Revenue growth, TAM, disruption potential
    Loves: High growth, new markets, exponential trends
    """
    name = "Cathie"
    style = "Growth / Innovation"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        score = 50
        reasons = []
        risk_flag = None

        fundamentals = data.get("fundamentals", {})
        technicals = data.get("technicals", {})
        price_data = data.get("price_data", {})

        # Revenue growth is king
        rev_growth = fundamentals.get("revenue_growth")
        if rev_growth is not None:
            if rev_growth > 0.40:
                score += 30
                reasons.append(f"Explosive {rev_growth:.0%} revenue growth - disruption underway")
            elif rev_growth > 0.20:
                score += 20
                reasons.append(f"Strong {rev_growth:.0%} revenue growth")
            elif rev_growth > 0.10:
                score += 5
                reasons.append(f"Moderate {rev_growth:.0%} growth")
            elif rev_growth < 0:
                score -= 20
                reasons.append(f"Revenue declining ({rev_growth:.0%}) - innovation stalled")

        # Don't care about current earnings if growth is strong
        eps_growth = fundamentals.get("earnings_growth")
        if eps_growth is not None and eps_growth > 0.25:
            score += 10
            reasons.append(f"Earnings growth {eps_growth:.0%} - approaching profitability inflection")

        # High P/E is OK if growth justifies it (PEG ratio)
        pe = fundamentals.get("pe_ratio")
        growth = fundamentals.get("earnings_growth", fundamentals.get("revenue_growth"))
        if pe and growth and growth > 0:
            peg = pe / (growth * 100)
            if peg < 1:
                score += 15
                reasons.append(f"PEG ratio {peg:.1f} - growth underpriced")
            elif peg > 3:
                score -= 10
                reasons.append(f"PEG ratio {peg:.1f} - even for growth, expensive")

        # Momentum - innovation stocks need momentum
        trend = technicals.get("trend")
        if trend in ("strong_up", "up"):
            score += 10
            reasons.append(f"Positive momentum supports innovation thesis")
        elif trend in ("strong_down",):
            score -= 5
            reasons.append(f"Downtrend, but could be accumulation opportunity")

        # Price drawdown = opportunity for growth
        high_52w = price_data.get("high_52w")
        current = price_data.get("current_price", 0)
        if high_52w and current:
            drawdown = (high_52w - current) / high_52w
            if drawdown > 0.40:
                score += 10
                reasons.append(f"Down {drawdown:.0%} from highs - potential 5-year opportunity")

        # Sector preference
        sector = fundamentals.get("sector", "")
        innovation_sectors = ["Technology", "Healthcare", "Communication Services"]
        if sector in innovation_sectors:
            score += 5
            reasons.append(f"Innovation sector: {sector}")

        if score >= 70:
            stance = "STRONG_BUY"
        elif score >= 55:
            stance = "BUY"
        elif score <= 25:
            stance = "STRONG_SELL"
        elif score <= 40:
            stance = "SELL"
        else:
            stance = "HOLD"

        conviction = min(100, abs(score - 50) * 2 + 20)

        return PersonaVerdict(
            persona_name=self.name,
            persona_style=self.style,
            stance=stance,
            conviction=conviction,
            reasoning=reasons,
            key_metric="Revenue Growth",
            key_value=f"{rev_growth:.0%}" if rev_growth else "N/A",
            risk_flag=risk_flag,
        )


class RayPersona(BasePersona):
    """
    Ray - The Macro / All-Weather Strategist
    Focus: Regime awareness, correlation, risk parity
    Cares about: VIX, yield curve, sector rotation, global macro
    """
    name = "Ray"
    style = "Macro Strategist"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        score = 50
        reasons = []
        risk_flag = None

        macro = data.get("macro", {})
        technicals = data.get("technicals", {})
        sentiment = data.get("sentiment", {})

        # VIX regime
        vix = macro.get("vix")
        if vix is not None:
            if vix > 30:
                score -= 20
                reasons.append(f"VIX at {vix:.0f} - crisis regime, reduce exposure")
                risk_flag = f"High volatility regime (VIX {vix:.0f})"
            elif vix > 20:
                score -= 5
                reasons.append(f"VIX elevated at {vix:.0f} - caution warranted")
            elif vix < 13:
                score -= 5
                reasons.append(f"VIX complacent at {vix:.0f} - watch for vol spike")
            else:
                score += 5
                reasons.append(f"VIX normal at {vix:.0f} - healthy risk environment")

        # Market regime
        regime = macro.get("regime", "unknown")
        if regime == "trending_up":
            score += 15
            reasons.append("Macro regime: risk-on trending, favor longs")
        elif regime == "high_vol":
            score -= 15
            reasons.append("Macro regime: high vol, reduce all positions")
        elif regime == "mean_reverting":
            reasons.append("Macro regime: range-bound, favor mean reversion")
        elif regime == "crisis":
            score -= 25
            reasons.append("Macro regime: crisis mode, cash is king")

        # Fear/Greed as macro indicator
        fg = sentiment.get("fear_greed")
        if fg is not None:
            if fg < 25:
                score += 10
                reasons.append(f"Extreme fear ({fg}) - macro bottom signal")
            elif fg > 75:
                score -= 10
                reasons.append(f"Extreme greed ({fg}) - macro top warning")

        # Yield curve signal
        yield_spread = macro.get("yield_spread")
        if yield_spread is not None:
            if yield_spread < 0:
                score -= 15
                reasons.append(f"Inverted yield curve ({yield_spread:.2f}%) - recession signal")
                risk_flag = "Inverted yield curve"
            elif yield_spread < 0.5:
                score -= 5
                reasons.append(f"Flat yield curve ({yield_spread:.2f}%) - late cycle")

        # Trend alignment with macro
        trend = technicals.get("trend")
        if regime == "trending_up" and trend in ("strong_up", "up"):
            score += 10
            reasons.append("Price trend aligned with macro regime")
        elif regime in ("crisis", "high_vol") and trend in ("strong_down", "down"):
            score -= 10
            reasons.append("Price trend confirms macro weakness")

        if score >= 70:
            stance = "STRONG_BUY"
        elif score >= 55:
            stance = "BUY"
        elif score <= 25:
            stance = "STRONG_SELL"
        elif score <= 40:
            stance = "SELL"
        else:
            stance = "HOLD"

        conviction = min(100, abs(score - 50) * 2 + 30)

        return PersonaVerdict(
            persona_name=self.name,
            persona_style=self.style,
            stance=stance,
            conviction=conviction,
            reasoning=reasons,
            key_metric="VIX",
            key_value=f"{vix:.0f}" if vix else "N/A",
            risk_flag=risk_flag,
        )


class NancyPersona(BasePersona):
    """
    Nancy - The Flow / Smart Money Tracker
    Focus: Options flow, institutional activity, insider/congressional trades
    Follows: Unusual options, dark pool activity, insider buys
    """
    name = "Nancy"
    style = "Smart Money / Flow"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        score = 50
        reasons = []
        risk_flag = None

        flow = data.get("flow", {})
        fundamentals = data.get("fundamentals", {})

        # Options flow
        options = flow.get("options", {})
        if options:
            pcr = options.get("put_call_ratio")
            if pcr is not None:
                if pcr < 0.5:
                    score += 20
                    reasons.append(f"Bullish options flow (P/C ratio: {pcr:.2f})")
                elif pcr > 1.5:
                    score -= 20
                    reasons.append(f"Bearish options flow (P/C ratio: {pcr:.2f})")

            if options.get("unusual_activity"):
                score += 10
                reasons.append("Unusual options activity detected - smart money moving")

            bull_prem = options.get("bullish_premium", 0)
            bear_prem = options.get("bearish_premium", 0)
            total = bull_prem + bear_prem
            if total > 0:
                bull_pct = bull_prem / total
                if bull_pct > 0.70:
                    score += 15
                    reasons.append(f"Heavy call buying ({bull_pct:.0%} of premium)")
                elif bull_pct < 0.30:
                    score -= 15
                    reasons.append(f"Heavy put buying ({1-bull_pct:.0%} of premium)")

        # Congressional trades (from congressional-trading-system idea)
        congress = flow.get("congressional", {})
        if congress:
            recent_buys = congress.get("recent_buys", 0)
            recent_sells = congress.get("recent_sells", 0)
            if recent_buys > 0 and recent_sells == 0:
                score += 15
                reasons.append(f"Congress members buying ({recent_buys} recent buys)")
            elif recent_sells > 0 and recent_buys == 0:
                score -= 15
                reasons.append(f"Congress members selling ({recent_sells} recent sells)")

            conviction_score = congress.get("conviction_score", 0)
            if conviction_score > 70:
                reasons.append(f"High congressional conviction score: {conviction_score}")

        # Insider trading
        insider = flow.get("insider", {})
        if insider:
            insider_buys = insider.get("buys_90d", 0)
            insider_sells = insider.get("sells_90d", 0)
            if insider_buys > 3 and insider_sells == 0:
                score += 15
                reasons.append(f"Cluster insider buying ({insider_buys} buys, 0 sells)")
            elif insider_sells > 5 and insider_buys == 0:
                score -= 10
                reasons.append(f"Heavy insider selling ({insider_sells} sells)")
                risk_flag = "Insider selling"

        # Institutional ownership changes
        inst = flow.get("institutional", {})
        inst_change = inst.get("ownership_change")
        if inst_change is not None:
            if inst_change > 5:
                score += 10
                reasons.append(f"Institutions accumulating (+{inst_change:.1f}% ownership)")
            elif inst_change < -5:
                score -= 10
                reasons.append(f"Institutions distributing ({inst_change:.1f}% ownership)")

        if not flow:
            reasons.append("Limited flow data - no strong signal from smart money")

        if score >= 70:
            stance = "STRONG_BUY"
        elif score >= 55:
            stance = "BUY"
        elif score <= 25:
            stance = "STRONG_SELL"
        elif score <= 40:
            stance = "SELL"
        else:
            stance = "HOLD"

        conviction = min(100, abs(score - 50) * 2 + 20)

        return PersonaVerdict(
            persona_name=self.name,
            persona_style=self.style,
            stance=stance,
            conviction=conviction,
            reasoning=reasons,
            key_metric="Options P/C Ratio",
            key_value=f"{options.get('put_call_ratio', 'N/A')}" if options else "N/A",
            risk_flag=risk_flag,
        )


class JessePersona(BasePersona):
    """
    Jesse - The Momentum / Tape Reader
    Focus: Pure price action, trend strength, breakout patterns
    Cares about: Price > fundamentals, momentum > value
    """
    name = "Jesse"
    style = "Momentum Trader"

    def analyze(self, symbol: str, data: Dict) -> PersonaVerdict:
        score = 50
        reasons = []
        risk_flag = None

        technicals = data.get("technicals", {})
        price_data = data.get("price_data", {})

        # Trend is everything
        trend = technicals.get("trend")
        trend_strength = technicals.get("trend_strength", 0)
        if trend == "strong_up":
            score += 25
            reasons.append("Strong uptrend - ride the wave")
        elif trend == "up":
            score += 15
            reasons.append("Uptrend intact - bulls in control")
        elif trend == "strong_down":
            score -= 25
            reasons.append("Strong downtrend - don't catch falling knives")
        elif trend == "down":
            score -= 15
            reasons.append("Downtrend - bears in control")
        elif trend == "sideways":
            reasons.append("Sideways chop - wait for breakout")

        # RSI momentum
        rsi = technicals.get("rsi")
        if rsi is not None:
            if 50 < rsi < 70:
                score += 10
                reasons.append(f"RSI {rsi:.0f} - healthy bullish momentum")
            elif 30 < rsi < 50:
                score -= 5
                reasons.append(f"RSI {rsi:.0f} - bearish momentum")
            elif rsi > 70:
                # For momentum traders, overbought can mean strong
                score += 5
                reasons.append(f"RSI {rsi:.0f} - powerful momentum (overbought is strong)")
            elif rsi < 30:
                score -= 10
                reasons.append(f"RSI {rsi:.0f} - momentum collapsed")

        # Volume confirmation
        rel_vol = technicals.get("relative_volume", 1.0)
        if rel_vol > 2.0:
            score += 10
            reasons.append(f"Volume surge ({rel_vol:.1f}x avg) - institutional interest")
        elif rel_vol > 1.5:
            score += 5
            reasons.append(f"Above average volume ({rel_vol:.1f}x)")
        elif rel_vol < 0.5:
            score -= 5
            reasons.append("Low volume - no conviction in move")
            risk_flag = "Low volume"

        # Bollinger Band position
        bb_pos = technicals.get("bb_position")
        if bb_pos is not None:
            if bb_pos > 0.8 and trend in ("strong_up", "up"):
                score += 5
                reasons.append("Riding upper BB - strong momentum")
            elif bb_pos < 0.2 and trend in ("strong_down", "down"):
                score -= 5
                reasons.append("Breaking lower BB - momentum selling")

        # MACD
        macd_signal = technicals.get("macd_signal")
        if macd_signal == "bullish_cross":
            score += 10
            reasons.append("MACD bullish crossover - momentum shifting up")
        elif macd_signal == "bearish_cross":
            score -= 10
            reasons.append("MACD bearish crossover - momentum shifting down")

        # 52-week breakout
        high_52w = price_data.get("high_52w")
        current = price_data.get("current_price", 0)
        if high_52w and current:
            if current >= high_52w * 0.98:
                score += 15
                reasons.append("Near 52-week high breakout - new highs = bullish")

        if score >= 70:
            stance = "STRONG_BUY"
        elif score >= 55:
            stance = "BUY"
        elif score <= 25:
            stance = "STRONG_SELL"
        elif score <= 40:
            stance = "SELL"
        else:
            stance = "HOLD"

        conviction = min(100, abs(score - 50) * 2 + 25)

        return PersonaVerdict(
            persona_name=self.name,
            persona_style=self.style,
            stance=stance,
            conviction=conviction,
            reasoning=reasons,
            key_metric="Trend",
            key_value=trend or "N/A",
            risk_flag=risk_flag,
        )


def run_council(symbol: str, market_data: Dict) -> CouncilDecision:
    """Convenience function to run the full council"""
    council = PhantomCouncil()
    return council.convene(symbol, market_data)


if __name__ == "__main__":
    # Demo with sample data
    sample_data = {
        "price_data": {
            "current_price": 185.50,
            "high_52w": 200.0,
            "low_52w": 140.0,
        },
        "fundamentals": {
            "pe_ratio": 28.5,
            "pb_ratio": 8.2,
            "profit_margin": 0.25,
            "revenue_growth": 0.08,
            "earnings_growth": 0.12,
            "dividend_yield": 0.005,
            "sector": "Technology",
        },
        "technicals": {
            "rsi": 62,
            "trend": "up",
            "trend_strength": 0.6,
            "bb_position": 0.65,
            "relative_volume": 1.3,
            "macd_signal": "bullish_cross",
        },
        "sentiment": {
            "fear_greed": 55,
            "social_mentions": 150,
        },
        "flow": {
            "options": {
                "put_call_ratio": 0.8,
                "unusual_activity": False,
                "bullish_premium": 60,
                "bearish_premium": 40,
            },
        },
        "macro": {
            "vix": 18,
            "regime": "trending_up",
            "yield_spread": 1.2,
        },
    }

    decision = run_council("AAPL", sample_data)
    print(f"\n{'='*60}")
    print(f"PHANTOM COUNCIL DECISION: {decision.symbol}")
    print(f"{'='*60}")
    print(f"Action: {decision.consensus_action}")
    print(f"Score: {decision.consensus_score}")
    print(f"Conviction: {decision.conviction_level} ({decision.bulls}B / {decision.bears}S / {decision.neutrals}H)")
    print(f"Size Modifier: {decision.position_size_modifier}x")
    print(f"\nDebate: {decision.debate_summary}")
    print(f"\nRisks: {decision.key_risks}")
    print(f"Catalysts: {decision.key_catalysts}")
    for v in decision.verdicts:
        print(f"\n  {v.persona_name} ({v.persona_style}): {v.stance} ({v.conviction:.0f}%)")
        for r in v.reasoning:
            print(f"    - {r}")
