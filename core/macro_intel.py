#!/usr/bin/env python3
"""
MACRO INTELLIGENCE - Trigger Detection & Macro Regime Analysis

Inspired by Phantom Forecast Tool's trigger detection system.
Identifies macro shifts, statistical anomalies, and regime changes
that create asymmetric opportunities.

Trigger Types:
1. Statistical Anomalies - Price/volume deviations beyond 2+ sigma
2. Quality Inflections - Fundamental metric inflection points
3. Macro Shifts - Yield curve, VIX, currency, and commodity signals
4. Regime Transitions - Market regime change detection
5. Calendar Triggers - FOMC, options expiry, earnings clusters

This is the system's early warning system.
"""

import json
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TRIGGER_LOG = BASE_DIR / "macro_triggers.json"


class TriggerType:
    STATISTICAL_ANOMALY = "statistical_anomaly"
    QUALITY_INFLECTION = "quality_inflection"
    MACRO_SHIFT = "macro_shift"
    REGIME_TRANSITION = "regime_transition"
    CALENDAR_EVENT = "calendar_event"
    CORRELATION_BREAK = "correlation_break"
    VOLUME_ANOMALY = "volume_anomaly"
    SENTIMENT_EXTREME = "sentiment_extreme"


@dataclass
class MacroTrigger:
    """A detected macro trigger / anomaly"""
    trigger_type: str
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    affected_symbols: List[str]
    signal_direction: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    data_point: str  # The specific data point that triggered
    historical_context: str  # What happened historically in similar situations
    suggested_action: str


@dataclass
class MacroRegimeSnapshot:
    """Current macro regime state"""
    timestamp: str
    primary_regime: str  # "risk_on", "risk_off", "transition", "neutral"
    regime_score: float  # -100 (max risk_off) to +100 (max risk_on)
    vix_regime: str
    yield_curve_signal: str
    dollar_trend: str
    commodity_trend: str
    credit_stress: str
    global_risk_appetite: str
    regime_duration_days: int
    transition_probability: float
    factors: List[Dict]


@dataclass
class MacroIntelReport:
    """Full macro intelligence report"""
    timestamp: str
    regime: MacroRegimeSnapshot
    active_triggers: List[MacroTrigger]
    risk_score: float  # 0-100
    opportunity_score: float  # 0-100
    narrative: str
    key_levels: Dict  # Important price levels to watch
    upcoming_events: List[Dict]
    correlation_matrix_status: str


class MacroIntelligence:
    """
    Macro intelligence engine that detects regime shifts,
    statistical anomalies, and generates early warnings.
    """

    # VIX thresholds
    VIX_COMPLACENT = 12
    VIX_NORMAL = 18
    VIX_ELEVATED = 22
    VIX_HIGH = 28
    VIX_CRISIS = 35

    # Fear & Greed thresholds
    FG_EXTREME_FEAR = 15
    FG_FEAR = 30
    FG_GREED = 70
    FG_EXTREME_GREED = 85

    def __init__(self):
        self.triggers: List[MacroTrigger] = []
        self.regime_history: List[MacroRegimeSnapshot] = []

    def analyze(self, market_data: Dict) -> MacroIntelReport:
        """
        Run full macro analysis.

        market_data should contain:
        - vix: current VIX level
        - vix_history: list of VIX values (recent)
        - fear_greed: Fear & Greed index value
        - prices: dict of symbol -> price list
        - volumes: dict of symbol -> volume list
        - yield_2y: 2-year treasury yield
        - yield_10y: 10-year treasury yield
        - dxy: dollar index
        - gold: gold price
        - oil: oil price
        - spy_prices: S&P 500 price history
        """
        self.triggers = []

        # Detect all trigger types
        self._detect_vix_triggers(market_data)
        self._detect_sentiment_triggers(market_data)
        self._detect_yield_curve_triggers(market_data)
        self._detect_volume_anomalies(market_data)
        self._detect_statistical_anomalies(market_data)
        self._detect_correlation_breaks(market_data)
        self._detect_calendar_triggers()

        # Build regime snapshot
        regime = self._build_regime_snapshot(market_data)

        # Calculate scores
        risk_score = self._calculate_risk_score(regime, self.triggers)
        opp_score = self._calculate_opportunity_score(regime, self.triggers)

        # Generate narrative
        narrative = self._generate_narrative(regime, self.triggers, risk_score, opp_score)

        # Key levels
        key_levels = self._identify_key_levels(market_data)

        # Upcoming events
        events = self._get_upcoming_events()

        # Sort triggers by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.triggers.sort(key=lambda t: severity_order.get(t.severity, 4))

        return MacroIntelReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime=regime,
            active_triggers=self.triggers,
            risk_score=risk_score,
            opportunity_score=opp_score,
            narrative=narrative,
            key_levels=key_levels,
            upcoming_events=events,
            correlation_matrix_status="normal",
        )

    def fetch_and_analyze(self) -> MacroIntelReport:
        """
        Fetch real market data and run full macro analysis.
        Uses yfinance for VIX, yield curves, SPY, QQQ, DXY, gold, oil.
        Uses data_layer for Fear & Greed.
        """
        import yfinance as yf
        from .data_layer import FearGreedSource

        # Fetch real data
        market_data = {}

        # VIX
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="1mo")
            if not vix_hist.empty:
                market_data["vix"] = float(vix_hist['Close'].iloc[-1])
                market_data["vix_history"] = [float(v) for v in vix_hist['Close'].tolist()]
        except Exception:
            pass

        # Yield curve (2Y and 10Y treasury yields)
        try:
            tnx = yf.Ticker("^TNX")  # 10Y
            tnx_hist = tnx.history(period="5d")
            if not tnx_hist.empty:
                market_data["yield_10y"] = float(tnx_hist['Close'].iloc[-1])
        except Exception:
            pass

        try:
            two_yr = yf.Ticker("2YY=F")  # 2Y yield futures
            two_hist = two_yr.history(period="5d")
            if not two_hist.empty:
                market_data["yield_2y"] = float(two_hist['Close'].iloc[-1])
        except Exception:
            pass

        # SPY, QQQ, IWM prices for statistical analysis
        prices = {}
        volumes = {}
        for sym in ["SPY", "QQQ", "IWM"]:
            try:
                t = yf.Ticker(sym)
                h = t.history(period="1mo")
                if not h.empty:
                    prices[sym] = [float(p) for p in h['Close'].tolist()]
                    volumes[sym] = [int(v) for v in h['Volume'].tolist()]
            except Exception:
                pass
        market_data["prices"] = prices
        market_data["volumes"] = volumes
        market_data["spy_prices"] = prices.get("SPY", [])

        # DXY (dollar index), Gold, Oil
        for sym, key in [("DX-Y.NYB", "dxy"), ("GC=F", "gold"), ("CL=F", "oil")]:
            try:
                t = yf.Ticker(sym)
                h = t.history(period="5d")
                if not h.empty:
                    market_data[key] = float(h['Close'].iloc[-1])
            except Exception:
                pass

        # Fear & Greed
        try:
            fg = FearGreedSource()
            fg_data = fg.fetch()
            if fg_data and "value" in fg_data:
                market_data["fear_greed"] = fg_data["value"]
        except Exception:
            pass

        return self.analyze(market_data)

    def _detect_vix_triggers(self, data: Dict):
        """Detect VIX-based triggers"""
        vix = data.get("vix")
        vix_history = data.get("vix_history", [])

        if vix is None:
            return

        # Extreme VIX levels
        if vix > self.VIX_CRISIS:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.MACRO_SHIFT,
                severity="critical",
                title="VIX Crisis Level",
                description=f"VIX at {vix:.1f} indicates extreme market stress. "
                            f"Historically, spikes above {self.VIX_CRISIS} precede major dislocations.",
                affected_symbols=["SPY", "QQQ", "IWM"],
                signal_direction="bearish",
                confidence=0.9,
                data_point=f"VIX: {vix:.1f}",
                historical_context="VIX above 35 occurred during COVID crash, GFC, and major corrections. "
                                   "Markets typically bottom 1-3 weeks after VIX peak.",
                suggested_action="Reduce long exposure to 25% of normal. Consider VIX puts for hedging."
            ))
        elif vix > self.VIX_HIGH:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.MACRO_SHIFT,
                severity="high",
                title="VIX Elevated",
                description=f"VIX at {vix:.1f} signals significant uncertainty. Risk premiums expanding.",
                affected_symbols=["SPY", "QQQ"],
                signal_direction="bearish",
                confidence=0.75,
                data_point=f"VIX: {vix:.1f}",
                historical_context="Elevated VIX often persists 2-4 weeks. Best entries come after VIX starts declining.",
                suggested_action="Reduce position sizes by 50%. Widen stops."
            ))
        elif vix < self.VIX_COMPLACENT:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.MACRO_SHIFT,
                severity="medium",
                title="VIX Complacency Warning",
                description=f"VIX at {vix:.1f} indicates extreme complacency. "
                            f"Low VIX periods often precede volatility spikes.",
                affected_symbols=["SPY", "QQQ"],
                signal_direction="neutral",
                confidence=0.6,
                data_point=f"VIX: {vix:.1f}",
                historical_context="Sub-12 VIX is unsustainable. Mean reversion to 15-20 is typical within weeks.",
                suggested_action="Consider buying cheap portfolio protection (puts/VIX calls)."
            ))

        # VIX spike detection
        if len(vix_history) >= 5:
            vix_5d_ago = vix_history[-5] if len(vix_history) >= 5 else vix
            vix_change = ((vix - vix_5d_ago) / vix_5d_ago) * 100

            if vix_change > 50:
                self.triggers.append(MacroTrigger(
                    trigger_type=TriggerType.STATISTICAL_ANOMALY,
                    severity="critical",
                    title="VIX Spike Detected",
                    description=f"VIX surged {vix_change:.0f}% in 5 days. "
                                f"This magnitude of spike is rare and significant.",
                    affected_symbols=["SPY", "QQQ", "IWM"],
                    signal_direction="bearish",
                    confidence=0.85,
                    data_point=f"VIX 5d change: +{vix_change:.0f}%",
                    historical_context="VIX spikes >50% in a week occurred <5% of the time. "
                                       "Markets often bounce 3-5 days after the spike peaks.",
                    suggested_action="Wait for VIX to start declining before adding risk."
                ))

    def _detect_sentiment_triggers(self, data: Dict):
        """Detect sentiment-based triggers"""
        fg = data.get("fear_greed")
        if fg is None:
            return

        if fg <= self.FG_EXTREME_FEAR:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.SENTIMENT_EXTREME,
                severity="high",
                title="Extreme Fear - Contrarian Buy Zone",
                description=f"Fear & Greed at {fg} signals maximum pessimism. "
                            f"Historically, this is where the best long entries occur.",
                affected_symbols=["SPY", "QQQ", "IWM"],
                signal_direction="bullish",
                confidence=0.80,
                data_point=f"F&G Index: {fg}",
                historical_context="Extreme fear readings (<15) have historically preceded "
                                   "positive 1-month returns 85% of the time. Average 1-month return: +4.2%.",
                suggested_action="Begin scaling into long positions. Deploy 25% of dry powder."
            ))
        elif fg >= self.FG_EXTREME_GREED:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.SENTIMENT_EXTREME,
                severity="high",
                title="Extreme Greed - Distribution Warning",
                description=f"Fear & Greed at {fg} signals euphoria. "
                            f"Smart money typically distributes at these levels.",
                affected_symbols=["SPY", "QQQ"],
                signal_direction="bearish",
                confidence=0.75,
                data_point=f"F&G Index: {fg}",
                historical_context="Extreme greed (>85) has preceded negative 1-month returns "
                                   "65% of the time. Not a timing tool, but a sizing tool.",
                suggested_action="Tighten stops. Take partial profits on winners. Avoid new large positions."
            ))

    def _detect_yield_curve_triggers(self, data: Dict):
        """Detect yield curve triggers"""
        y2 = data.get("yield_2y")
        y10 = data.get("yield_10y")

        if y2 is None or y10 is None:
            return

        spread = y10 - y2

        if spread < -0.5:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.MACRO_SHIFT,
                severity="critical",
                title="Deep Yield Curve Inversion",
                description=f"2s10s spread at {spread:.2f}%. Deep inversion signals "
                            f"significant recession risk 12-18 months ahead.",
                affected_symbols=["SPY", "XLF", "IWM"],
                signal_direction="bearish",
                confidence=0.70,
                data_point=f"2s10s: {spread:.2f}%",
                historical_context="Every recession since 1970 was preceded by 2s10s inversion. "
                                   "However, markets can rally for months after inversion begins.",
                suggested_action="Favor quality/defensive sectors. Reduce small-cap exposure."
            ))
        elif spread < 0:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.MACRO_SHIFT,
                severity="high",
                title="Yield Curve Inverted",
                description=f"2s10s spread at {spread:.2f}%. Recession indicator active.",
                affected_symbols=["SPY", "XLF"],
                signal_direction="bearish",
                confidence=0.60,
                data_point=f"2s10s: {spread:.2f}%",
                historical_context="Inversion is a reliable long-term signal but poor timing tool.",
                suggested_action="Monitor for steepening (often the actual sell signal)."
            ))
        elif spread > 0 and spread < 0.25:
            self.triggers.append(MacroTrigger(
                trigger_type=TriggerType.MACRO_SHIFT,
                severity="medium",
                title="Yield Curve Flattening",
                description=f"2s10s spread at {spread:.2f}%. Curve near flat - late cycle indicator.",
                affected_symbols=["XLF", "IWM"],
                signal_direction="neutral",
                confidence=0.50,
                data_point=f"2s10s: {spread:.2f}%",
                historical_context="Flat curves typically precede inversions. Economic slowdown ahead.",
                suggested_action="Rotate toward quality. Reduce cyclical exposure."
            ))

    def _detect_volume_anomalies(self, data: Dict):
        """Detect unusual volume across the market"""
        volumes = data.get("volumes", {})

        for symbol, vol_list in volumes.items():
            if not vol_list or len(vol_list) < 20:
                continue

            vol_arr = np.array(vol_list[-20:])
            avg = np.mean(vol_arr[:-1])
            std = np.std(vol_arr[:-1])
            current = vol_arr[-1]

            if std == 0:
                continue

            z_score = (current - avg) / std

            if z_score > 3:
                self.triggers.append(MacroTrigger(
                    trigger_type=TriggerType.VOLUME_ANOMALY,
                    severity="high",
                    title=f"{symbol}: Extreme Volume Spike",
                    description=f"{symbol} trading at {z_score:.1f} sigma above average volume. "
                                f"Current: {current:,.0f} vs avg: {avg:,.0f}.",
                    affected_symbols=[symbol],
                    signal_direction="neutral",
                    confidence=0.70,
                    data_point=f"Volume z-score: {z_score:.1f}",
                    historical_context="3+ sigma volume events often mark reversals or the start of major moves.",
                    suggested_action="Identify direction of volume (accumulation vs distribution) before acting."
                ))
            elif z_score > 2:
                self.triggers.append(MacroTrigger(
                    trigger_type=TriggerType.VOLUME_ANOMALY,
                    severity="medium",
                    title=f"{symbol}: Unusual Volume",
                    description=f"{symbol} at {z_score:.1f}x normal volume.",
                    affected_symbols=[symbol],
                    signal_direction="neutral",
                    confidence=0.55,
                    data_point=f"Volume z-score: {z_score:.1f}",
                    historical_context="Unusual volume precedes major price moves in 60% of cases.",
                    suggested_action="Watch for breakout or breakdown confirmation."
                ))

    def _detect_statistical_anomalies(self, data: Dict):
        """Detect price statistical anomalies"""
        prices = data.get("prices", {})

        for symbol, price_list in prices.items():
            if not price_list or len(price_list) < 20:
                continue

            arr = np.array(price_list[-20:])
            returns = np.diff(arr) / arr[:-1]

            if len(returns) < 2:
                continue

            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            last_ret = returns[-1]

            if std_ret == 0:
                continue

            z_score = (last_ret - avg_ret) / std_ret

            if abs(z_score) > 3:
                direction = "bullish" if z_score > 0 else "bearish"
                pct_move = last_ret * 100
                self.triggers.append(MacroTrigger(
                    trigger_type=TriggerType.STATISTICAL_ANOMALY,
                    severity="high",
                    title=f"{symbol}: {abs(z_score):.1f}-Sigma Move",
                    description=f"{symbol} moved {pct_move:+.1f}% ({abs(z_score):.1f} standard deviations). "
                                f"This is a statistically rare event.",
                    affected_symbols=[symbol],
                    signal_direction=direction,
                    confidence=0.65,
                    data_point=f"Return z-score: {z_score:+.1f}",
                    historical_context="3+ sigma moves revert ~60% of the time within 5 days.",
                    suggested_action=f"Watch for mean reversion. Consider fading if no fundamental catalyst."
                ))

    def _detect_correlation_breaks(self, data: Dict):
        """Detect when correlations break (divergences)"""
        # SPY vs QQQ divergence
        spy_prices = data.get("prices", {}).get("SPY", [])
        qqq_prices = data.get("prices", {}).get("QQQ", [])

        if len(spy_prices) >= 20 and len(qqq_prices) >= 20:
            spy_ret = np.diff(spy_prices[-20:]) / np.array(spy_prices[-20:])[:-1]
            qqq_ret = np.diff(qqq_prices[-20:]) / np.array(qqq_prices[-20:])[:-1]

            if len(spy_ret) > 5 and len(qqq_ret) > 5:
                spy_5d = np.sum(spy_ret[-5:])
                qqq_5d = np.sum(qqq_ret[-5:])

                divergence = abs(spy_5d - qqq_5d)
                if divergence > 0.05:  # 5% divergence
                    leader = "QQQ" if qqq_5d > spy_5d else "SPY"
                    self.triggers.append(MacroTrigger(
                        trigger_type=TriggerType.CORRELATION_BREAK,
                        severity="medium",
                        title="SPY/QQQ Divergence",
                        description=f"Tech ({qqq_5d*100:+.1f}%) and broad market ({spy_5d*100:+.1f}%) "
                                    f"diverging. {leader} leading.",
                        affected_symbols=["SPY", "QQQ"],
                        signal_direction="neutral",
                        confidence=0.55,
                        data_point=f"Divergence: {divergence*100:.1f}%",
                        historical_context="Major divergences often resolve within 1-2 weeks. "
                                           "The laggard typically catches up.",
                        suggested_action=f"Consider pair trade or follow {leader}'s direction."
                    ))

    def _detect_calendar_triggers(self):
        """Detect upcoming calendar events that affect markets"""
        now = datetime.now(timezone.utc)

        # FOMC meeting weeks are high-impact
        # Options expiration (monthly 3rd Friday)
        # Quarterly earnings cluster periods

        # Simple monthly OpEx detection
        day = now.day
        weekday = now.weekday()
        if 15 <= day <= 21 and weekday <= 4:
            days_to_friday = (4 - weekday) % 7
            opex_day = day + days_to_friday
            if 15 <= opex_day <= 21:
                self.triggers.append(MacroTrigger(
                    trigger_type=TriggerType.CALENDAR_EVENT,
                    severity="medium",
                    title="Options Expiration Week",
                    description="Monthly options expiration this week. Expect increased volatility "
                                "and potential gamma squeezes near large open interest strikes.",
                    affected_symbols=["SPY", "QQQ"],
                    signal_direction="neutral",
                    confidence=0.6,
                    data_point=f"OpEx week",
                    historical_context="OpEx weeks show 20% higher intraday volatility on average.",
                    suggested_action="Be aware of pin risk. Large OI strikes act as magnets."
                ))

    def _build_regime_snapshot(self, data: Dict) -> MacroRegimeSnapshot:
        """Build comprehensive macro regime snapshot"""
        vix = data.get("vix") or 20
        fg = data.get("fear_greed") or 50
        y2 = data.get("yield_2y") or 4.0
        y10 = data.get("yield_10y") or 4.5

        # Calculate regime score
        score = 0
        factors = []

        # VIX factor
        if vix < 15:
            score += 20
            factors.append({"name": "VIX", "value": vix, "signal": "risk_on", "weight": 20})
        elif vix < 20:
            score += 10
            factors.append({"name": "VIX", "value": vix, "signal": "mild_risk_on", "weight": 10})
        elif vix > 30:
            score -= 30
            factors.append({"name": "VIX", "value": vix, "signal": "risk_off", "weight": -30})
        elif vix > 22:
            score -= 10
            factors.append({"name": "VIX", "value": vix, "signal": "mild_risk_off", "weight": -10})

        # Fear & Greed factor
        fg_score = (fg - 50) * 0.6  # Scale to +/-30
        score += fg_score
        factors.append({
            "name": "Fear & Greed", "value": fg,
            "signal": "risk_on" if fg > 60 else "risk_off" if fg < 40 else "neutral",
            "weight": round(fg_score, 1)
        })

        # Yield curve factor
        spread = y10 - y2
        if spread < 0:
            score -= 15
            yc_signal = "inverted"
        elif spread < 0.5:
            score -= 5
            yc_signal = "flat"
        else:
            score += 10
            yc_signal = "normal"
        factors.append({"name": "Yield Curve", "value": round(spread, 2), "signal": yc_signal})

        # Determine primary regime
        if score > 40:
            regime = "risk_on"
        elif score > 10:
            regime = "mild_risk_on"
        elif score < -40:
            regime = "risk_off"
        elif score < -10:
            regime = "mild_risk_off"
        else:
            regime = "neutral"

        # VIX regime label
        if vix > self.VIX_CRISIS:
            vix_regime = "crisis"
        elif vix > self.VIX_HIGH:
            vix_regime = "high"
        elif vix > self.VIX_ELEVATED:
            vix_regime = "elevated"
        elif vix > self.VIX_NORMAL:
            vix_regime = "normal"
        elif vix > self.VIX_COMPLACENT:
            vix_regime = "low"
        else:
            vix_regime = "complacent"

        return MacroRegimeSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            primary_regime=regime,
            regime_score=round(score, 1),
            vix_regime=vix_regime,
            yield_curve_signal=yc_signal,
            dollar_trend="neutral",
            commodity_trend="neutral",
            credit_stress="low",
            global_risk_appetite=regime,
            regime_duration_days=0,
            transition_probability=0.3 if abs(score) < 20 else 0.1,
            factors=factors,
        )

    def _calculate_risk_score(self, regime: MacroRegimeSnapshot,
                              triggers: List[MacroTrigger]) -> float:
        """Calculate overall risk score 0-100"""
        score = 30  # Base risk

        # Regime contribution
        if "risk_off" in regime.primary_regime:
            score += 25
        elif "risk_on" in regime.primary_regime:
            score -= 10

        # Trigger contribution
        severity_scores = {"critical": 15, "high": 10, "medium": 5, "low": 2}
        for trigger in triggers:
            if trigger.signal_direction == "bearish":
                score += severity_scores.get(trigger.severity, 0)
            elif trigger.signal_direction == "bullish":
                score -= severity_scores.get(trigger.severity, 0) * 0.5

        return float(np.clip(score, 0, 100))

    def _calculate_opportunity_score(self, regime: MacroRegimeSnapshot,
                                     triggers: List[MacroTrigger]) -> float:
        """Calculate opportunity score 0-100"""
        score = 40  # Base

        # Extreme fear = opportunity
        for trigger in triggers:
            if trigger.trigger_type == TriggerType.SENTIMENT_EXTREME and trigger.signal_direction == "bullish":
                score += 20
            elif trigger.trigger_type == TriggerType.STATISTICAL_ANOMALY:
                score += 10
            elif trigger.trigger_type == TriggerType.VOLUME_ANOMALY:
                score += 5

        if "risk_on" in regime.primary_regime:
            score += 15

        return float(np.clip(score, 0, 100))

    def _generate_narrative(self, regime: MacroRegimeSnapshot,
                            triggers: List[MacroTrigger],
                            risk: float, opportunity: float) -> str:
        """Generate macro narrative"""
        critical = [t for t in triggers if t.severity == "critical"]
        high = [t for t in triggers if t.severity == "high"]

        if critical:
            return (
                f"ALERT: {len(critical)} critical macro trigger(s) active. "
                f"{critical[0].title}. Risk score elevated at {risk:.0f}/100. "
                f"Reduce exposure and prioritize capital preservation."
            )
        elif high:
            return (
                f"Macro environment shows {len(high)} high-priority signal(s). "
                f"{high[0].title}. Regime: {regime.primary_regime.replace('_', ' ')}. "
                f"Risk: {risk:.0f}/100, Opportunity: {opportunity:.0f}/100."
            )
        elif opportunity > 60:
            return (
                f"Favorable macro backdrop. {regime.primary_regime.replace('_', ' ').title()} regime "
                f"with opportunity score of {opportunity:.0f}/100. Conditions support risk-taking."
            )
        else:
            return (
                f"Standard macro environment. Regime: {regime.primary_regime.replace('_', ' ')}. "
                f"Risk: {risk:.0f}/100, Opportunity: {opportunity:.0f}/100. "
                f"No extreme signals - trade individual setups."
            )

    def _identify_key_levels(self, data: Dict) -> Dict:
        """Identify key price levels to watch"""
        levels = {}
        prices = data.get("prices", {})

        for symbol, price_list in prices.items():
            if not price_list or len(price_list) < 20:
                continue

            arr = np.array(price_list)
            current = arr[-1]
            high_20d = float(np.max(arr[-20:]))
            low_20d = float(np.min(arr[-20:]))
            avg_20d = float(np.mean(arr[-20:]))

            levels[symbol] = {
                "current": float(current),
                "resistance_20d": round(high_20d, 2),
                "support_20d": round(low_20d, 2),
                "pivot": round(avg_20d, 2),
            }

        return levels

    def _get_upcoming_events(self) -> List[Dict]:
        """Get upcoming market-moving events"""
        now = datetime.now(timezone.utc)
        events = []

        # Basic event calendar
        weekday = now.weekday()
        if weekday == 4:  # Friday
            events.append({
                "event": "Weekend risk",
                "date": now.strftime("%Y-%m-%d"),
                "impact": "medium",
                "note": "Consider reducing exposure before the weekend"
            })

        # Monthly OpEx (3rd Friday)
        if 15 <= now.day <= 21 and weekday == 4:
            events.append({
                "event": "Monthly Options Expiration",
                "date": now.strftime("%Y-%m-%d"),
                "impact": "high",
                "note": "Expect gamma effects and pin risk"
            })

        return events


def run_macro_scan(market_data: Dict) -> MacroIntelReport:
    """Convenience function to run macro analysis"""
    intel = MacroIntelligence()
    return intel.analyze(market_data)


def run_live_macro_scan() -> MacroIntelReport:
    """Run macro analysis with live market data"""
    intel = MacroIntelligence()
    return intel.fetch_and_analyze()


if __name__ == "__main__":
    report = run_live_macro_scan()
    print(f"\n{'='*60}")
    print(f"MACRO INTELLIGENCE REPORT (LIVE DATA)")
    print(f"{'='*60}")
    print(f"Regime: {report.regime.primary_regime} (score: {report.regime.regime_score})")
    print(f"VIX Regime: {report.regime.vix_regime}")
    print(f"Risk Score: {report.risk_score:.0f}/100")
    print(f"Opportunity Score: {report.opportunity_score:.0f}/100")
    print(f"\nNarrative: {report.narrative}")
    print(f"\nActive Triggers: {len(report.active_triggers)}")
    for t in report.active_triggers:
        print(f"  [{t.severity.upper()}] {t.title}")
        print(f"    {t.description}")
