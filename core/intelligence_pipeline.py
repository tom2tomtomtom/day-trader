#!/usr/bin/env python3
"""
INTELLIGENCE PIPELINE - Unified Pipeline That Runs All Intelligence Modules

This is the enhanced orchestration layer that coordinates:
1. Standard signal analysis (existing)
2. Phantom Council evaluation (new)
3. Congressional intelligence (new)
4. Macro intelligence (new)
5. Opportunity scoring (new)
6. Trade narration (new)

Produces a comprehensive IntelligenceReport for each symbol
and a SystemBriefing for the full market.
"""

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path

from .phantom_council import PhantomCouncil, CouncilDecision
from .congressional_intel import CongressionalIntelligence, CongressionalReport
from .macro_intel import MacroIntelligence, MacroIntelReport
from .opportunity_scorer import OpportunityScorer, OpportunityScore
from .trade_narrator import TradeNarratorEngine, TradeNarrative, MarketDigest
from .db import get_db

BASE_DIR = Path(__file__).parent.parent
INTEL_REPORT_PATH = BASE_DIR / "intelligence_report.json"


@dataclass
class SymbolIntelligence:
    """Complete intelligence package for a single symbol"""
    symbol: str
    timestamp: str
    # Standard recommendation
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    # Enhanced intelligence
    opportunity_score: float
    conviction_label: str
    score_breakdown: Dict
    # Council
    council_action: str
    council_score: float
    council_conviction: str
    council_bulls: int
    council_bears: int
    persona_verdicts: List[Dict]
    # Congressional
    congress_buying: int
    congress_selling: int
    congress_conviction: float
    congress_notable: List[str]
    # Narrative
    headline: str
    thesis: str
    bull_case: str
    bear_case: str
    risk_briefing: str
    smart_money_note: str
    timing_note: str
    tags: List[str]
    # Drivers
    key_drivers: List[str]
    key_risks: List[str]
    position_size_pct: float


@dataclass
class SystemBriefing:
    """Complete system-wide intelligence briefing"""
    timestamp: str
    # Market state
    market_regime: str
    regime_score: float
    fear_greed: int
    vix: float
    macro_risk_score: float
    macro_opportunity_score: float
    # Triggers
    active_triggers: List[Dict]
    critical_triggers: int
    # Digest
    market_headline: str
    market_mood: str
    regime_narrative: str
    risk_warnings: List[str]
    smart_money_summary: str
    closing_thought: str
    # Opportunities
    top_opportunities: List[SymbolIntelligence]
    # Congressional
    congress_report_summary: Dict
    # Stats
    symbols_analyzed: int
    actionable_signals: int


class IntelligencePipeline:
    """
    Runs the complete intelligence pipeline across all modules.
    """

    def __init__(self, portfolio_value: float = 100000):
        self.council = PhantomCouncil()
        self.congress = CongressionalIntelligence()
        self.macro_intel = MacroIntelligence()
        self.scorer = OpportunityScorer()
        self.narrator = TradeNarratorEngine()
        self.portfolio_value = portfolio_value

    def analyze_symbol(self, symbol: str,
                       recommendation: Dict,
                       market_data: Dict,
                       macro_report: Optional[MacroIntelReport] = None,
                       congress_signal: Optional[Dict] = None) -> SymbolIntelligence:
        """
        Run full intelligence pipeline for a single symbol.
        """
        # 1. Run Phantom Council
        council_data = self._prepare_council_data(recommendation, market_data)
        council_decision = self.council.convene(symbol, council_data)

        # 2. Score opportunity
        congress_dict = asdict(congress_signal) if congress_signal and hasattr(congress_signal, '__dataclass_fields__') else congress_signal
        council_dict = {
            "consensus_action": council_decision.consensus_action,
            "consensus_score": council_decision.consensus_score,
            "conviction_level": council_decision.conviction_level,
            "bulls": council_decision.bulls,
            "bears": council_decision.bears,
        }
        macro_dict = None
        if macro_report:
            macro_dict = {
                "regime": asdict(macro_report.regime) if hasattr(macro_report.regime, '__dataclass_fields__') else macro_report.regime,
                "risk_score": macro_report.risk_score,
                "active_triggers": [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in macro_report.active_triggers],
            }

        opp_score = self.scorer.score(
            symbol=symbol,
            recommendation=recommendation,
            council_decision=council_dict,
            congress_intel=congress_dict,
            macro_report=macro_dict,
            market_data=market_data,
        )

        # 3. Generate narrative
        narrative = self.narrator.narrate_trade(
            symbol=symbol,
            recommendation=recommendation,
            council_decision=council_dict,
            congress_intel=congress_dict,
            market_context=market_data.get("macro", {}),
        )

        # 4. Build comprehensive intel package
        persona_verdicts = []
        for v in council_decision.verdicts:
            persona_verdicts.append({
                "name": v.persona_name,
                "style": v.persona_style,
                "stance": v.stance,
                "conviction": v.conviction,
                "reasoning": v.reasoning[:3],
                "key_metric": v.key_metric,
                "key_value": v.key_value,
                "risk_flag": v.risk_flag,
            })

        # Congressional info
        c_buying = congress_dict.get("members_buying", 0) if congress_dict else 0
        c_selling = congress_dict.get("members_selling", 0) if congress_dict else 0
        c_conviction = congress_dict.get("conviction", 0) if congress_dict else 0
        c_notable = congress_dict.get("notable_traders", []) if congress_dict else []

        return SymbolIntelligence(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=recommendation.get("action", "HOLD"),
            confidence=recommendation.get("confidence", 0),
            entry_price=recommendation.get("entry_price", 0),
            stop_loss=recommendation.get("stop_loss", 0),
            take_profit=recommendation.get("take_profit", 0),
            risk_reward=recommendation.get("risk_reward", 0),
            opportunity_score=opp_score.total_score,
            conviction_label=opp_score.conviction_label,
            score_breakdown=opp_score.weighted_breakdown,
            council_action=council_decision.consensus_action,
            council_score=council_decision.consensus_score,
            council_conviction=council_decision.conviction_level,
            council_bulls=council_decision.bulls,
            council_bears=council_decision.bears,
            persona_verdicts=persona_verdicts,
            congress_buying=c_buying,
            congress_selling=c_selling,
            congress_conviction=c_conviction,
            congress_notable=c_notable,
            headline=narrative.headline,
            thesis=narrative.thesis,
            bull_case=narrative.bull_case,
            bear_case=narrative.bear_case,
            risk_briefing=narrative.risk_briefing,
            smart_money_note=narrative.smart_money_note,
            timing_note=narrative.timing_note,
            tags=narrative.tags,
            key_drivers=opp_score.key_drivers,
            key_risks=opp_score.key_risks,
            position_size_pct=opp_score.position_size_pct,
        )

    def run_full_briefing(self, symbols: List[str],
                          recommendations: Dict[str, Dict],
                          market_context: Dict) -> SystemBriefing:
        """
        Run complete intelligence pipeline for all symbols.
        Produces a full SystemBriefing.
        """
        # 1. Run macro analysis
        macro_report = self.macro_intel.analyze(market_context)

        # 2. Fetch congressional intelligence
        self.congress.fetch_trades()
        congress_report = self.congress.generate_report()

        # Build congress signal lookup
        congress_lookup = {}
        for sig in congress_report.signals:
            congress_lookup[sig.symbol] = asdict(sig)

        # 3. Analyze each symbol
        intel_results = []
        for symbol in symbols:
            rec = recommendations.get(symbol, {
                "action": "HOLD", "confidence": 0,
                "entry_price": 0, "stop_loss": 0,
                "take_profit": 0, "risk_reward": 0,
                "reasons": [],
            })
            c_signal = congress_lookup.get(symbol)

            # Build market data for this symbol
            mdata = self._build_symbol_data(symbol, rec, market_context)

            intel = self.analyze_symbol(
                symbol=symbol,
                recommendation=rec,
                market_data=mdata,
                macro_report=macro_report,
                congress_signal=c_signal,
            )
            intel_results.append(intel)

        # Sort by opportunity score
        intel_results.sort(key=lambda x: x.opportunity_score, reverse=True)

        # 4. Generate market digest
        state = {
            "market_regime": market_context.get("market_regime", "unknown"),
            "fear_greed": market_context.get("fear_greed", 50),
            "vix": market_context.get("vix", 20),
            "recommendations": [
                {"symbol": i.symbol, "action": i.action, "confidence": i.confidence}
                for i in intel_results if i.action != "HOLD"
            ],
        }
        digest = self.narrator.generate_market_digest(
            state=state,
            congress_report=asdict(congress_report) if hasattr(congress_report, '__dataclass_fields__') else {},
        )

        # 5. Build triggers summary
        triggers = []
        for t in macro_report.active_triggers:
            triggers.append({
                "type": t.trigger_type,
                "severity": t.severity,
                "title": t.title,
                "description": t.description,
                "direction": t.signal_direction,
            })

        critical_count = len([t for t in macro_report.active_triggers if t.severity == "critical"])

        # 6. Congress summary
        congress_summary = {
            "total_trades": congress_report.total_trades_analyzed,
            "signals": len(congress_report.signals),
            "cluster_buys": len(congress_report.recent_cluster_buys),
            "hot_symbols": congress_report.hot_symbols[:5],
            "notable_activity": congress_report.notable_activity[:3],
        }

        # 7. Build briefing
        actionable = [i for i in intel_results if i.action != "HOLD" and i.opportunity_score >= 40]

        return SystemBriefing(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_regime=market_context.get("market_regime", "unknown"),
            regime_score=macro_report.regime.regime_score,
            fear_greed=market_context.get("fear_greed", 50),
            vix=market_context.get("vix", 20),
            macro_risk_score=macro_report.risk_score,
            macro_opportunity_score=macro_report.opportunity_score,
            active_triggers=triggers,
            critical_triggers=critical_count,
            market_headline=digest.headline,
            market_mood=digest.market_mood,
            regime_narrative=digest.regime_narrative,
            risk_warnings=digest.risk_warnings,
            smart_money_summary=digest.smart_money_summary,
            closing_thought=digest.closing_thought,
            top_opportunities=intel_results[:10],
            congress_report_summary=congress_summary,
            symbols_analyzed=len(symbols),
            actionable_signals=len(actionable),
        )

    def _prepare_council_data(self, recommendation: Dict,
                               market_data: Dict) -> Dict:
        """Prepare data in the format the Phantom Council expects"""
        return {
            "price_data": market_data.get("price_data", {}),
            "fundamentals": market_data.get("fundamentals", {}),
            "technicals": market_data.get("technicals", {}),
            "sentiment": market_data.get("sentiment", {}),
            "flow": market_data.get("flow", {}),
            "macro": market_data.get("macro", {}),
        }

    def _build_symbol_data(self, symbol: str, recommendation: Dict,
                           market_context: Dict) -> Dict:
        """Build comprehensive market data dict for a symbol"""
        return {
            "price_data": {
                "current_price": recommendation.get("entry_price", 0),
            },
            "fundamentals": {},
            "technicals": {
                "trend": "up" if recommendation.get("action") in ("BUY", "STRONG_BUY") else
                         "down" if recommendation.get("action") in ("SELL", "STRONG_SELL") else "sideways",
            },
            "sentiment": {
                "fear_greed": market_context.get("fear_greed", 50),
            },
            "flow": {},
            "macro": {
                "vix": market_context.get("vix", 20),
                "regime": market_context.get("market_regime", "unknown"),
            },
        }

    def save_briefing(self, briefing: SystemBriefing):
        """Save briefing to disk and Supabase for dashboard consumption."""
        data = {
            "timestamp": briefing.timestamp,
            "market": {
                "regime": briefing.market_regime,
                "regime_score": briefing.regime_score,
                "fear_greed": briefing.fear_greed,
                "vix": briefing.vix,
                "risk_score": briefing.macro_risk_score,
                "opportunity_score": briefing.macro_opportunity_score,
            },
            "triggers": briefing.active_triggers,
            "critical_triggers": briefing.critical_triggers,
            "digest": {
                "headline": briefing.market_headline,
                "mood": briefing.market_mood,
                "regime_narrative": briefing.regime_narrative,
                "risk_warnings": briefing.risk_warnings,
                "smart_money": briefing.smart_money_summary,
                "closing_thought": briefing.closing_thought,
            },
            "opportunities": [asdict(o) for o in briefing.top_opportunities],
            "congress": briefing.congress_report_summary,
            "stats": {
                "symbols_analyzed": briefing.symbols_analyzed,
                "actionable_signals": briefing.actionable_signals,
            },
        }
        # Save to local JSON
        INTEL_REPORT_PATH.write_text(json.dumps(data, indent=2, default=str))

        # Save to Supabase for dashboard access on Railway
        db = get_db()
        db.save_intelligence_briefing(data)

        return INTEL_REPORT_PATH
