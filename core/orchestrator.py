#!/usr/bin/env python3
"""
MAIN ORCHESTRATOR - The brain that ties everything together

Flow:
1. Fetch data from all sources
2. Detect market regime
3. Route to appropriate strategy
4. Generate signals via ensemble
5. Size position via risk engine
6. Execute or recommend

This is the single entry point for the trading system.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .data_layer import DataAggregator
from .regime_engine import RegimeDetector, StrategyRouter, MarketRegime
from .signal_ensemble import SignalEnsemble, SignalType
from .risk_engine import RiskEngine
from .intelligence_pipeline import IntelligencePipeline

BASE_DIR = Path(__file__).parent.parent
ORCHESTRATOR_LOG = BASE_DIR / "orchestrator_log.jsonl"


@dataclass
class TradeRecommendation:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    position_size: int  # 1-5 scale
    shares: int
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasons: List[str]
    regime: str
    timestamp: str


@dataclass
class SystemState:
    timestamp: str
    market_regime: str
    regime_confidence: float
    fear_greed: int
    vix: float
    portfolio_value: float
    portfolio_heat: float
    positions_count: int
    recommendations: List[TradeRecommendation]


class TradingOrchestrator:
    """
    Main orchestrator that coordinates all components
    """

    def __init__(self, portfolio_value: float = 100000):
        self.data = DataAggregator()
        self.regime_detector = RegimeDetector()
        self.strategy_router = StrategyRouter()
        self.signal_ensemble = SignalEnsemble()
        self.risk_engine = RiskEngine(portfolio_value)
        self.intelligence = IntelligencePipeline(portfolio_value)

        self.last_state: Optional[SystemState] = None
    
    def analyze_symbol(self, symbol: str) -> TradeRecommendation:
        """
        Full analysis pipeline for a single symbol
        """
        # 1. Get all data
        data = self.data.get_full_picture(symbol)
        
        if not data.get("price"):
            return TradeRecommendation(
                symbol=symbol,
                action="HOLD",
                confidence=0,
                position_size=0,
                shares=0,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                risk_reward=0,
                reasons=["No price data available"],
                regime="unknown",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        price_data = data["price"]
        fg_data = data.get("fear_greed", {})
        options_data = data.get("options_flow", {})
        
        # 2. Detect regime
        regime_state = self.regime_detector.detect_regime(
            prices=price_data.get("prices", []),
            vix=fg_data.get("vix"),
            fear_greed=fg_data.get("value")
        )
        
        # 3. Generate signals
        signals = []
        
        # Technical signals
        signals.extend(self.signal_ensemble.add_technical_signals(
            prices=price_data.get("prices", []),
            volumes=price_data.get("volumes")
        ))
        
        # Sentiment signals
        signals.extend(self.signal_ensemble.add_sentiment_signals(
            fear_greed=fg_data.get("value")
        ))
        
        # Options flow signals (if available)
        if options_data.get("available"):
            signals.extend(self.signal_ensemble.add_flow_signals(
                bullish_premium=options_data.get("bullish_premium"),
                bearish_premium=options_data.get("bearish_premium")
            ))
        
        # 4. Combine signals
        ensemble_result = self.signal_ensemble.combine_signals(signals)
        
        # 5. Check if strategy is appropriate for regime
        can_trade, regime_note = self.strategy_router.should_trade(
            regime_state.regime,
            "momentum" if ensemble_result.action in ["BUY", "STRONG_BUY"] else "mean_reversion"
        )
        
        # Adjust confidence based on regime
        adjusted_confidence = ensemble_result.confidence * regime_state.position_size_multiplier
        
        # 6. Calculate position size
        entry_price = price_data.get("current_price", 0)
        
        risk_metrics = self.risk_engine.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_pct=0.02 if regime_state.regime != MarketRegime.HIGH_VOL else 0.04,
            confidence=adjusted_confidence,
            win_rate=0.55,  # Conservative estimate
            avg_win_pct=3.0,
            avg_loss_pct=2.0
        )
        
        # 7. Build recommendation
        action = ensemble_result.action if can_trade and risk_metrics.can_trade else "HOLD"
        
        reasons = ensemble_result.reasoning.copy()
        reasons.append(f"Regime: {regime_state.regime.value} ({regime_state.recommended_strategy})")
        reasons.append(regime_note)
        if not risk_metrics.can_trade:
            reasons.append(risk_metrics.reason)
        
        return TradeRecommendation(
            symbol=symbol,
            action=action,
            confidence=round(adjusted_confidence, 3),
            position_size=ensemble_result.position_size if action != "HOLD" else 0,
            shares=risk_metrics.recommended_shares if action != "HOLD" else 0,
            entry_price=round(entry_price, 2),
            stop_loss=risk_metrics.stop_loss_price,
            take_profit=risk_metrics.take_profit_price,
            risk_reward=risk_metrics.risk_reward_ratio,
            reasons=reasons,
            regime=regime_state.regime.value,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def scan_universe(self, symbols: List[str]) -> SystemState:
        """
        Scan entire universe and return system state with recommendations
        """
        # Get market context
        context = self.data.get_market_context()
        
        # Analyze each symbol
        recommendations = []
        for symbol in symbols:
            rec = self.analyze_symbol(symbol)
            recommendations.append(rec)
        
        # Sort by confidence and action
        actionable = [r for r in recommendations if r.action != "HOLD"]
        actionable.sort(key=lambda x: x.confidence, reverse=True)
        
        # Get portfolio summary
        portfolio = self.risk_engine.get_portfolio_summary()
        
        # Build state
        state = SystemState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_regime=context.get("market_regime", "unknown"),
            regime_confidence=0.7,  # TODO: from regime detector
            fear_greed=context.get("fear_greed", {}).get("value", 50),
            vix=context.get("vix", 20),
            portfolio_value=portfolio["portfolio_value"],
            portfolio_heat=portfolio["portfolio_heat"],
            positions_count=portfolio["num_positions"],
            recommendations=actionable[:10]  # Top 10
        )
        
        self.last_state = state
        self._log_state(state)
        
        return state
    
    def run_intelligence_briefing(self, symbols: List[str]):
        """
        Run the FULL intelligence pipeline including:
        - Phantom Council analysis per symbol
        - Congressional trade intelligence
        - Macro trigger detection
        - Multi-factor opportunity scoring
        - AI-powered trade narratives

        Returns a comprehensive SystemBriefing saved to disk.
        """
        # First, run standard analysis to get recommendations
        state = self.scan_universe(symbols)

        # Build recommendations dict
        recommendations = {}
        for rec in state.recommendations:
            recommendations[rec.symbol] = asdict(rec)

        # Also include HOLD symbols with basic data
        for symbol in symbols:
            if symbol not in recommendations:
                data = self.data.get_full_picture(symbol)
                recommendations[symbol] = {
                    "action": "HOLD",
                    "confidence": 0,
                    "entry_price": data.get("price", {}).get("current_price", 0),
                    "stop_loss": 0,
                    "take_profit": 0,
                    "risk_reward": 0,
                    "reasons": ["No signal"],
                }

        # Build market context
        context = self.data.get_market_context()
        market_context = {
            "market_regime": state.market_regime,
            "fear_greed": state.fear_greed,
            "vix": state.vix,
        }

        # Run the full intelligence pipeline
        briefing = self.intelligence.run_full_briefing(
            symbols=symbols,
            recommendations=recommendations,
            market_context=market_context,
        )

        # Save to disk for dashboard
        self.intelligence.save_briefing(briefing)

        return briefing

    def _log_state(self, state: SystemState):
        """Log state for analysis"""
        with open(ORCHESTRATOR_LOG, "a") as f:
            log_entry = {
                "timestamp": state.timestamp,
                "regime": state.market_regime,
                "fear_greed": state.fear_greed,
                "vix": state.vix,
                "recommendations": len(state.recommendations),
                "top_pick": state.recommendations[0].symbol if state.recommendations else None,
            }
            f.write(json.dumps(log_entry) + "\n")
    
    def print_report(self, state: SystemState):
        """Print human-readable report"""
        print("\n" + "="*70)
        print("🤖 TRADING SYSTEM REPORT")
        print(f"   {state.timestamp}")
        print("="*70)
        
        # Market context
        fg_emoji = "😱" if state.fear_greed < 25 else "😰" if state.fear_greed < 45 else "😐" if state.fear_greed < 55 else "😊" if state.fear_greed < 75 else "🤑"
        print(f"\n📊 MARKET CONTEXT")
        print(f"   Regime: {state.market_regime.upper()}")
        print(f"   {fg_emoji} Fear & Greed: {state.fear_greed}")
        print(f"   📉 VIX: {state.vix:.1f}")
        
        # Portfolio
        print(f"\n💰 PORTFOLIO")
        print(f"   Value: ${state.portfolio_value:,.2f}")
        print(f"   Heat: {state.portfolio_heat:.1f}%")
        print(f"   Positions: {state.positions_count}")
        
        # Recommendations
        if state.recommendations:
            print(f"\n🎯 TOP RECOMMENDATIONS ({len(state.recommendations)})")
            for i, rec in enumerate(state.recommendations[:5], 1):
                action_emoji = "🟢" if "BUY" in rec.action else "🔴"
                print(f"\n   {i}. {action_emoji} {rec.symbol} - {rec.action}")
                print(f"      Price: ${rec.entry_price:.2f} | Confidence: {rec.confidence:.0%}")
                print(f"      Stop: ${rec.stop_loss} | Target: ${rec.take_profit} | R:R {rec.risk_reward}")
                print(f"      Size: {rec.position_size}/5 ({rec.shares} shares)")
                print(f"      {rec.reasons[0]}")
        else:
            print(f"\n⏸️  No actionable signals - market in wait mode")
        
        print()


# Default universe
DEFAULT_UNIVERSE = [
    # Major ETFs
    "SPY", "QQQ", "IWM",
    # Top stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Memes
    "DOGE-USD", "MSTR", "COIN"
]


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trading System Orchestrator")
    parser.add_argument("--symbol", "-s", help="Analyze single symbol")
    parser.add_argument("--portfolio", "-p", type=float, default=100000, help="Portfolio value")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--intel", action="store_true", help="Run full intelligence briefing")
    args = parser.parse_args()

    orchestrator = TradingOrchestrator(portfolio_value=args.portfolio)

    if args.intel:
        print("Running full intelligence pipeline...")
        briefing = orchestrator.run_intelligence_briefing(DEFAULT_UNIVERSE)
        print(f"\n{'='*70}")
        print(f"INTELLIGENCE BRIEFING")
        print(f"{'='*70}")
        print(f"\n{briefing.market_headline}")
        print(f"\n{briefing.market_mood}")
        print(f"\n{briefing.regime_narrative}")
        print(f"\nRisk Score: {briefing.macro_risk_score:.0f}/100")
        print(f"Opportunity Score: {briefing.macro_opportunity_score:.0f}/100")
        if briefing.active_triggers:
            print(f"\nActive Triggers ({len(briefing.active_triggers)}):")
            for t in briefing.active_triggers[:5]:
                print(f"  [{t['severity'].upper()}] {t['title']}")
        print(f"\nTop Opportunities ({briefing.actionable_signals} actionable):")
        for opp in briefing.top_opportunities[:5]:
            print(f"  {opp.symbol}: {opp.action} | Score: {opp.opportunity_score:.0f}/100 ({opp.conviction_label})")
            print(f"    Council: {opp.council_action} ({opp.council_conviction}) | Congress: {opp.congress_buying}B/{opp.congress_selling}S")
            print(f"    {opp.headline}")
        if briefing.congress_report_summary:
            print(f"\nCongressional Intel: {briefing.congress_report_summary.get('signals', 0)} signals")
        print(f"\n{briefing.closing_thought}")
        print(f"\nFull report saved to intelligence_report.json")
    elif args.symbol:
        rec = orchestrator.analyze_symbol(args.symbol)
        if args.json:
            print(json.dumps(asdict(rec), indent=2))
        else:
            print(f"\n{rec.symbol}: {rec.action}")
            print(f"Confidence: {rec.confidence:.0%}")
            print(f"Entry: ${rec.entry_price} | Stop: ${rec.stop_loss} | Target: ${rec.take_profit}")
            print(f"Shares: {rec.shares} | R:R: {rec.risk_reward}")
            print(f"Regime: {rec.regime}")
            print(f"\nReasons:")
            for r in rec.reasons:
                print(f"  - {r}")
    else:
        state = orchestrator.scan_universe(DEFAULT_UNIVERSE)
        if args.json:
            print(json.dumps({
                "timestamp": state.timestamp,
                "regime": state.market_regime,
                "fear_greed": state.fear_greed,
                "vix": state.vix,
                "portfolio_value": state.portfolio_value,
                "recommendations": [asdict(r) for r in state.recommendations]
            }, indent=2))
        else:
            orchestrator.print_report(state)
