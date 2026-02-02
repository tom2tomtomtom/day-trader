#!/usr/bin/env python3
"""
SIGNAL ENSEMBLE - Combines multiple weak signals into strong predictions

The key insight: Individual indicators are ~55% accurate.
Combined properly, they can reach 65-75%.

Ensemble methods:
1. Weighted voting (each signal votes, weights by historical accuracy)
2. Stacking (signals become features for meta-model)
3. Confidence-weighted (only act when multiple signals agree)

We use #3 - only trade when signals CONFIRM each other.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class SignalType(Enum):
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FLOW = "flow"
    MOMENTUM = "momentum"
    REGIME = "regime"


@dataclass
class Signal:
    """Individual trading signal"""
    name: str
    signal_type: SignalType
    direction: float  # -1 to +1 (bearish to bullish)
    confidence: float  # 0 to 1
    weight: float = 1.0  # Historical accuracy weight
    reason: str = ""
    
    @property
    def weighted_score(self) -> float:
        return self.direction * self.confidence * self.weight


@dataclass
class EnsembleResult:
    """Result of ensemble signal combination"""
    final_score: float  # -1 to +1
    confidence: float  # 0 to 1
    action: str  # BUY, SELL, HOLD
    signals_agree: int  # How many signals agree
    signals_total: int
    position_size: int  # 1-5 scale
    signals: List[Signal] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)


class SignalEnsemble:
    """
    Combines multiple signals using confidence-weighted voting
    """
    
    # Default weights based on research/backtesting
    DEFAULT_WEIGHTS = {
        SignalType.TECHNICAL: 0.30,
        SignalType.SENTIMENT: 0.25,
        SignalType.FLOW: 0.25,
        SignalType.MOMENTUM: 0.20,
        SignalType.REGIME: 0.15,  # Regime is context, not signal
    }
    
    # Minimum agreement required to act
    MIN_AGREEMENT = 0.6  # 60% of weighted signals must agree
    MIN_CONFIDENCE = 0.4  # Minimum combined confidence
    
    def __init__(self, weights: Dict[SignalType, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.signal_history: List[Tuple[EnsembleResult, float]] = []  # (result, actual_pnl)
    
    def add_technical_signals(self, prices: List[float], volumes: List[float] = None) -> List[Signal]:
        """Generate technical analysis signals"""
        signals = []
        
        if len(prices) < 20:
            return signals
        
        prices = np.array(prices)
        
        # === RSI ===
        returns = np.diff(prices) / prices[:-1]
        gains = np.maximum(returns, 0)
        losses = np.abs(np.minimum(returns, 0))
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        
        if rsi < 30:
            signals.append(Signal(
                name="RSI",
                signal_type=SignalType.TECHNICAL,
                direction=0.7,
                confidence=min(0.9, (30 - rsi) / 30),
                weight=0.8,
                reason=f"RSI oversold at {rsi:.0f}"
            ))
        elif rsi > 70:
            signals.append(Signal(
                name="RSI",
                signal_type=SignalType.TECHNICAL,
                direction=-0.7,
                confidence=min(0.9, (rsi - 70) / 30),
                weight=0.8,
                reason=f"RSI overbought at {rsi:.0f}"
            ))
        
        # === Bollinger Bands ===
        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        upper_bb = sma_20 + 2 * std_20
        lower_bb = sma_20 - 2 * std_20
        current = prices[-1]
        
        bb_position = (current - lower_bb) / (upper_bb - lower_bb)
        
        if bb_position < 0.1:
            signals.append(Signal(
                name="Bollinger",
                signal_type=SignalType.TECHNICAL,
                direction=0.8,
                confidence=0.7,
                weight=0.85,
                reason="Price at lower Bollinger Band"
            ))
        elif bb_position > 0.9:
            signals.append(Signal(
                name="Bollinger",
                signal_type=SignalType.TECHNICAL,
                direction=-0.8,
                confidence=0.7,
                weight=0.85,
                reason="Price at upper Bollinger Band"
            ))
        
        # === Moving Average Trend ===
        if len(prices) >= 50:
            sma_10 = np.mean(prices[-10:])
            sma_50 = np.mean(prices[-50:])
            
            if sma_10 > sma_50 * 1.02:  # 2% above
                signals.append(Signal(
                    name="MA_Trend",
                    signal_type=SignalType.TECHNICAL,
                    direction=0.5,
                    confidence=0.6,
                    weight=0.7,
                    reason="Short MA above long MA (uptrend)"
                ))
            elif sma_10 < sma_50 * 0.98:
                signals.append(Signal(
                    name="MA_Trend",
                    signal_type=SignalType.TECHNICAL,
                    direction=-0.5,
                    confidence=0.6,
                    weight=0.7,
                    reason="Short MA below long MA (downtrend)"
                ))
        
        # === Volume Confirmation ===
        if volumes and len(volumes) >= 20:
            volumes = np.array(volumes)
            avg_vol = np.mean(volumes[-20:])
            current_vol = volumes[-1]
            rel_vol = current_vol / avg_vol
            
            price_change = (prices[-1] - prices[-2]) / prices[-2]
            
            if rel_vol > 1.5 and price_change > 0.01:
                signals.append(Signal(
                    name="Volume_Confirm",
                    signal_type=SignalType.TECHNICAL,
                    direction=0.4,
                    confidence=min(0.8, rel_vol / 3),
                    weight=0.75,
                    reason=f"Bullish move on {rel_vol:.1f}x volume"
                ))
            elif rel_vol > 1.5 and price_change < -0.01:
                signals.append(Signal(
                    name="Volume_Confirm",
                    signal_type=SignalType.TECHNICAL,
                    direction=-0.4,
                    confidence=min(0.8, rel_vol / 3),
                    weight=0.75,
                    reason=f"Bearish move on {rel_vol:.1f}x volume"
                ))
        
        return signals
    
    def add_sentiment_signals(self, fear_greed: int = None, 
                             reddit_mentions: int = None,
                             news_sentiment: float = None) -> List[Signal]:
        """Generate sentiment-based signals"""
        signals = []
        
        # Fear & Greed (contrarian)
        if fear_greed is not None:
            if fear_greed < 20:
                signals.append(Signal(
                    name="Fear_Greed",
                    signal_type=SignalType.SENTIMENT,
                    direction=0.9,  # Strong buy at extreme fear
                    confidence=0.85,
                    weight=0.9,
                    reason=f"Extreme Fear ({fear_greed}) - contrarian buy"
                ))
            elif fear_greed < 35:
                signals.append(Signal(
                    name="Fear_Greed",
                    signal_type=SignalType.SENTIMENT,
                    direction=0.5,
                    confidence=0.6,
                    weight=0.9,
                    reason=f"Fear ({fear_greed}) - lean bullish"
                ))
            elif fear_greed > 80:
                signals.append(Signal(
                    name="Fear_Greed",
                    signal_type=SignalType.SENTIMENT,
                    direction=-0.9,  # Strong sell at extreme greed
                    confidence=0.85,
                    weight=0.9,
                    reason=f"Extreme Greed ({fear_greed}) - contrarian sell"
                ))
            elif fear_greed > 65:
                signals.append(Signal(
                    name="Fear_Greed",
                    signal_type=SignalType.SENTIMENT,
                    direction=-0.5,
                    confidence=0.6,
                    weight=0.9,
                    reason=f"Greed ({fear_greed}) - lean bearish"
                ))
        
        return signals
    
    def add_flow_signals(self, bullish_premium: float = None,
                        bearish_premium: float = None) -> List[Signal]:
        """Generate options flow signals"""
        signals = []
        
        if bullish_premium is not None and bearish_premium is not None:
            total = bullish_premium + bearish_premium
            if total > 0:
                ratio = bullish_premium / total
                
                if ratio > 0.7:
                    signals.append(Signal(
                        name="Options_Flow",
                        signal_type=SignalType.FLOW,
                        direction=0.7,
                        confidence=ratio,
                        weight=0.95,  # Flow is very predictive
                        reason=f"Heavy call buying ({ratio:.0%} bullish)"
                    ))
                elif ratio < 0.3:
                    signals.append(Signal(
                        name="Options_Flow",
                        signal_type=SignalType.FLOW,
                        direction=-0.7,
                        confidence=1 - ratio,
                        weight=0.95,
                        reason=f"Heavy put buying ({1-ratio:.0%} bearish)"
                    ))
        
        return signals
    
    def combine_signals(self, signals: List[Signal]) -> EnsembleResult:
        """
        Combine all signals into final decision
        """
        if not signals:
            return EnsembleResult(
                final_score=0,
                confidence=0,
                action="HOLD",
                signals_agree=0,
                signals_total=0,
                position_size=0,
                reasoning=["No signals available"]
            )
        
        # Group by type and calculate weighted scores
        type_scores = {}
        type_confidences = {}
        
        for signal in signals:
            st = signal.signal_type
            if st not in type_scores:
                type_scores[st] = []
                type_confidences[st] = []
            type_scores[st].append(signal.weighted_score)
            type_confidences[st].append(signal.confidence)
        
        # Calculate weighted average for each type
        weighted_sum = 0
        total_weight = 0
        
        for st, scores in type_scores.items():
            type_weight = self.weights.get(st, 0.2)
            type_avg = np.mean(scores)
            weighted_sum += type_avg * type_weight
            total_weight += type_weight
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate agreement
        bullish = sum(1 for s in signals if s.direction > 0)
        bearish = sum(1 for s in signals if s.direction < 0)
        agreement = max(bullish, bearish) / len(signals)
        
        # Calculate confidence
        avg_confidence = np.mean([s.confidence for s in signals])
        combined_confidence = avg_confidence * agreement
        
        # Determine action
        if combined_confidence < self.MIN_CONFIDENCE:
            action = "HOLD"
            reasoning = [f"Low confidence ({combined_confidence:.2f} < {self.MIN_CONFIDENCE})"]
        elif agreement < self.MIN_AGREEMENT:
            action = "HOLD"
            reasoning = [f"Low agreement ({agreement:.0%} < {self.MIN_AGREEMENT:.0%})"]
        elif final_score > 0.4:
            action = "STRONG_BUY"
            reasoning = [f"Strong bullish consensus (score: {final_score:.2f})"]
        elif final_score > 0.2:
            action = "BUY"
            reasoning = [f"Bullish (score: {final_score:.2f})"]
        elif final_score < -0.4:
            action = "STRONG_SELL"
            reasoning = [f"Strong bearish consensus (score: {final_score:.2f})"]
        elif final_score < -0.2:
            action = "SELL"
            reasoning = [f"Bearish (score: {final_score:.2f})"]
        else:
            action = "HOLD"
            reasoning = [f"Neutral (score: {final_score:.2f})"]
        
        # Position size (1-5)
        position_size = min(5, max(1, int(abs(final_score) * 10 * combined_confidence)))
        if action == "HOLD":
            position_size = 0
        
        # Add signal reasons
        reasoning.extend([f"{s.name}: {s.reason}" for s in signals])
        
        return EnsembleResult(
            final_score=round(final_score, 3),
            confidence=round(combined_confidence, 3),
            action=action,
            signals_agree=max(bullish, bearish),
            signals_total=len(signals),
            position_size=position_size,
            signals=signals,
            reasoning=reasoning
        )
    
    def update_weights(self, result: EnsembleResult, actual_pnl: float):
        """
        Update signal weights based on actual outcomes (learning)
        """
        self.signal_history.append((result, actual_pnl))
        
        # Need enough history to learn
        if len(self.signal_history) < 20:
            return
        
        # Calculate accuracy for each signal type
        type_accuracy = {st: [] for st in SignalType}
        
        for past_result, pnl in self.signal_history[-100:]:
            for signal in past_result.signals:
                correct = (signal.direction > 0 and pnl > 0) or (signal.direction < 0 and pnl < 0)
                type_accuracy[signal.signal_type].append(1 if correct else 0)
        
        # Update weights based on accuracy
        for st, accuracies in type_accuracy.items():
            if accuracies:
                new_weight = np.mean(accuracies)
                # Smoothed update
                self.weights[st] = self.weights.get(st, 0.2) * 0.8 + new_weight * 0.2


# Test
if __name__ == "__main__":
    ensemble = SignalEnsemble()
    
    # Simulate some data
    prices = list(range(100, 95, -1)) + list(range(95, 100, 1))  # V-shaped
    prices = [p + np.random.randn() * 0.5 for p in prices]
    
    signals = []
    signals.extend(ensemble.add_technical_signals(prices))
    signals.extend(ensemble.add_sentiment_signals(fear_greed=14))  # Current extreme fear
    
    result = ensemble.combine_signals(signals)
    
    print(f"Action: {result.action}")
    print(f"Score: {result.final_score}")
    print(f"Confidence: {result.confidence}")
    print(f"Position Size: {result.position_size}/5")
    print(f"Agreement: {result.signals_agree}/{result.signals_total}")
    print(f"\nReasoning:")
    for r in result.reasoning:
        print(f"  - {r}")
