#!/usr/bin/env python3
"""
REGIME ENGINE - Hidden Markov Model for market regime detection

Regimes:
1. TRENDING_UP - Strong uptrend, momentum strategies work
2. TRENDING_DOWN - Strong downtrend, short or stay out
3. RANGING - Sideways, mean reversion works
4. HIGH_VOL - Volatile, reduce size, wider stops
5. CRISIS - Extreme vol, cash is king

The regime determines WHICH strategy to use, not WHETHER to trade.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    duration_days: int
    transition_prob: Dict[str, float]
    recommended_strategy: str
    position_size_multiplier: float


class RegimeDetector:
    """
    Simplified HMM-inspired regime detector
    
    Uses observable features to estimate hidden regime:
    - Trend strength (ADX-like)
    - Volatility level (ATR / VIX)
    - Mean reversion tendency
    - Momentum persistence
    """
    
    def __init__(self):
        self.regime_history: List[Tuple[str, MarketRegime]] = []
        self.current_regime: Optional[RegimeState] = None
    
    def detect_regime(self, prices: List[float], volumes: List[float] = None,
                     vix: float = None, fear_greed: int = None) -> RegimeState:
        """
        Detect current market regime from price data
        """
        if len(prices) < 20:
            return RegimeState(
                regime=MarketRegime.RANGING,
                confidence=0.5,
                duration_days=0,
                transition_prob={},
                recommended_strategy="mean_reversion",
                position_size_multiplier=0.5
            )
        
        prices = np.array(prices)
        
        # === FEATURE EXTRACTION ===
        
        # 1. Trend Strength (simplified ADX)
        returns = np.diff(prices) / prices[:-1]
        trend_strength = abs(np.mean(returns[-20:])) / (np.std(returns[-20:]) + 0.0001)
        trend_direction = 1 if np.mean(returns[-10:]) > 0 else -1
        
        # 2. Volatility (normalized ATR)
        daily_ranges = []
        for i in range(1, min(20, len(prices))):
            daily_range = abs(prices[-i] - prices[-i-1]) / prices[-i-1]
            daily_ranges.append(daily_range)
        volatility = np.mean(daily_ranges) * 100  # As percentage
        
        # 3. Mean Reversion Score (how much price reverts to mean)
        sma_20 = np.mean(prices[-20:])
        distance_from_mean = (prices[-1] - sma_20) / sma_20
        
        # Recent reversion tendency
        above_mean = sum(1 for p in prices[-10:] if p > sma_20)
        reversion_score = 1 - abs(above_mean - 5) / 5  # 1 = perfect oscillation, 0 = trending
        
        # 4. Momentum Persistence
        short_mom = np.mean(returns[-5:])
        long_mom = np.mean(returns[-20:])
        momentum_persistence = 1 if np.sign(short_mom) == np.sign(long_mom) else 0
        
        # 5. External factors
        vix_factor = 1 if vix is None else (2 if vix > 30 else (1.5 if vix > 25 else 1))
        fg_extreme = fear_greed is not None and (fear_greed < 25 or fear_greed > 75)
        
        # === REGIME CLASSIFICATION ===
        
        # Crisis check first
        if vix and vix > 35:
            regime = MarketRegime.CRISIS
            confidence = min(0.95, 0.7 + (vix - 35) / 30)
            strategy = "cash_or_hedge"
            size_mult = 0.25
        
        # High volatility
        elif volatility > 3 or (vix and vix > 25):
            regime = MarketRegime.HIGH_VOL
            confidence = min(0.85, 0.6 + volatility / 10)
            strategy = "reduced_momentum"
            size_mult = 0.5
        
        # Strong trend up
        elif trend_strength > 0.15 and trend_direction > 0 and momentum_persistence:
            regime = MarketRegime.TRENDING_UP
            confidence = min(0.9, 0.5 + trend_strength * 2)
            strategy = "momentum_long"
            size_mult = 1.2 if not fg_extreme else 0.8  # Reduce at extremes
        
        # Strong trend down
        elif trend_strength > 0.15 and trend_direction < 0 and momentum_persistence:
            regime = MarketRegime.TRENDING_DOWN
            confidence = min(0.9, 0.5 + trend_strength * 2)
            strategy = "momentum_short_or_cash"
            size_mult = 0.7
        
        # Ranging / Mean reversion
        else:
            regime = MarketRegime.RANGING
            confidence = 0.5 + reversion_score * 0.3
            strategy = "mean_reversion"
            size_mult = 1.0 if fg_extreme else 0.8  # Boost at extremes
        
        # Calculate transition probabilities (simplified)
        transition_prob = self._estimate_transitions(regime, trend_strength, volatility)
        
        # Track duration
        duration = self._calculate_duration(regime)
        
        state = RegimeState(
            regime=regime,
            confidence=round(confidence, 3),
            duration_days=duration,
            transition_prob=transition_prob,
            recommended_strategy=strategy,
            position_size_multiplier=round(size_mult, 2)
        )
        
        self.current_regime = state
        return state
    
    def _estimate_transitions(self, current: MarketRegime, trend: float, vol: float) -> Dict[str, float]:
        """Estimate probability of transitioning to other regimes"""
        
        # Base transition matrix (simplified)
        if current == MarketRegime.TRENDING_UP:
            return {
                "stay": 0.7 - vol * 0.1,
                "to_ranging": 0.15,
                "to_down": 0.1,
                "to_high_vol": 0.05 + vol * 0.1
            }
        elif current == MarketRegime.TRENDING_DOWN:
            return {
                "stay": 0.6,
                "to_ranging": 0.2,
                "to_up": 0.1,
                "to_high_vol": 0.1
            }
        elif current == MarketRegime.RANGING:
            return {
                "stay": 0.5,
                "to_up": 0.2,
                "to_down": 0.2,
                "to_high_vol": 0.1
            }
        else:  # HIGH_VOL or CRISIS
            return {
                "stay": 0.6,
                "to_ranging": 0.25,
                "to_trending": 0.15
            }
    
    def _calculate_duration(self, regime: MarketRegime) -> int:
        """Calculate how long we've been in current regime"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for date, past_regime in reversed(self.regime_history[-30:]):
            if past_regime == regime:
                duration += 1
            else:
                break
        return duration


class StrategyRouter:
    """
    Routes to appropriate strategy based on regime
    """
    
    STRATEGY_MAP = {
        MarketRegime.TRENDING_UP: {
            "primary": "momentum_long",
            "secondary": "breakout",
            "avoid": ["mean_reversion_short", "fade"],
            "indicators": ["MA_crossover", "ADX", "momentum"],
            "stop_type": "trailing",
            "target_type": "trend_following"
        },
        MarketRegime.TRENDING_DOWN: {
            "primary": "cash",
            "secondary": "momentum_short",
            "avoid": ["buy_dip", "breakout_long"],
            "indicators": ["MA_crossover", "support_break"],
            "stop_type": "fixed",
            "target_type": "fixed"
        },
        MarketRegime.RANGING: {
            "primary": "mean_reversion",
            "secondary": "range_fade",
            "avoid": ["momentum", "breakout"],
            "indicators": ["RSI", "Bollinger", "support_resistance"],
            "stop_type": "fixed",
            "target_type": "mean"
        },
        MarketRegime.HIGH_VOL: {
            "primary": "reduced_size",
            "secondary": "volatility_fade",
            "avoid": ["momentum", "tight_stops"],
            "indicators": ["VIX", "ATR", "Bollinger_width"],
            "stop_type": "wide_atr",
            "target_type": "quick_profit"
        },
        MarketRegime.CRISIS: {
            "primary": "cash",
            "secondary": "hedge",
            "avoid": ["all_longs"],
            "indicators": ["VIX", "correlation"],
            "stop_type": "none",
            "target_type": "preservation"
        }
    }
    
    def get_strategy(self, regime: MarketRegime) -> Dict:
        """Get recommended strategy for regime"""
        return self.STRATEGY_MAP.get(regime, self.STRATEGY_MAP[MarketRegime.RANGING])
    
    def should_trade(self, regime: MarketRegime, signal_type: str) -> Tuple[bool, str]:
        """Check if a signal type is appropriate for current regime"""
        strategy = self.STRATEGY_MAP.get(regime)
        
        if signal_type in strategy.get("avoid", []):
            return False, f"Signal type '{signal_type}' not recommended in {regime.value} regime"
        
        if signal_type == strategy["primary"]:
            return True, f"Primary strategy for {regime.value}"
        elif signal_type == strategy["secondary"]:
            return True, f"Secondary strategy for {regime.value}"
        else:
            return True, f"Neutral for {regime.value}"


# Test
if __name__ == "__main__":
    import yfinance as yf
    
    # Get SPY data
    spy = yf.Ticker("SPY")
    hist = spy.history(period="3mo")
    prices = hist['Close'].tolist()
    
    # Get VIX
    vix = yf.Ticker("^VIX")
    vix_price = float(vix.history(period="1d")['Close'].iloc[-1])
    
    detector = RegimeDetector()
    regime = detector.detect_regime(prices, vix=vix_price, fear_greed=14)
    
    print(f"Current Regime: {regime.regime.value}")
    print(f"Confidence: {regime.confidence}")
    print(f"Duration: {regime.duration_days} days")
    print(f"Recommended Strategy: {regime.recommended_strategy}")
    print(f"Position Size Multiplier: {regime.position_size_multiplier}")
    print(f"Transition Probabilities: {regime.transition_prob}")
    
    router = StrategyRouter()
    strategy = router.get_strategy(regime.regime)
    print(f"\nStrategy Details:")
    print(f"  Primary: {strategy['primary']}")
    print(f"  Indicators: {strategy['indicators']}")
    print(f"  Avoid: {strategy['avoid']}")
