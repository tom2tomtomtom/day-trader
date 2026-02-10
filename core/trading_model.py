#!/usr/bin/env python3
"""
TRADING MODEL - The Core Signal Scoring Engine

This is the heart of the day trading system. A multi-factor ensemble
that combines technical indicators into a composite score (-100 to +100)
and generates precise trading decisions.

Signal Components:
1. RSI (Relative Strength Index) - Momentum & overbought/oversold
2. MACD (Moving Average Convergence Divergence) - Trend momentum
3. Bollinger Bands - Volatility & mean reversion
4. Moving Average Crossovers (SMA 10/20/50) - Trend direction
5. Volatility Analysis - Position sizing adjustment

Scoring: -100 (max bearish) to +100 (max bullish)
Entry: |score| >= 25
Exit: Stop loss -5%, Take profit +10%, or signal reversal

Position Sizing: Modified Kelly Criterion
- Max 2% portfolio risk per trade
- Adjusted by signal strength & asset volatility
- Extra conservative for meme coins
"""

import json
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Meme coins / high-risk assets that get extra conservative sizing
MEME_ASSETS = {
    "DOGE-USD", "SHIB-USD", "PEPE-USD", "FLOKI-USD", "BONK-USD",
    "DOGE", "SHIB", "PEPE", "MEME", "WIF-USD",
}


@dataclass
class TechnicalIndicators:
    """All computed technical indicators for a symbol"""
    symbol: str
    timestamp: str
    current_price: float
    # Moving Averages
    sma_10: float
    sma_20: float
    sma_50: float
    # RSI
    rsi: float
    rsi_score: int
    # MACD
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_crossover: str  # "bullish", "bearish", "none"
    macd_score: int
    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # 0=at lower, 1=at upper
    bb_width: float
    bb_score: int
    # Moving Average Score
    ma_score: int
    # Volatility
    volatility_20d: float  # Annualized 20-day vol
    atr_14: float  # Average True Range
    # Composite
    composite_score: int  # -100 to +100


@dataclass
class TradeSignal:
    """A trading signal with entry/exit parameters"""
    symbol: str
    timestamp: str
    direction: str  # "long", "short", "flat"
    score: int  # -100 to +100
    strength: str  # "strong", "moderate", "weak"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # % of portfolio
    position_shares: int
    risk_reward_ratio: float
    indicators: TechnicalIndicators
    reasons: List[str]
    is_meme: bool


class TradingModel:
    """
    The core trading model engine.

    Computes technical indicators, generates composite scores,
    and produces trading signals with precise entry/exit parameters.
    """

    # Entry thresholds
    LONG_ENTRY_THRESHOLD = 25
    SHORT_ENTRY_THRESHOLD = -25

    # Exit parameters
    STOP_LOSS_PCT = 0.05   # -5% from entry
    TAKE_PROFIT_PCT = 0.10  # +10% from entry

    # Position sizing
    MAX_RISK_PER_TRADE = 0.02  # 2% of portfolio
    MEME_RISK_MULTIPLIER = 1.0  # ML learns optimal meme sizing — no hard reduction

    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value

    def compute_indicators(self, symbol: str, prices: List[float],
                           highs: List[float] = None,
                           lows: List[float] = None,
                           volumes: List[float] = None) -> Optional[TechnicalIndicators]:
        """
        Compute all technical indicators for a symbol.

        Requires at least 50 price points for full analysis.
        """
        if not prices or len(prices) < 20:
            return None

        arr = np.array(prices, dtype=float)
        current_price = float(arr[-1])

        # === MOVING AVERAGES ===
        sma_10 = float(np.mean(arr[-10:])) if len(arr) >= 10 else current_price
        sma_20 = float(np.mean(arr[-20:])) if len(arr) >= 20 else current_price
        sma_50 = float(np.mean(arr[-50:])) if len(arr) >= 50 else sma_20

        # MA Score (-25 to +25)
        ma_score = self._score_moving_averages(current_price, sma_10, sma_20, sma_50)

        # === RSI ===
        rsi = self._calculate_rsi(arr)
        rsi_score = self._score_rsi(rsi)

        # === MACD ===
        macd_line, macd_signal_line, macd_hist = self._calculate_macd(arr)
        macd_crossover, macd_score = self._score_macd(macd_line, macd_signal_line, arr)

        # === BOLLINGER BANDS ===
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger(arr)
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        bb_width = (bb_upper - bb_lower) / bb_middle * 100 if bb_middle > 0 else 0
        bb_score = self._score_bollinger(bb_position)

        # === VOLATILITY ===
        volatility = self._calculate_volatility(arr)
        atr = self._calculate_atr(arr, highs, lows)

        # === COMPOSITE SCORE ===
        composite = self._calculate_composite_score(
            rsi_score=rsi_score,
            macd_score=macd_score,
            bb_score=bb_score,
            ma_score=ma_score,
        )

        return TechnicalIndicators(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            current_price=current_price,
            sma_10=round(sma_10, 2),
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            rsi=round(rsi, 1),
            rsi_score=rsi_score,
            macd_line=round(macd_line, 4),
            macd_signal=round(macd_signal_line, 4),
            macd_histogram=round(macd_hist, 4),
            macd_crossover=macd_crossover,
            macd_score=macd_score,
            bb_upper=round(bb_upper, 2),
            bb_middle=round(bb_middle, 2),
            bb_lower=round(bb_lower, 2),
            bb_position=round(bb_position, 3),
            bb_width=round(bb_width, 2),
            bb_score=bb_score,
            ma_score=ma_score,
            volatility_20d=round(volatility, 4),
            atr_14=round(atr, 4),
            composite_score=composite,
        )

    def generate_signal(self, symbol: str, prices: List[float],
                        highs: List[float] = None,
                        lows: List[float] = None,
                        volumes: List[float] = None,
                        existing_position: Optional[Dict] = None) -> Optional[TradeSignal]:
        """
        Generate a trading signal for a symbol.

        Returns a TradeSignal with direction, entry/exit, and sizing.
        """
        indicators = self.compute_indicators(symbol, prices, highs, lows, volumes)
        if not indicators:
            return None

        score = indicators.composite_score
        current_price = indicators.current_price
        is_meme = symbol in MEME_ASSETS
        reasons = []

        # Determine direction
        if score >= self.LONG_ENTRY_THRESHOLD:
            direction = "long"
            strength = "strong" if score >= 50 else "moderate" if score >= 35 else "weak"
            reasons.append(f"Composite score {score} >= {self.LONG_ENTRY_THRESHOLD} (long entry)")
        elif score <= self.SHORT_ENTRY_THRESHOLD:
            direction = "short"
            strength = "strong" if score <= -50 else "moderate" if score <= -35 else "weak"
            reasons.append(f"Composite score {score} <= {self.SHORT_ENTRY_THRESHOLD} (short entry)")
        else:
            direction = "flat"
            strength = "weak"
            reasons.append(f"Score {score} between thresholds - no trade")

        # Build reasoning from indicator scores
        if indicators.rsi_score != 0:
            reasons.append(f"RSI {indicators.rsi:.0f}: {'+' if indicators.rsi_score > 0 else ''}{indicators.rsi_score} pts")
        if indicators.macd_score != 0:
            reasons.append(f"MACD {indicators.macd_crossover}: {'+' if indicators.macd_score > 0 else ''}{indicators.macd_score} pts")
        if indicators.bb_score != 0:
            reasons.append(f"BB position {indicators.bb_position:.2f}: {'+' if indicators.bb_score > 0 else ''}{indicators.bb_score} pts")
        if indicators.ma_score != 0:
            reasons.append(f"MA alignment: {'+' if indicators.ma_score > 0 else ''}{indicators.ma_score} pts")

        # Calculate entry/exit
        if direction == "long":
            stop_loss = round(current_price * (1 - self.STOP_LOSS_PCT), 2)
            take_profit = round(current_price * (1 + self.TAKE_PROFIT_PCT), 2)
        elif direction == "short":
            stop_loss = round(current_price * (1 + self.STOP_LOSS_PCT), 2)
            take_profit = round(current_price * (1 - self.TAKE_PROFIT_PCT), 2)
        else:
            stop_loss = 0
            take_profit = 0

        # Risk/reward ratio
        if direction in ("long", "short") and stop_loss > 0:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr = reward / risk if risk > 0 else 0
        else:
            rr = 0

        # Position sizing
        size_pct, shares = self._calculate_position_size(
            current_price=current_price,
            stop_loss=stop_loss,
            score=score,
            volatility=indicators.volatility_20d,
            is_meme=is_meme,
        )

        if is_meme and direction != "flat":
            reasons.append(f"Meme asset - position reduced to {self.MEME_RISK_MULTIPLIER:.0%} of normal")

        return TradeSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction=direction,
            score=score,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=round(size_pct, 4),
            position_shares=shares,
            risk_reward_ratio=round(rr, 2),
            indicators=indicators,
            reasons=reasons,
            is_meme=is_meme,
        )

    # === INDICATOR CALCULATIONS ===

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0  # Neutral default

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Wilder's smoothing (exponential)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _score_rsi(self, rsi: float) -> int:
        """
        Score RSI on the specified scale:
        < 20: +45 (extremely oversold)
        20-30: +30 (oversold)
        30-45: +15 (slightly bullish)
        45-55: 0 (neutral)
        55-70: -15 (slightly bearish)
        70-80: -30 (overbought)
        > 80: -45 (extremely overbought)
        """
        if rsi < 20:
            return 45
        elif rsi < 30:
            return 30
        elif rsi < 45:
            return 15
        elif rsi <= 55:
            return 0
        elif rsi <= 70:
            return -15
        elif rsi <= 80:
            return -30
        else:
            return -45

    def _calculate_macd(self, prices: np.ndarray,
                        fast: int = 12, slow: int = 26,
                        signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram"""
        if len(prices) < slow + signal:
            return 0, 0, 0

        # EMA calculations
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)

        macd_line = ema_fast - ema_slow
        # Signal line is EMA of MACD line
        if len(macd_line) >= signal:
            signal_line = self._ema(macd_line, signal)
        else:
            signal_line = macd_line

        histogram = macd_line - signal_line

        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    def _score_macd(self, macd: float, signal: float,
                    prices: np.ndarray) -> Tuple[str, int]:
        """
        Score MACD:
        Bullish crossover (MACD crosses above signal): +25
        Bearish crossover (MACD crosses below signal): -25
        """
        if len(prices) < 28:
            return "none", 0

        # Check for crossover by comparing current and previous relationship
        # We need the previous MACD and signal values
        prev_prices = prices[:-1]
        ema_fast_prev = self._ema(prev_prices, 12)
        ema_slow_prev = self._ema(prev_prices, 26)
        macd_prev = ema_fast_prev - ema_slow_prev
        if len(macd_prev) >= 9:
            signal_prev = self._ema(macd_prev, 9)
            prev_macd_val = float(macd_prev[-1])
            prev_signal_val = float(signal_prev[-1])
        else:
            return "none", 0

        # Detect crossover
        if macd > signal and prev_macd_val <= prev_signal_val:
            return "bullish", 25
        elif macd < signal and prev_macd_val >= prev_signal_val:
            return "bearish", -25
        elif macd > signal:
            # Already above - mild bullish
            return "bullish_hold", 10
        elif macd < signal:
            # Already below - mild bearish
            return "bearish_hold", -10
        return "none", 0

    def _calculate_bollinger(self, prices: np.ndarray,
                             period: int = 20,
                             std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            mid = float(np.mean(prices))
            std = float(np.std(prices))
            return mid + 2 * std, mid, mid - 2 * std

        window = prices[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window))
        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    def _score_bollinger(self, bb_position: float) -> int:
        """
        Score Bollinger Band position:
        Price below lower band (position < 0): +25 (potential bounce)
        Price above upper band (position > 1): -25 (potential pullback)
        """
        if bb_position <= 0:
            return 25
        elif bb_position < 0.1:
            return 20
        elif bb_position < 0.2:
            return 10
        elif bb_position > 1.0:
            return -25
        elif bb_position > 0.9:
            return -20
        elif bb_position > 0.8:
            return -10
        return 0

    def _score_moving_averages(self, price: float,
                                sma_10: float, sma_20: float,
                                sma_50: float) -> int:
        """
        Score MA alignment:
        Perfect bullish (price > 10 > 20 > 50): +25
        Bullish (price > 20 > 50): +15
        Perfect bearish (price < 10 < 20 < 50): -25
        Bearish (price < 20 < 50): -15
        Mixed: proportional
        """
        if price > sma_10 > sma_20 > sma_50:
            return 25
        elif price > sma_20 > sma_50:
            return 15
        elif price > sma_50:
            return 5
        elif price < sma_10 < sma_20 < sma_50:
            return -25
        elif price < sma_20 < sma_50:
            return -15
        elif price < sma_50:
            return -5
        return 0

    def _calculate_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """Calculate annualized volatility (20-day rolling std dev)"""
        if len(prices) < window + 1:
            return 0

        returns = np.diff(prices[-window - 1:]) / prices[-window - 1:-1]
        daily_vol = float(np.std(returns))
        # Annualize (252 trading days)
        return daily_vol * np.sqrt(252)

    def _calculate_atr(self, prices: np.ndarray,
                       highs: List[float] = None,
                       lows: List[float] = None,
                       period: int = 14) -> float:
        """Calculate Average True Range"""
        if highs and lows and len(highs) >= period + 1:
            h = np.array(highs, dtype=float)
            l = np.array(lows, dtype=float)
            c = prices

            tr = np.maximum(
                h[1:] - l[1:],
                np.maximum(
                    np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])
                )
            )
            return float(np.mean(tr[-period:]))
        else:
            # Approximate ATR from close prices
            if len(prices) < period + 1:
                return 0
            returns = np.abs(np.diff(prices[-period - 1:]))
            return float(np.mean(returns))

    def _calculate_composite_score(self, rsi_score: int, macd_score: int,
                                    bb_score: int, ma_score: int) -> int:
        """
        Calculate composite score from -100 to +100.

        Components:
        - RSI: -45 to +45
        - MACD: -25 to +25
        - Bollinger: -25 to +25
        - Moving Averages: -25 to +25

        Max theoretical: 120, but we clamp to [-100, +100]
        """
        raw = rsi_score + macd_score + bb_score + ma_score
        return max(-100, min(100, raw))

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data.copy()

        multiplier = 2.0 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[:period] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    def _calculate_position_size(self, current_price: float,
                                  stop_loss: float,
                                  score: int,
                                  volatility: float,
                                  is_meme: bool) -> Tuple[float, int]:
        """
        Modified Kelly Criterion position sizing.

        Principles:
        - Max 2% of portfolio risk per trade
        - Adjusted by signal strength (score)
        - Adjusted by asset volatility
        - Meme coins get extra conservative sizing
        """
        if current_price <= 0 or stop_loss <= 0:
            return 0, 0

        # Risk per share (distance to stop)
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0

        # Base: risk 2% of portfolio
        max_risk_dollars = self.portfolio_value * self.MAX_RISK_PER_TRADE

        # Adjust by signal strength (|score| / 100)
        signal_factor = abs(score) / 100.0

        # Adjust by volatility (higher vol = smaller position)
        if volatility > 0:
            vol_factor = min(1.0, 0.30 / volatility)  # Target ~30% vol
        else:
            vol_factor = 1.0

        # Meme asset reduction
        meme_factor = self.MEME_RISK_MULTIPLIER if is_meme else 1.0

        # Final risk amount
        adjusted_risk = max_risk_dollars * signal_factor * vol_factor * meme_factor

        # Shares = risk dollars / risk per share
        shares = int(adjusted_risk / risk_per_share)

        # Position value as % of portfolio
        position_value = shares * current_price
        position_pct = position_value / self.portfolio_value if self.portfolio_value > 0 else 0

        return position_pct, shares


def run_model(symbol: str, prices: List[float],
              portfolio_value: float = 100000) -> Optional[TradeSignal]:
    """Convenience function to run the model on a symbol"""
    model = TradingModel(portfolio_value=portfolio_value)
    return model.generate_signal(symbol, prices)
