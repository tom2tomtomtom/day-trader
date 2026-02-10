#!/usr/bin/env python3
"""
FEATURE ENGINE — Unified feature computation for ML pipeline.

Combines technical indicators, sentiment, regime, volume profile,
and time features into a single feature vector for ML model input
and trade logging.
"""

import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class FeatureVector:
    """Complete feature vector for a symbol at a point in time."""
    symbol: str
    timestamp: str

    # Technical indicators
    rsi_14: float = 50.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_position: float = 0.5      # 0 = at lower band, 1 = at upper band
    bb_width: float = 0.0
    sma_10_dist: float = 0.0      # % distance from SMA 10
    sma_20_dist: float = 0.0
    sma_50_dist: float = 0.0
    ma_alignment: int = 0         # -25 to +25

    # Volatility
    volatility_20d: float = 0.0   # Annualized
    atr_14: float = 0.0
    atr_pct: float = 0.0          # ATR as % of price

    # Volume
    relative_volume: float = 1.0   # Current / 20-day avg
    volume_trend: float = 0.0     # Slope of volume over 5 days

    # Momentum
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_20d: float = 0.0
    price_vs_high_52w: float = 0.0  # % from 52-week high
    price_vs_low_52w: float = 0.0   # % from 52-week low

    # Regime / Sentiment
    regime: str = "unknown"
    regime_confidence: float = 0.0
    fear_greed: int = 50
    fear_greed_trend: str = "stable"

    # Time features
    day_of_week: int = 0          # 0=Mon, 4=Fri
    hour_of_day: int = 12
    is_market_open: bool = True

    # Multi-timeframe (hourly)
    hourly_rsi: float = float('nan')           # RSI computed on hourly bars
    hourly_macd_signal: float = float('nan')   # MACD signal line on hourly bars
    hourly_trend_alignment: float = 0.0        # 1.0 = hourly agrees with daily, -1.0 = opposed

    # Composite scores
    composite_score: int = 0       # -100 to +100
    ensemble_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_ml_array(self) -> np.ndarray:
        """Convert to numeric array for ML model input."""
        regime_map = {
            "trending_up": 1, "trending_down": -1, "ranging": 0,
            "high_vol": 2, "crisis": -2, "unknown": 0,
            "NORMAL": 0, "ELEVATED_VOL": 1, "EXTREME_FEAR": -1,
            "EXTREME_GREED": 1, "CRISIS": -2,
        }
        fg_trend_map = {"improving": 1, "declining": -1, "stable": 0}

        # For ML array, replace NaN with 0.0 so models don't choke
        hourly_rsi = self.hourly_rsi if not np.isnan(self.hourly_rsi) else 0.0
        hourly_macd = self.hourly_macd_signal if not np.isnan(self.hourly_macd_signal) else 0.0

        return np.array([
            self.rsi_14,
            self.macd_line,
            self.macd_signal,
            self.macd_histogram,
            self.bb_position,
            self.bb_width,
            self.sma_10_dist,
            self.sma_20_dist,
            self.sma_50_dist,
            self.ma_alignment,
            self.volatility_20d,
            self.atr_14,
            self.atr_pct,
            self.relative_volume,
            self.volume_trend,
            self.return_1d,
            self.return_5d,
            self.return_20d,
            self.price_vs_high_52w,
            self.price_vs_low_52w,
            regime_map.get(self.regime, 0),
            self.regime_confidence,
            self.fear_greed,
            fg_trend_map.get(self.fear_greed_trend, 0),
            self.day_of_week,
            self.hour_of_day,
            1 if self.is_market_open else 0,
            hourly_rsi,
            hourly_macd,
            self.hourly_trend_alignment,
            self.composite_score,
            self.ensemble_confidence,
        ], dtype=float)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "rsi_14", "macd_line", "macd_signal", "macd_histogram",
            "bb_position", "bb_width", "sma_10_dist", "sma_20_dist",
            "sma_50_dist", "ma_alignment", "volatility_20d", "atr_14",
            "atr_pct", "relative_volume", "volume_trend",
            "return_1d", "return_5d", "return_20d",
            "price_vs_high_52w", "price_vs_low_52w",
            "regime_encoded", "regime_confidence", "fear_greed",
            "fear_greed_trend_encoded", "day_of_week", "hour_of_day",
            "is_market_open",
            "hourly_rsi", "hourly_macd_signal", "hourly_trend_alignment",
            "composite_score", "ensemble_confidence",
        ]


class FeatureEngine:
    """Computes unified feature vectors from raw market data."""

    def compute(
        self,
        symbol: str,
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        volumes: Optional[List[float]] = None,
        indicators: Optional[Any] = None,
        regime_state: Optional[Any] = None,
        fear_greed_data: Optional[Dict] = None,
        ensemble_confidence: float = 0.0,
        hourly_data: Optional[Dict] = None,
    ) -> FeatureVector:
        """Compute full feature vector for a symbol."""
        now = datetime.now(timezone.utc)
        arr = np.array(prices, dtype=float) if prices else np.array([0.0])
        current_price = float(arr[-1]) if len(arr) > 0 else 0

        fv = FeatureVector(
            symbol=symbol,
            timestamp=now.isoformat(),
            day_of_week=now.weekday(),
            hour_of_day=now.hour,
        )

        if len(arr) < 20:
            return fv

        # ── Technical indicators ─────────────────────────────
        if indicators:
            fv.rsi_14 = getattr(indicators, "rsi", 50.0)
            fv.macd_line = getattr(indicators, "macd_line", 0.0)
            fv.macd_signal = getattr(indicators, "macd_signal", 0.0)
            fv.macd_histogram = getattr(indicators, "macd_histogram", 0.0)
            fv.bb_position = getattr(indicators, "bb_position", 0.5)
            fv.bb_width = getattr(indicators, "bb_width", 0.0)
            fv.ma_alignment = getattr(indicators, "ma_score", 0)
            fv.composite_score = getattr(indicators, "composite_score", 0)
            fv.volatility_20d = getattr(indicators, "volatility_20d", 0.0)
            fv.atr_14 = getattr(indicators, "atr_14", 0.0)

            sma_10 = getattr(indicators, "sma_10", current_price)
            sma_20 = getattr(indicators, "sma_20", current_price)
            sma_50 = getattr(indicators, "sma_50", current_price)
        else:
            # Compute from raw prices
            fv.rsi_14 = self._rsi(arr)
            sma_10 = float(np.mean(arr[-10:])) if len(arr) >= 10 else current_price
            sma_20 = float(np.mean(arr[-20:])) if len(arr) >= 20 else current_price
            sma_50 = float(np.mean(arr[-50:])) if len(arr) >= 50 else sma_20
            fv.ma_alignment = self._ma_alignment(current_price, sma_10, sma_20, sma_50)

            std_20 = float(np.std(arr[-20:])) if len(arr) >= 20 else 1.0
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            if bb_upper != bb_lower:
                fv.bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            fv.bb_width = (bb_upper - bb_lower) / sma_20 * 100 if sma_20 > 0 else 0

        # SMA distances
        if current_price > 0:
            fv.sma_10_dist = (current_price - sma_10) / current_price * 100
            fv.sma_20_dist = (current_price - sma_20) / current_price * 100
            fv.sma_50_dist = (current_price - sma_50) / current_price * 100

        # ATR as % of price
        if fv.atr_14 > 0 and current_price > 0:
            fv.atr_pct = fv.atr_14 / current_price * 100

        # ── Volume features ──────────────────────────────────
        if volumes and len(volumes) >= 20:
            vol_arr = np.array(volumes, dtype=float)
            avg_vol = float(np.mean(vol_arr[-20:]))
            if avg_vol > 0:
                fv.relative_volume = float(vol_arr[-1]) / avg_vol
            # Volume trend (linear regression slope over last 5 days)
            if len(vol_arr) >= 5:
                x = np.arange(5)
                y = vol_arr[-5:]
                if np.std(y) > 0:
                    fv.volume_trend = float(np.polyfit(x, y, 1)[0])

        # ── Momentum features ────────────────────────────────
        if len(arr) >= 2:
            fv.return_1d = (arr[-1] - arr[-2]) / arr[-2] * 100
        if len(arr) >= 6:
            fv.return_5d = (arr[-1] - arr[-6]) / arr[-6] * 100
        if len(arr) >= 21:
            fv.return_20d = (arr[-1] - arr[-21]) / arr[-21] * 100

        high_52w = float(np.max(arr)) if len(arr) > 0 else current_price
        low_52w = float(np.min(arr)) if len(arr) > 0 else current_price
        if high_52w > 0:
            fv.price_vs_high_52w = (current_price - high_52w) / high_52w * 100
        if low_52w > 0:
            fv.price_vs_low_52w = (current_price - low_52w) / low_52w * 100

        # ── Regime / Sentiment ───────────────────────────────
        if regime_state:
            fv.regime = getattr(regime_state, "regime", "unknown")
            if hasattr(fv.regime, "value"):
                fv.regime = fv.regime.value
            fv.regime_confidence = getattr(regime_state, "confidence", 0.0)

        if fear_greed_data:
            fv.fear_greed = fear_greed_data.get("value", 50)
            fv.fear_greed_trend = fear_greed_data.get("trend", "stable")

        fv.ensemble_confidence = ensemble_confidence

        # ── Multi-timeframe (hourly) features ────────────────
        if hourly_data and hourly_data.get("prices"):
            hourly_prices = np.array(hourly_data["prices"], dtype=float)
            if len(hourly_prices) >= 15:
                # Hourly RSI (14-bar)
                fv.hourly_rsi = self._rsi(hourly_prices, period=14)

                # Hourly MACD signal line (12, 26, 9)
                fv.hourly_macd_signal = self._macd_signal(hourly_prices)

                # Hourly trend alignment vs daily trend
                # Determine daily trend direction from daily prices
                daily_trend = 0.0
                if len(arr) >= 10:
                    daily_sma_short = float(np.mean(arr[-5:]))
                    daily_sma_long = float(np.mean(arr[-20:])) if len(arr) >= 20 else float(np.mean(arr))
                    if daily_sma_long > 0:
                        daily_trend = (daily_sma_short - daily_sma_long) / daily_sma_long

                # Determine hourly trend direction
                hourly_trend = 0.0
                if len(hourly_prices) >= 10:
                    hourly_sma_short = float(np.mean(hourly_prices[-5:]))
                    hourly_sma_long = float(np.mean(hourly_prices[-20:])) if len(hourly_prices) >= 20 else float(np.mean(hourly_prices))
                    if hourly_sma_long > 0:
                        hourly_trend = (hourly_sma_short - hourly_sma_long) / hourly_sma_long

                # Alignment: same sign = aligned (+1), opposite = opposed (-1)
                if daily_trend != 0 and hourly_trend != 0:
                    if (daily_trend > 0) == (hourly_trend > 0):
                        # Both trending same direction — scale by agreement strength
                        fv.hourly_trend_alignment = 1.0
                    else:
                        fv.hourly_trend_alignment = -1.0
                else:
                    fv.hourly_trend_alignment = 0.0

        return fv

    def _rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute exponential moving average."""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _macd_signal(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """Compute the MACD signal line value (last bar)."""
        if len(prices) < slow + signal:
            return 0.0
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        return float(signal_line[-1])

    def _ma_alignment(self, price: float, sma10: float, sma20: float, sma50: float) -> int:
        if price > sma10 > sma20 > sma50:
            return 25
        elif price > sma20 > sma50:
            return 15
        elif price > sma50:
            return 5
        elif price < sma10 < sma20 < sma50:
            return -25
        elif price < sma20 < sma50:
            return -15
        elif price < sma50:
            return -5
        return 0
