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

    # New indicators
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    adx: float = 0.0
    obv_slope: float = 0.0
    vwap_distance_pct: float = 0.0

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

    # Microstructure
    garman_klass_vol: float = 0.0         # Garman-Klass volatility estimator (OHLC)
    parkinson_vol: float = 0.0            # Parkinson volatility (high-low range)
    intraday_range_pct: float = 0.0       # (High - Low) / Close * 100
    up_volume_ratio: float = 0.5          # Up volume / total volume (last 10 bars)
    volume_price_trend: float = 0.0       # Cumulative volume * price change
    close_to_high_pct: float = 0.0        # (Close - Low) / (High - Low) * 100
    avg_bar_range_pct: float = 0.0        # Average (H-L)/C over 14 bars
    volume_momentum: float = 0.0          # Volume weighted price momentum

    # Advanced Momentum
    roc_5: float = 0.0                    # Rate of change 5 bars
    roc_10: float = 0.0                   # Rate of change 10 bars
    roc_20: float = 0.0                   # Rate of change 20 bars
    momentum_oscillator: float = 0.0      # Price - SMA(10) normalized
    price_acceleration: float = 0.0       # Second derivative of price
    mean_reversion_zscore: float = 0.0    # (Price - SMA20) / StdDev20
    hurst_estimate: float = 0.5           # Simplified Hurst exponent
    rsi_divergence: float = 0.0           # Difference between price trend and RSI trend

    # Pattern
    consecutive_up_days: int = 0          # Count of consecutive up closes
    consecutive_down_days: int = 0        # Count of consecutive down closes
    gap_pct: float = 0.0                  # Overnight gap %
    inside_bar: int = 0                   # 1 if today's range inside yesterday's
    higher_highs_count: int = 0           # Count of HH in last 5 bars

    # Calendar
    day_of_month: int = 15                # 1-31
    week_of_month: int = 2                # 1-5
    is_month_end: int = 0                 # Last 3 trading days of month
    is_quarter_end: int = 0               # Last 5 trading days of quarter

    # Multi-timeframe (hourly)
    hourly_rsi: float = float('nan')           # RSI computed on hourly bars
    hourly_macd_signal: float = float('nan')   # MACD signal line on hourly bars
    hourly_trend_alignment: float = 0.0        # 1.0 = hourly agrees with daily, -1.0 = opposed

    # Multi-timeframe extended (weekly/monthly)
    weekly_rsi: float = 50.0              # RSI on weekly aggregated bars
    weekly_macd_signal: float = 0.0       # Weekly MACD signal
    weekly_bb_position: float = 0.5       # Weekly Bollinger position
    monthly_trend: float = 0.0            # Monthly SMA direction
    daily_weekly_divergence: float = 0.0  # Daily vs weekly trend disagreement

    # Cross-asset proxy
    price_vs_sma200: float = 0.0          # Distance from 200 SMA
    volatility_regime: float = 0.0        # Current vol / 6-month average vol
    trend_strength_adx: float = 0.0       # ADX value

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
            # Original 37 features
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
            self.stochastic_k,
            self.stochastic_d,
            self.adx,
            self.obv_slope,
            self.vwap_distance_pct,
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
            # Microstructure (8)
            self.garman_klass_vol,
            self.parkinson_vol,
            self.intraday_range_pct,
            self.up_volume_ratio,
            self.volume_price_trend,
            self.close_to_high_pct,
            self.avg_bar_range_pct,
            self.volume_momentum,
            # Advanced Momentum (8)
            self.roc_5,
            self.roc_10,
            self.roc_20,
            self.momentum_oscillator,
            self.price_acceleration,
            self.mean_reversion_zscore,
            self.hurst_estimate,
            self.rsi_divergence,
            # Pattern (5)
            self.consecutive_up_days,
            self.consecutive_down_days,
            self.gap_pct,
            self.inside_bar,
            self.higher_highs_count,
            # Calendar (4)
            self.day_of_month,
            self.week_of_month,
            self.is_month_end,
            self.is_quarter_end,
            # Multi-timeframe extended (5)
            self.weekly_rsi,
            self.weekly_macd_signal,
            self.weekly_bb_position,
            self.monthly_trend,
            self.daily_weekly_divergence,
            # Cross-asset proxy (3)
            self.price_vs_sma200,
            self.volatility_regime,
            self.trend_strength_adx,
        ], dtype=float)

    @staticmethod
    def feature_categories() -> Dict[str, List[str]]:
        """Map category names to feature name lists. Used by hypothesis engine."""
        return {
            "technical": [
                "rsi_14", "macd_line", "macd_signal", "macd_histogram",
                "bb_position", "bb_width", "sma_10_dist", "sma_20_dist",
                "sma_50_dist", "ma_alignment", "stochastic_k", "stochastic_d",
                "adx", "obv_slope", "vwap_distance_pct",
            ],
            "volatility": ["volatility_20d", "atr_14", "atr_pct"],
            "volume": ["relative_volume", "volume_trend"],
            "momentum": [
                "return_1d", "return_5d", "return_20d",
                "price_vs_high_52w", "price_vs_low_52w",
            ],
            "regime_sentiment": [
                "regime_encoded", "regime_confidence", "fear_greed",
                "fear_greed_trend_encoded",
            ],
            "time": ["day_of_week", "hour_of_day", "is_market_open"],
            "microstructure": [
                "garman_klass_vol", "parkinson_vol", "intraday_range_pct",
                "up_volume_ratio", "volume_price_trend", "close_to_high_pct",
                "avg_bar_range_pct", "volume_momentum",
            ],
            "advanced_momentum": [
                "roc_5", "roc_10", "roc_20", "momentum_oscillator",
                "price_acceleration", "mean_reversion_zscore",
                "hurst_estimate", "rsi_divergence",
            ],
            "pattern": [
                "consecutive_up_days", "consecutive_down_days", "gap_pct",
                "inside_bar", "higher_highs_count",
            ],
            "calendar": [
                "day_of_month", "week_of_month", "is_month_end", "is_quarter_end",
            ],
            "multi_timeframe": [
                "weekly_rsi", "weekly_macd_signal", "weekly_bb_position",
                "monthly_trend", "daily_weekly_divergence",
            ],
            "cross_asset": [
                "price_vs_sma200", "volatility_regime", "trend_strength_adx",
            ],
            "composite": ["composite_score", "ensemble_confidence"],
            "hourly": [
                "hourly_rsi", "hourly_macd_signal", "hourly_trend_alignment",
            ],
        }

    @staticmethod
    def feature_names() -> List[str]:
        return [
            # Original 37
            "rsi_14", "macd_line", "macd_signal", "macd_histogram",
            "bb_position", "bb_width", "sma_10_dist", "sma_20_dist",
            "sma_50_dist", "ma_alignment",
            "stochastic_k", "stochastic_d", "adx", "obv_slope", "vwap_distance_pct",
            "volatility_20d", "atr_14",
            "atr_pct", "relative_volume", "volume_trend",
            "return_1d", "return_5d", "return_20d",
            "price_vs_high_52w", "price_vs_low_52w",
            "regime_encoded", "regime_confidence", "fear_greed",
            "fear_greed_trend_encoded", "day_of_week", "hour_of_day",
            "is_market_open",
            "hourly_rsi", "hourly_macd_signal", "hourly_trend_alignment",
            "composite_score", "ensemble_confidence",
            # Microstructure (8)
            "garman_klass_vol", "parkinson_vol", "intraday_range_pct",
            "up_volume_ratio", "volume_price_trend", "close_to_high_pct",
            "avg_bar_range_pct", "volume_momentum",
            # Advanced Momentum (8)
            "roc_5", "roc_10", "roc_20", "momentum_oscillator",
            "price_acceleration", "mean_reversion_zscore",
            "hurst_estimate", "rsi_divergence",
            # Pattern (5)
            "consecutive_up_days", "consecutive_down_days", "gap_pct",
            "inside_bar", "higher_highs_count",
            # Calendar (4)
            "day_of_month", "week_of_month", "is_month_end", "is_quarter_end",
            # Multi-timeframe extended (5)
            "weekly_rsi", "weekly_macd_signal", "weekly_bb_position",
            "monthly_trend", "daily_weekly_divergence",
            # Cross-asset proxy (3)
            "price_vs_sma200", "volatility_regime", "trend_strength_adx",
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
            fv.stochastic_k = getattr(indicators, "stochastic_k", 50.0)
            fv.stochastic_d = getattr(indicators, "stochastic_d", 50.0)
            fv.adx = getattr(indicators, "adx", 0.0)
            fv.obv_slope = getattr(indicators, "obv_slope", 0.0)
            vwap = getattr(indicators, "vwap", 0)
            if vwap > 0 and current_price > 0:
                fv.vwap_distance_pct = (current_price - vwap) / vwap * 100

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

        # ── Microstructure features ───────────────────────────
        h_arr = np.array(highs, dtype=float) if highs else None
        l_arr = np.array(lows, dtype=float) if lows else None
        v_arr = np.array(volumes, dtype=float) if volumes else None

        if h_arr is not None and l_arr is not None and len(h_arr) >= 20 and len(l_arr) >= 20:
            h20, l20, c20 = h_arr[-20:], l_arr[-20:], arr[-20:]
            # Approximate open as previous close
            o20 = arr[-21:-1] if len(arr) >= 21 else c20
            # Garman-Klass volatility
            with np.errstate(divide='ignore', invalid='ignore'):
                hl_ratio = np.where((l20 > 0), h20 / l20, 1.0)
                co_ratio = np.where((o20 > 0), c20 / o20, 1.0)
                gk = np.mean(0.5 * np.log(hl_ratio)**2 - (2 * np.log(2) - 1) * np.log(co_ratio)**2)
                if np.isfinite(gk) and gk > 0:
                    fv.garman_klass_vol = float(np.sqrt(gk * 252))

            # Parkinson volatility
            with np.errstate(divide='ignore', invalid='ignore'):
                park = np.mean(np.log(hl_ratio)**2) / (4 * np.log(2))
                if np.isfinite(park) and park > 0:
                    fv.parkinson_vol = float(np.sqrt(park * 252))

        # Intraday range
        if h_arr is not None and l_arr is not None and len(h_arr) > 0 and current_price > 0:
            fv.intraday_range_pct = float((h_arr[-1] - l_arr[-1]) / current_price * 100)

        # Close-to-high position within bar
        if h_arr is not None and l_arr is not None and len(h_arr) > 0:
            bar_range = h_arr[-1] - l_arr[-1]
            if bar_range > 0:
                fv.close_to_high_pct = float((current_price - l_arr[-1]) / bar_range * 100)

        # Average bar range over 14 bars
        if h_arr is not None and l_arr is not None and len(h_arr) >= 14:
            bar_ranges = (h_arr[-14:] - l_arr[-14:])
            bar_closes = arr[-14:]
            safe_closes = np.where(bar_closes > 0, bar_closes, 1.0)
            fv.avg_bar_range_pct = float(np.mean(bar_ranges / safe_closes) * 100)

        # Up volume ratio (last 10 bars)
        if v_arr is not None and len(v_arr) >= 10 and len(arr) >= 11:
            price_changes = np.diff(arr[-11:])
            vols_10 = v_arr[-10:]
            up_mask = price_changes > 0
            total_vol = float(np.sum(vols_10))
            if total_vol > 0:
                fv.up_volume_ratio = float(np.sum(vols_10[up_mask]) / total_vol)

        # Volume price trend (cumulative)
        if v_arr is not None and len(v_arr) >= 20 and len(arr) >= 21:
            pct_changes = np.diff(arr[-21:]) / np.where(arr[-21:-1] > 0, arr[-21:-1], 1.0)
            fv.volume_price_trend = float(np.sum(v_arr[-20:] * pct_changes))

        # Volume momentum (volume-weighted price momentum over 10 bars)
        if v_arr is not None and len(v_arr) >= 10 and len(arr) >= 11:
            pct_chg = np.diff(arr[-11:]) / np.where(arr[-11:-1] > 0, arr[-11:-1], 1.0)
            vol_weights = v_arr[-10:]
            total_w = float(np.sum(vol_weights))
            if total_w > 0:
                fv.volume_momentum = float(np.sum(pct_chg * vol_weights) / total_w * 100)

        # ── Advanced Momentum features ────────────────────────
        if len(arr) >= 6:
            fv.roc_5 = float((arr[-1] - arr[-6]) / arr[-6] * 100) if arr[-6] != 0 else 0.0
        if len(arr) >= 11:
            fv.roc_10 = float((arr[-1] - arr[-11]) / arr[-11] * 100) if arr[-11] != 0 else 0.0
        if len(arr) >= 21:
            fv.roc_20 = float((arr[-1] - arr[-21]) / arr[-21] * 100) if arr[-21] != 0 else 0.0

        # Momentum oscillator: (Price - SMA10) / SMA10 * 100
        if len(arr) >= 10:
            sma10_val = float(np.mean(arr[-10:]))
            if sma10_val > 0:
                fv.momentum_oscillator = float((current_price - sma10_val) / sma10_val * 100)

        # Price acceleration (second derivative)
        if len(arr) >= 3:
            mom_now = arr[-1] - arr[-2]
            mom_prev = arr[-2] - arr[-3]
            if arr[-2] != 0:
                fv.price_acceleration = float((mom_now - mom_prev) / abs(arr[-2]) * 100)

        # Mean reversion z-score
        if len(arr) >= 20:
            sma20_val = float(np.mean(arr[-20:]))
            std20_val = float(np.std(arr[-20:]))
            if std20_val > 0:
                fv.mean_reversion_zscore = float((current_price - sma20_val) / std20_val)

        # Hurst exponent estimate
        if len(arr) >= 20:
            fv.hurst_estimate = self._hurst_estimate(arr)

        # RSI divergence: compare price trend direction vs RSI trend direction
        if len(arr) >= 15:
            price_slope = float(np.polyfit(np.arange(10), arr[-10:], 1)[0])
            rsi_recent = self._rsi(arr[-15:], period=14)
            rsi_earlier = self._rsi(arr[-20:-5], period=14) if len(arr) >= 20 else 50.0
            rsi_slope = rsi_recent - rsi_earlier
            # Normalize: positive = bullish divergence (price down, RSI up)
            price_dir = 1.0 if price_slope > 0 else -1.0
            rsi_dir = 1.0 if rsi_slope > 0 else -1.0
            fv.rsi_divergence = float(rsi_dir - price_dir)  # 0=agree, +2=bull div, -2=bear div

        # ── Pattern features ──────────────────────────────────
        if len(arr) >= 2:
            # Consecutive up/down days
            up_count = 0
            down_count = 0
            for i in range(len(arr) - 1, 0, -1):
                if arr[i] > arr[i - 1]:
                    if down_count > 0:
                        break
                    up_count += 1
                elif arr[i] < arr[i - 1]:
                    if up_count > 0:
                        break
                    down_count += 1
                else:
                    break
            fv.consecutive_up_days = up_count
            fv.consecutive_down_days = down_count

            # Gap percentage (approximate: open ≈ current close vs prev close)
            if len(arr) >= 2 and arr[-2] != 0:
                # Use high/low to estimate if we have them, otherwise use closes
                if h_arr is not None and l_arr is not None and len(h_arr) >= 1:
                    # Gap = today's low vs yesterday's close (for gap up)
                    # or today's high vs yesterday's close (for gap down)
                    gap = float(l_arr[-1] - arr[-2]) if l_arr[-1] > arr[-2] else float(h_arr[-1] - arr[-2]) if h_arr[-1] < arr[-2] else 0.0
                    fv.gap_pct = gap / arr[-2] * 100
                else:
                    fv.gap_pct = 0.0

        # Inside bar
        if h_arr is not None and l_arr is not None and len(h_arr) >= 2:
            if h_arr[-1] <= h_arr[-2] and l_arr[-1] >= l_arr[-2]:
                fv.inside_bar = 1

        # Higher highs count in last 5 bars
        if h_arr is not None and len(h_arr) >= 5:
            hh = 0
            for i in range(-4, 0):
                if h_arr[i] > h_arr[i - 1]:
                    hh += 1
            fv.higher_highs_count = hh

        # ── Calendar features ─────────────────────────────────
        fv.day_of_month = now.day
        fv.week_of_month = (now.day - 1) // 7 + 1
        # Month-end: last 3 trading days (approximate: day >= 27)
        if now.month in (1, 3, 5, 7, 8, 10, 12):
            fv.is_month_end = 1 if now.day >= 29 else 0
        elif now.month == 2:
            fv.is_month_end = 1 if now.day >= 26 else 0
        else:
            fv.is_month_end = 1 if now.day >= 28 else 0
        # Quarter-end: last 5 trading days of Mar, Jun, Sep, Dec
        if now.month in (3, 6, 9, 12) and now.day >= 25:
            fv.is_quarter_end = 1

        # ── Multi-timeframe extended (weekly/monthly) ─────────
        if len(arr) >= 5:
            # Simulate weekly bars by taking every 5th close
            weekly_closes = arr[::5] if len(arr) >= 25 else arr[::max(1, len(arr) // 5)]
            if len(weekly_closes) >= 15:
                fv.weekly_rsi = self._rsi(weekly_closes, period=14)
            if len(weekly_closes) >= 36:
                fv.weekly_macd_signal = self._macd_signal(weekly_closes)
            # Weekly Bollinger position
            if len(weekly_closes) >= 20:
                w_sma20 = float(np.mean(weekly_closes[-20:]))
                w_std20 = float(np.std(weekly_closes[-20:]))
                if w_std20 > 0:
                    w_upper = w_sma20 + 2 * w_std20
                    w_lower = w_sma20 - 2 * w_std20
                    if w_upper != w_lower:
                        fv.weekly_bb_position = float((weekly_closes[-1] - w_lower) / (w_upper - w_lower))

        # Monthly trend (SMA direction over ~20 day bars = 1 month)
        if len(arr) >= 40:
            monthly_sma_now = float(np.mean(arr[-20:]))
            monthly_sma_prev = float(np.mean(arr[-40:-20]))
            if monthly_sma_prev > 0:
                fv.monthly_trend = float((monthly_sma_now - monthly_sma_prev) / monthly_sma_prev * 100)

        # Daily vs weekly divergence
        if len(arr) >= 25:
            daily_sma5 = float(np.mean(arr[-5:]))
            daily_sma20 = float(np.mean(arr[-20:]))
            weekly_closes = arr[::5]
            if len(weekly_closes) >= 5:
                weekly_sma5 = float(np.mean(weekly_closes[-5:]))
                weekly_sma_long = float(np.mean(weekly_closes[-min(20, len(weekly_closes)):]))
                if daily_sma20 > 0 and weekly_sma_long > 0:
                    daily_dir = (daily_sma5 - daily_sma20) / daily_sma20
                    weekly_dir = (weekly_sma5 - weekly_sma_long) / weekly_sma_long
                    # Positive = daily bullish but weekly bearish (or vice versa)
                    if (daily_dir > 0) != (weekly_dir > 0):
                        fv.daily_weekly_divergence = float(daily_dir - weekly_dir)

        # ── Cross-asset proxy features ────────────────────────
        # Price vs SMA200
        if len(arr) >= 200:
            sma200 = float(np.mean(arr[-200:]))
            if sma200 > 0:
                fv.price_vs_sma200 = float((current_price - sma200) / sma200 * 100)
        elif len(arr) >= 50:
            # Approximate with longest available SMA
            sma_long = float(np.mean(arr))
            if sma_long > 0:
                fv.price_vs_sma200 = float((current_price - sma_long) / sma_long * 100)

        # Volatility regime: current 20d vol / 120d (~6mo) average vol
        if len(arr) >= 120:
            vol_20 = float(np.std(np.diff(np.log(np.where(arr[-20:] > 0, arr[-20:], 1.0)))) * np.sqrt(252))
            vol_120 = float(np.std(np.diff(np.log(np.where(arr[-120:] > 0, arr[-120:], 1.0)))) * np.sqrt(252))
            if vol_120 > 0:
                fv.volatility_regime = vol_20 / vol_120
        elif fv.volatility_20d > 0 and len(arr) >= 40:
            vol_long = float(np.std(np.diff(np.log(np.where(arr > 0, arr, 1.0)))) * np.sqrt(252))
            if vol_long > 0:
                fv.volatility_regime = fv.volatility_20d / vol_long

        # Trend strength ADX (use computed ADX if available, otherwise approximate)
        fv.trend_strength_adx = fv.adx

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

    def _hurst_estimate(self, prices: np.ndarray, max_lag: int = 20) -> float:
        """Simplified Hurst exponent via R/S analysis. 0.5=random, >0.5=trending."""
        if len(prices) < max_lag + 1:
            return 0.5
        lags = list(range(2, min(max_lag, len(prices) // 2)))
        if len(lags) < 2:
            return 0.5
        tau = []
        for lag in lags:
            pp = prices[-lag * 2:]
            std_val = float(np.std(np.subtract(pp[lag:], pp[:-lag])))
            if std_val > 0:
                tau.append(std_val)
            else:
                tau.append(1e-10)
        if not tau or min(tau) <= 0:
            return 0.5
        try:
            log_lags = np.log(np.array(lags[:len(tau)], dtype=float))
            log_tau = np.log(np.array(tau, dtype=float))
            reg = np.polyfit(log_lags, log_tau, 1)
            return max(0.0, min(1.0, float(reg[0])))
        except (np.linalg.LinAlgError, ValueError):
            return 0.5

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
