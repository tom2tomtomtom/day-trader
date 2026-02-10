#!/usr/bin/env python3
"""
MEME / PENNY STOCK STRATEGY — Social momentum scanner & volume spike detector.

Targets high-volatility assets where social sentiment drives price.
Lets ML learn optimal sizing instead of hard-coding conservative limits.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Expanded meme / high-vol universe
MEME_UNIVERSE = {
    # Crypto memes
    "DOGE-USD", "SHIB-USD", "AVAX-USD", "FLOKI-USD", "BONK-USD", "WIF-USD",
    # Crypto majors (volatile)
    "SOL-USD", "AVAX-USD", "LINK-USD",
    # Meme stocks / high-beta
    "GME", "AMC", "BBBY", "MSTR", "COIN",
    # Penny stocks (dynamic — populated from scanner)
}


@dataclass
class MemeSignal:
    """Signal from meme/penny stock scanner."""
    symbol: str
    social_score: float       # 0-1, normalized social momentum
    volume_spike: float       # Current vol / 20d avg
    price_momentum_1d: float  # 1-day % change
    price_momentum_5d: float  # 5-day % change
    is_trending: bool         # Appears in social trending lists
    mentions_24h: int
    composite_score: float    # 0-100 combined score
    reasons: List[str]


class MemeStrategy:
    """Scanner for meme coins, penny stocks, and social-momentum plays."""

    # Thresholds (lower than standard — we want to catch moves early)
    MIN_VOLUME_SPIKE = 1.5     # 1.5x average volume
    MIN_SOCIAL_SCORE = 0.3     # Minimum social momentum
    MIN_COMPOSITE = 30         # Minimum composite score to signal

    def scan(
        self,
        symbol: str,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        social_data: Optional[Dict] = None,
        reddit_data: Optional[Dict] = None,
    ) -> Optional[MemeSignal]:
        """Scan a single symbol for meme/momentum characteristics."""
        if not prices or len(prices) < 5:
            return None

        arr = np.array(prices, dtype=float)
        current = float(arr[-1])
        reasons = []

        # Volume spike
        volume_spike = 1.0
        if volumes and len(volumes) >= 20:
            vol_arr = np.array(volumes, dtype=float)
            avg_vol = float(np.mean(vol_arr[-20:]))
            if avg_vol > 0:
                volume_spike = float(vol_arr[-1]) / avg_vol

        if volume_spike >= self.MIN_VOLUME_SPIKE:
            reasons.append(f"Volume spike: {volume_spike:.1f}x average")

        # Price momentum
        mom_1d = (arr[-1] - arr[-2]) / arr[-2] * 100 if len(arr) >= 2 else 0
        mom_5d = (arr[-1] - arr[-6]) / arr[-6] * 100 if len(arr) >= 6 else 0

        if mom_1d > 5:
            reasons.append(f"1d momentum: +{mom_1d:.1f}%")
        if mom_5d > 15:
            reasons.append(f"5d momentum: +{mom_5d:.1f}%")

        # Social score
        social_score = 0.0
        mentions = 0
        is_trending = False

        if social_data and social_data.get("available"):
            reddit_mentions = social_data.get("reddit_mentions", 0)
            twitter_mentions = social_data.get("twitter_mentions", 0)
            mentions = reddit_mentions + twitter_mentions
            # Normalize: 100+ mentions = 1.0
            social_score = min(1.0, mentions / 100)
            if social_score > 0.5:
                reasons.append(f"High social buzz: {mentions} mentions")

        if reddit_data and reddit_data.get("available"):
            trending_symbols = [t.get("symbol") for t in reddit_data.get("trending", [])]
            if symbol.replace("-USD", "") in trending_symbols:
                is_trending = True
                social_score = max(social_score, 0.7)
                # Find rank
                for t in reddit_data.get("trending", []):
                    if t.get("symbol") == symbol.replace("-USD", ""):
                        mentions = max(mentions, t.get("mentions_24h", 0))
                        reasons.append(f"Trending on Reddit (rank #{t.get('rank', '?')})")
                        break

        # Composite score (0-100)
        vol_component = min(30, (volume_spike - 1) * 20)    # Max 30 pts
        social_component = social_score * 30                  # Max 30 pts
        mom_component = min(20, max(0, mom_1d * 2))           # Max 20 pts
        trend_component = 20 if is_trending else 0            # 20 pts bonus

        composite = vol_component + social_component + mom_component + trend_component
        composite = max(0, min(100, composite))

        if composite < self.MIN_COMPOSITE and not reasons:
            return None

        return MemeSignal(
            symbol=symbol,
            social_score=round(social_score, 3),
            volume_spike=round(volume_spike, 2),
            price_momentum_1d=round(mom_1d, 2),
            price_momentum_5d=round(mom_5d, 2),
            is_trending=is_trending,
            mentions_24h=mentions,
            composite_score=round(composite, 1),
            reasons=reasons,
        )

    def scan_universe(
        self,
        price_data: Dict[str, Dict],
        social_data: Optional[Dict] = None,
        reddit_data: Optional[Dict] = None,
    ) -> List[MemeSignal]:
        """Scan all meme/penny symbols and return sorted signals."""
        signals = []
        for symbol in MEME_UNIVERSE:
            data = price_data.get(symbol, {})
            if not data:
                continue
            sig = self.scan(
                symbol=symbol,
                prices=data.get("prices", []),
                volumes=data.get("volumes"),
                social_data=social_data,
                reddit_data=reddit_data,
            )
            if sig and sig.composite_score >= self.MIN_COMPOSITE:
                signals.append(sig)

        signals.sort(key=lambda s: s.composite_score, reverse=True)
        return signals
