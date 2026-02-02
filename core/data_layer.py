#!/usr/bin/env python3
"""
DATA LAYER - Unified data aggregation from multiple sources

Sources:
- Price: Yahoo Finance (free)
- Sentiment: Finnhub, ApeWisdom (free)
- Options Flow: Unusual Whales (paid - optional)
- On-Chain: Glassnode (free tier)
- Fear & Greed: Alternative.me (free)

Each source is modular - system works with whatever is available.
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# API Keys from environment
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
UNUSUAL_WHALES_KEY = os.environ.get("UNUSUAL_WHALES_KEY", "")
GLASSNODE_KEY = os.environ.get("GLASSNODE_KEY", "")


class DataSource(ABC):
    """Base class for data sources"""
    
    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl = cache_ttl_seconds
    
    def _get_cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{self.__class__.__name__}_{key}.json"
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        path = self._get_cache_path(key)
        if path.exists():
            data = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now(timezone.utc) - cached_at < timedelta(seconds=self.cache_ttl):
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict):
        data["_cached_at"] = datetime.now(timezone.utc).isoformat()
        path = self._get_cache_path(key)
        path.write_text(json.dumps(data, indent=2))
    
    @abstractmethod
    def fetch(self, symbol: str) -> Optional[Dict]:
        pass


# === PRICE DATA ===

class YahooFinanceSource(DataSource):
    """Yahoo Finance for price/volume data"""
    
    def fetch(self, symbol: str, period: str = "3mo") -> Optional[Dict]:
        cache_key = f"{symbol}_{period}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1d")
            
            if hist.empty:
                return None
            
            data = {
                "symbol": symbol,
                "current_price": float(hist['Close'].iloc[-1]),
                "open": float(hist['Open'].iloc[-1]),
                "high": float(hist['High'].iloc[-1]),
                "low": float(hist['Low'].iloc[-1]),
                "volume": int(hist['Volume'].iloc[-1]),
                "change_pct": float((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100),
                "high_52w": float(hist['High'].max()),
                "low_52w": float(hist['Low'].min()),
                "avg_volume_20d": float(hist['Volume'].iloc[-20:].mean()),
                "prices": [float(p) for p in hist['Close'].tolist()[-60:]],
                "volumes": [int(v) for v in hist['Volume'].tolist()[-60:]],
                "dates": [d.isoformat() for d in hist.index.tolist()[-60:]],
            }
            
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            return {"error": str(e)}


# === SENTIMENT DATA ===

class FearGreedSource(DataSource):
    """Crypto Fear & Greed Index"""
    
    def __init__(self):
        super().__init__(cache_ttl_seconds=3600)  # 1 hour cache
    
    def fetch(self, symbol: str = "BTC") -> Optional[Dict]:
        cached = self._get_cached("fear_greed")
        if cached:
            return cached
        
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=30", timeout=10)
            data = resp.json()
            
            if data.get("data"):
                history = data["data"]
                current = history[0]
                
                result = {
                    "value": int(current["value"]),
                    "label": current["value_classification"],
                    "timestamp": current["timestamp"],
                    "history_7d": [int(d["value"]) for d in history[:7]],
                    "history_30d": [int(d["value"]) for d in history],
                    "avg_7d": sum(int(d["value"]) for d in history[:7]) / 7,
                    "trend": "improving" if int(history[0]["value"]) > int(history[6]["value"]) else "declining",
                    "extreme_fear": int(current["value"]) < 25,
                    "extreme_greed": int(current["value"]) > 75,
                }
                
                self._set_cache("fear_greed", result)
                return result
        except Exception as e:
            return {"error": str(e), "value": 50}
        
        return None


class FinnhubSentiment(DataSource):
    """Finnhub social sentiment (Twitter mentions, etc.)"""
    
    def fetch(self, symbol: str) -> Optional[Dict]:
        if not FINNHUB_API_KEY:
            return {"error": "FINNHUB_API_KEY not set", "available": False}
        
        cached = self._get_cached(symbol)
        if cached:
            return cached
        
        try:
            # Social sentiment
            url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={symbol}&token={FINNHUB_API_KEY}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            
            if data.get("reddit") or data.get("twitter"):
                result = {
                    "symbol": symbol,
                    "reddit": data.get("reddit", []),
                    "twitter": data.get("twitter", []),
                    "reddit_mentions": sum(d.get("mention", 0) for d in data.get("reddit", [])),
                    "twitter_mentions": sum(d.get("mention", 0) for d in data.get("twitter", [])),
                    "reddit_sentiment": sum(d.get("positiveMention", 0) - d.get("negativeMention", 0) 
                                           for d in data.get("reddit", [])),
                    "available": True
                }
                self._set_cache(symbol, result)
                return result
        except Exception as e:
            return {"error": str(e), "available": False}
        
        return {"available": False}


class ApeWisdomSource(DataSource):
    """Reddit WSB/stocks mentions from ApeWisdom"""
    
    def __init__(self):
        super().__init__(cache_ttl_seconds=1800)  # 30 min cache
    
    def fetch(self, symbol: str = None) -> Optional[Dict]:
        cached = self._get_cached("trending")
        if cached:
            return cached
        
        try:
            # Get trending tickers
            resp = requests.get("https://apewisdom.io/api/v1.0/filter/all-stocks/page/1", timeout=10)
            data = resp.json()
            
            if data.get("results"):
                tickers = data["results"][:20]
                result = {
                    "trending": [
                        {
                            "symbol": t.get("ticker"),
                            "name": t.get("name"),
                            "mentions_24h": t.get("mentions"),
                            "rank": t.get("rank"),
                            "upvotes": t.get("upvotes", 0),
                        }
                        for t in tickers
                    ],
                    "top_mentioned": tickers[0].get("ticker") if tickers else None,
                    "available": True
                }
                self._set_cache("trending", result)
                return result
        except Exception as e:
            return {"error": str(e), "available": False}
        
        return {"available": False}


# === OPTIONS FLOW ===

class UnusualWhalesSource(DataSource):
    """Unusual Whales options flow (requires paid API)"""
    
    def fetch(self, symbol: str) -> Optional[Dict]:
        if not UNUSUAL_WHALES_KEY:
            return {"error": "UNUSUAL_WHALES_KEY not set", "available": False}
        
        cached = self._get_cached(symbol)
        if cached:
            return cached
        
        try:
            headers = {"Authorization": f"Bearer {UNUSUAL_WHALES_KEY}"}
            url = f"https://api.unusualwhales.com/api/stock/{symbol}/options-flow"
            resp = requests.get(url, headers=headers, timeout=10)
            data = resp.json()
            
            if data.get("data"):
                flows = data["data"]
                
                # Aggregate bullish vs bearish flow
                bullish_premium = sum(f.get("premium", 0) for f in flows 
                                     if f.get("sentiment") == "bullish")
                bearish_premium = sum(f.get("premium", 0) for f in flows 
                                     if f.get("sentiment") == "bearish")
                
                result = {
                    "symbol": symbol,
                    "flows": flows[:10],  # Top 10 recent
                    "bullish_premium": bullish_premium,
                    "bearish_premium": bearish_premium,
                    "net_sentiment": "bullish" if bullish_premium > bearish_premium else "bearish",
                    "flow_ratio": bullish_premium / bearish_premium if bearish_premium > 0 else 999,
                    "available": True
                }
                self._set_cache(symbol, result)
                return result
        except Exception as e:
            return {"error": str(e), "available": False}
        
        return {"available": False}


# === UNIFIED DATA AGGREGATOR ===

class DataAggregator:
    """
    Unified data aggregator - combines all sources into single view
    """
    
    def __init__(self):
        self.price_source = YahooFinanceSource()
        self.fear_greed = FearGreedSource()
        self.finnhub = FinnhubSentiment()
        self.ape_wisdom = ApeWisdomSource()
        self.options_flow = UnusualWhalesSource()
    
    def get_full_picture(self, symbol: str) -> Dict[str, Any]:
        """Get all available data for a symbol"""
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_sources": {},
        }
        
        # Price data (always available)
        price_data = self.price_source.fetch(symbol)
        if price_data and "error" not in price_data:
            result["price"] = price_data
            result["data_sources"]["price"] = True
        else:
            result["data_sources"]["price"] = False
        
        # Fear & Greed (always available)
        fg_data = self.fear_greed.fetch()
        if fg_data and "error" not in fg_data:
            result["fear_greed"] = fg_data
            result["data_sources"]["fear_greed"] = True
        else:
            result["data_sources"]["fear_greed"] = False
        
        # Finnhub sentiment (needs API key)
        sentiment_data = self.finnhub.fetch(symbol)
        result["sentiment"] = sentiment_data
        result["data_sources"]["finnhub"] = sentiment_data.get("available", False)
        
        # Reddit trending
        reddit_data = self.ape_wisdom.fetch()
        result["reddit_trending"] = reddit_data
        result["data_sources"]["reddit"] = reddit_data.get("available", False)
        
        # Options flow (needs paid API)
        options_data = self.options_flow.fetch(symbol)
        result["options_flow"] = options_data
        result["data_sources"]["options_flow"] = options_data.get("available", False)
        
        return result
    
    def get_market_context(self) -> Dict[str, Any]:
        """Get overall market context"""
        
        fg = self.fear_greed.fetch()
        reddit = self.ape_wisdom.fetch()
        spy = self.price_source.fetch("SPY")
        vix_data = self.price_source.fetch("^VIX")
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fear_greed": fg,
            "vix": vix_data.get("current_price") if vix_data else None,
            "spy_change": spy.get("change_pct") if spy else None,
            "reddit_trending": reddit.get("trending", [])[:5] if reddit else [],
            "market_regime": self._determine_regime(fg, vix_data)
        }
    
    def _determine_regime(self, fg_data: Dict, vix_data: Dict) -> str:
        """Determine current market regime"""
        fg_value = fg_data.get("value", 50) if fg_data else 50
        vix = vix_data.get("current_price", 20) if vix_data else 20
        
        if vix > 30:
            return "CRISIS"
        elif fg_value < 25:
            return "EXTREME_FEAR"
        elif fg_value > 75:
            return "EXTREME_GREED"
        elif vix > 20:
            return "ELEVATED_VOL"
        else:
            return "NORMAL"


# Test
if __name__ == "__main__":
    agg = DataAggregator()
    
    print("=== Market Context ===")
    ctx = agg.get_market_context()
    print(json.dumps(ctx, indent=2, default=str))
    
    print("\n=== AAPL Full Picture ===")
    aapl = agg.get_full_picture("AAPL")
    print(f"Data sources available: {aapl['data_sources']}")
    if aapl.get("price"):
        print(f"Price: ${aapl['price']['current_price']:.2f}")
    if aapl.get("fear_greed"):
        print(f"Fear & Greed: {aapl['fear_greed']['value']}")
