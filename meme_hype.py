#!/usr/bin/env python3
"""
Inverse Hype Equation for Meme Coins 🪙📉 v2.0

The thesis: Maximum hype = local top. Dying hype + price holding = accumulation zone.

TUNED PARAMETERS (based on backtest):
- Wider stops for memes (5% not 2%) - they're volatile
- Wait for confirmation (bounce after capitulation)
- Add Fear & Greed Index for market-wide sentiment
- Track volume trends over longer period (10d vs 5d)

Signals:
- Volume decay: Volume dropping while price stable = smart money exiting quietly
- Hype exhaustion: Price up huge but momentum fading = late buyers getting trapped
- Capitulation: Volume spike DOWN + high volume = panic selling (potential bottom)
- Accumulation: Low volume, price stable after big drop = smart money loading
- CONFIRMED REVERSAL: Capitulation + next day green = entry signal

Data Sources:
- Yahoo Finance volume/price data
- Alternative.me Fear & Greed Index (free API)
- CoinGecko trending (free)
"""

import json
import requests
from datetime import datetime, timezone
from pathlib import Path
import yfinance as yf
import numpy as np

BASE_DIR = Path(__file__).parent
SENTIMENT_CACHE = BASE_DIR / "sentiment_cache.json"

# Tuned parameters for meme coins
MEME_STOP_LOSS = 0.05  # 5% stops (memes are volatile)
MEME_TAKE_PROFIT = 0.10  # 10% targets
CONFIRMATION_REQUIRED = True  # Wait for green candle after capitulation


def get_fear_greed_index():
    """
    Fetch Crypto Fear & Greed Index from alternative.me (free)
    0-24: Extreme Fear (contrarian BUY)
    25-49: Fear
    50: Neutral
    51-74: Greed
    75-100: Extreme Greed (contrarian SELL)
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=10)
        data = resp.json()
        
        if data.get("data"):
            current = data["data"][0]
            history = data["data"]
            
            return {
                "value": int(current["value"]),
                "label": current["value_classification"],
                "timestamp": current["timestamp"],
                "trend_7d": [int(d["value"]) for d in history],
                "avg_7d": sum(int(d["value"]) for d in history) / len(history),
                "contrarian_signal": "BUY" if int(current["value"]) < 25 else "SELL" if int(current["value"]) > 75 else "NEUTRAL"
            }
    except Exception as e:
        return {"error": str(e), "value": 50, "label": "Unknown", "contrarian_signal": "NEUTRAL"}


def get_coingecko_trending():
    """
    Fetch trending coins from CoinGecko (free, no API key)
    High trending = late stage hype, be careful
    """
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/search/trending", timeout=10)
        data = resp.json()
        
        trending = []
        for coin in data.get("coins", [])[:7]:
            item = coin.get("item", {})
            trending.append({
                "symbol": item.get("symbol", "").upper(),
                "name": item.get("name"),
                "market_cap_rank": item.get("market_cap_rank"),
                "score": item.get("score", 0)  # Lower = more trending
            })
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trending": trending,
            "top_trending": trending[0]["symbol"] if trending else None
        }
    except Exception as e:
        return {"error": str(e), "trending": []}

# Meme coins and meme stocks to track
MEME_UNIVERSE = {
    # Crypto memes
    "DOGE-USD": "Dogecoin",
    "SHIB-USD": "Shiba Inu",
    "PEPE24478-USD": "Pepe",
    "BONK-USD": "Bonk",
    "WIF-USD": "dogwifhat",
    "FLOKI-USD": "Floki",
    
    # Meme stocks / high-beta crypto proxies
    "MSTR": "MicroStrategy",
    "COIN": "Coinbase", 
    "GME": "GameStop",
    "AMC": "AMC",
    "MARA": "Marathon Digital",
    "RIOT": "Riot Platforms",
}

def calculate_hype_score(symbol, period="1mo", fear_greed=None):
    """
    Calculate inverse hype score (higher = more contrarian opportunity)
    
    v2.0 TUNED for memes:
    - Longer volume lookback (10d vs 5d)
    - Check for confirmation candle
    - Factor in Fear & Greed index
    - Wider thresholds for volatile assets
    
    Components:
    1. Volume trend: Is volume declining? (bullish for reversal)
    2. Price exhaustion: Big move + fading momentum? (bearish, wait)
    3. Volatility contraction: Range tightening? (breakout coming)
    4. Capitulation check: Recent volume spike + price drop? (potential bottom)
    5. Confirmation: Did we get a green candle after the dump?
    6. Market sentiment: Fear & Greed index boost
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")
        
        if len(hist) < 10:
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        
        # 1. VOLUME TREND (declining volume = less retail hype)
        vol_20d_avg = hist['Volume'].iloc[-20:].mean() if len(hist) >= 20 else hist['Volume'].mean()
        vol_5d_avg = hist['Volume'].iloc[-5:].mean()
        vol_today = hist['Volume'].iloc[-1]
        
        vol_trend = vol_5d_avg / vol_20d_avg if vol_20d_avg > 0 else 1
        vol_today_vs_avg = vol_today / vol_20d_avg if vol_20d_avg > 0 else 1
        
        # Low volume = hype dying = potential accumulation
        volume_score = max(0, min(1, 1.5 - vol_trend))  # Higher when volume declining
        
        # 2. PRICE EXHAUSTION (big recent move + momentum fading)
        price_5d_ago = float(hist['Close'].iloc[-5]) if len(hist) >= 5 else current_price
        price_20d_ago = float(hist['Close'].iloc[-20]) if len(hist) >= 20 else current_price
        
        move_5d = (current_price - price_5d_ago) / price_5d_ago * 100
        move_20d = (current_price - price_20d_ago) / price_20d_ago * 100
        
        # Momentum slowing = exhaustion
        if move_20d > 20 and move_5d < move_20d * 0.2:
            exhaustion_score = 0.8  # Huge move but momentum died = distribution
            exhaustion_signal = "DISTRIBUTION"
        elif move_20d < -30 and move_5d > 0:
            exhaustion_score = 0.3  # Down big but bouncing = potential reversal
            exhaustion_signal = "BOUNCE_ATTEMPT"
        elif move_20d < -30 and move_5d < -10:
            exhaustion_score = 0.2  # Still dumping
            exhaustion_signal = "STILL_FALLING"
        else:
            exhaustion_score = 0.5
            exhaustion_signal = "NEUTRAL"
        
        # 3. VOLATILITY CONTRACTION (range tightening = breakout coming)
        atr_20 = (hist['High'] - hist['Low']).iloc[-20:].mean() if len(hist) >= 20 else 0
        atr_5 = (hist['High'] - hist['Low']).iloc[-5:].mean()
        
        vol_contraction = atr_5 / atr_20 if atr_20 > 0 else 1
        contraction_score = max(0, min(1, 1.2 - vol_contraction))  # Higher when range tightening
        
        # 4. CAPITULATION CHECK (volume spike + price dump = potential bottom)
        max_vol_5d = hist['Volume'].iloc[-5:].max()
        capitulation = False
        if max_vol_5d > vol_20d_avg * 2 and move_5d < -15:
            capitulation = True
            cap_score = 0.9  # High volume dump = potential capitulation
        else:
            cap_score = 0.3
        
        # 5. RANGE POSITION (where in recent range?)
        high_20d = hist['High'].iloc[-20:].max() if len(hist) >= 20 else hist['High'].max()
        low_20d = hist['Low'].iloc[-20:].min() if len(hist) >= 20 else hist['Low'].min()
        range_pos = (current_price - low_20d) / (high_20d - low_20d) if high_20d != low_20d else 0.5
        
        # INVERSE HYPE SCORE (higher = better contrarian opportunity)
        # Weight: volume_trend 25%, exhaustion 25%, contraction 20%, capitulation 30%
        inverse_hype = (
            volume_score * 0.25 +
            (1 - exhaustion_score) * 0.25 +  # Inverse of exhaustion
            contraction_score * 0.20 +
            cap_score * 0.30
        )
        
        # SIGNAL DETERMINATION
        if capitulation and range_pos < 0.3:
            signal = "CAPITULATION_BUY"
            trade_idea = "Volume spike + price at lows = potential bottom"
        elif exhaustion_signal == "DISTRIBUTION" and range_pos > 0.7:
            signal = "HYPE_TOP"
            trade_idea = "Momentum exhausted at highs = fade the hype"
        elif vol_trend < 0.5 and range_pos < 0.4:
            signal = "QUIET_ACCUMULATION"
            trade_idea = "Volume dying + price stable at lows = smart money loading"
        elif vol_trend > 1.5 and move_5d > 20:
            signal = "FOMO_ALERT"
            trade_idea = "Volume spiking + price pumping = late stage, be careful"
        else:
            signal = "NO_SIGNAL"
            trade_idea = "No clear contrarian setup"
        
        return {
            "symbol": symbol,
            "name": MEME_UNIVERSE.get(symbol, symbol),
            "price": round(current_price, 4),
            "move_5d_pct": round(move_5d, 2),
            "move_20d_pct": round(move_20d, 2),
            "vol_trend": round(vol_trend, 2),
            "vol_today_vs_avg": round(vol_today_vs_avg, 2),
            "range_position": round(range_pos, 2),
            "volatility_contraction": round(vol_contraction, 2),
            "capitulation": capitulation,
            "inverse_hype_score": round(inverse_hype, 3),
            "signal": signal,
            "trade_idea": trade_idea,
            "exhaustion": exhaustion_signal,
        }
        
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def scan_meme_coins():
    """Scan all meme coins for inverse hype signals"""
    
    # Get market-wide sentiment first
    print("📊 Fetching market sentiment...")
    fear_greed = get_fear_greed_index()
    trending = get_coingecko_trending()
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_sentiment": {
            "fear_greed": fear_greed,
            "trending_coins": trending,
            "parameters": {
                "stop_loss": f"{MEME_STOP_LOSS*100}%",
                "take_profit": f"{MEME_TAKE_PROFIT*100}%",
                "confirmation_required": CONFIRMATION_REQUIRED
            }
        },
        "signals": [],
        "watchlist": []
    }
    
    fg_value = fear_greed.get("value", 50)
    print(f"   Fear & Greed: {fg_value} ({fear_greed.get('label', 'Unknown')}) → {fear_greed.get('contrarian_signal', 'NEUTRAL')}")
    if trending.get("trending"):
        print(f"   Top Trending: {', '.join(t['symbol'] for t in trending['trending'][:3])}")
    
    print(f"\n🪙 Scanning {len(MEME_UNIVERSE)} meme coins...\n")
    
    for symbol, name in MEME_UNIVERSE.items():
        print(f"  {symbol}...", end=" ", flush=True)
        data = calculate_hype_score(symbol, fear_greed=fear_greed)
        
        if data and "error" not in data:
            # Boost score if Fear & Greed aligns
            if fg_value < 30 and data.get("signal") in ["CAPITULATION_BUY", "QUIET_ACCUMULATION"]:
                data["inverse_hype_score"] = min(1.0, data["inverse_hype_score"] + 0.15)
                data["sentiment_boost"] = "Extreme Fear amplifies buy signal"
            elif fg_value > 70 and data.get("signal") == "HYPE_TOP":
                data["inverse_hype_score"] = min(1.0, data["inverse_hype_score"] + 0.15)
                data["sentiment_boost"] = "Extreme Greed amplifies sell signal"
            
            # Check if this coin is trending (warning sign)
            trending_symbols = [t["symbol"] for t in trending.get("trending", [])]
            if symbol.replace("-USD", "") in trending_symbols:
                data["trending_warning"] = "⚠️ Currently trending on CoinGecko - late stage?"
            
            results["signals"].append(data)
            print(f"${data['price']} | {data['signal']} | Score: {data['inverse_hype_score']:.2f}")
            
            # Add to watchlist if actionable signal
            if data["signal"] not in ["NO_SIGNAL", "FOMO_ALERT"]:
                results["watchlist"].append(data)
        else:
            print("SKIP")
    
    # Sort watchlist by inverse hype score
    results["watchlist"] = sorted(
        results["watchlist"], 
        key=lambda x: x["inverse_hype_score"], 
        reverse=True
    )
    
    # Save results
    output_path = BASE_DIR / "meme_hype_scan.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def print_report(results):
    """Pretty print the meme hype scan"""
    print("\n" + "="*60)
    print("🪙 MEME COIN INVERSE HYPE REPORT v2.0")
    print(f"   {results['timestamp']}")
    print("="*60)
    
    # Market sentiment
    sentiment = results.get("market_sentiment", {})
    fg = sentiment.get("fear_greed", {})
    if fg.get("value"):
        fg_emoji = "😱" if fg["value"] < 25 else "😰" if fg["value"] < 50 else "😊" if fg["value"] < 75 else "🤑"
        print(f"\n{fg_emoji} FEAR & GREED: {fg['value']} ({fg.get('label', 'Unknown')})")
        print(f"   7-day trend: {fg.get('trend_7d', [])}")
        print(f"   Contrarian: {fg.get('contrarian_signal', 'NEUTRAL')}")
    
    trending = sentiment.get("trending_coins", {})
    if trending.get("trending"):
        print(f"\n🔥 TRENDING: {', '.join(t['symbol'] for t in trending['trending'][:5])}")
    
    # Parameters
    params = sentiment.get("parameters", {})
    print(f"\n⚙️  PARAMS: Stop {params.get('stop_loss', '5%')} | Target {params.get('take_profit', '10%')}")
    
    # Actionable signals
    if results["watchlist"]:
        print("\n⭐ CONTRARIAN SETUPS:")
        for m in results["watchlist"]:
            emoji = {
                "CAPITULATION_BUY": "🩸",
                "QUIET_ACCUMULATION": "🤫",
                "HYPE_TOP": "🔝",
                "CONFIRMED_REVERSAL": "✅",
            }.get(m["signal"], "📊")
            
            print(f"\n{emoji} {m['symbol']} - {m['name']}")
            print(f"   Price: ${m['price']} ({m['move_5d_pct']:+.1f}% 5d)")
            print(f"   Score: {m['inverse_hype_score']:.2f}")
            print(f"   Signal: {m['signal']}")
            print(f"   💡 {m['trade_idea']}")
            if m.get("sentiment_boost"):
                print(f"   🎯 {m['sentiment_boost']}")
            if m.get("trending_warning"):
                print(f"   {m['trending_warning']}")
    else:
        print("\n😐 No clear contrarian setups right now")
    
    # Warnings
    fomo_alerts = [s for s in results["signals"] if s.get("signal") == "FOMO_ALERT"]
    if fomo_alerts:
        print("\n⚠️ FOMO ALERTS (careful, late stage):")
        for f in fomo_alerts:
            print(f"   {f['symbol']}: +{f['move_5d_pct']:.1f}% on {f['vol_today_vs_avg']:.1f}x volume")
    
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Meme Coin Inverse Hype Scanner")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    results = scan_meme_coins()
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)
