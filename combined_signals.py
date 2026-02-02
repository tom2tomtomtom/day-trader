#!/usr/bin/env python3
"""
Combined Signal System v1.0

Integrates multiple proven strategies:
1. Fear & Greed filter (only trade at extremes)
2. Mean Reversion (Bollinger Bands) - our best performer
3. RSI Adaptive Momentum
4. Volume Confirmation
5. Trend Filter (MA crossover context)

The key insight: Individual signals are mediocre.
Combined signals with sentiment filter = edge.
"""

import json
import requests
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yfinance as yf

BASE_DIR = Path(__file__).parent
COMBINED_SIGNALS_PATH = BASE_DIR / "combined_signals.json"

# === SENTIMENT LAYER ===

def get_fear_greed():
    """Get Fear & Greed Index"""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = resp.json()
        if data.get("data"):
            return int(data["data"][0]["value"])
    except:
        pass
    return 50  # Default neutral


def get_vix():
    """Get VIX for volatility regime"""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return 20  # Default normal


# === TECHNICAL INDICATORS ===

def calculate_bollinger(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    current = prices.iloc[-1]
    bb_position = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    
    return {
        "upper": float(upper.iloc[-1]),
        "middle": float(sma.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "position": float(bb_position),  # 0 = at lower, 1 = at upper
        "width": float((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1] * 100)
    }


def calculate_rsi(prices, period=14):
    """Calculate RSI with adaptive thresholds"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = float(rsi.iloc[-1])
    rsi_5d_ago = float(rsi.iloc[-5]) if len(rsi) >= 5 else current_rsi
    
    # Adaptive thresholds based on trend
    sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else prices.mean()
    in_uptrend = float(prices.iloc[-1]) > float(sma_50)
    
    if in_uptrend:
        oversold = 40  # Higher threshold in uptrends
        overbought = 80
    else:
        oversold = 20  # Lower threshold in downtrends
        overbought = 60
    
    return {
        "value": current_rsi,
        "oversold_threshold": oversold,
        "overbought_threshold": overbought,
        "is_oversold": bool(current_rsi < oversold),
        "is_overbought": bool(current_rsi > overbought),
        "divergence": "bullish" if current_rsi > rsi_5d_ago and float(prices.iloc[-1]) < float(prices.iloc[-5]) else
                     "bearish" if current_rsi < rsi_5d_ago and float(prices.iloc[-1]) > float(prices.iloc[-5]) else "none"
    }


def calculate_volume_signal(hist):
    """Volume confirmation signal"""
    vol_20d = hist['Volume'].iloc[-20:].mean() if len(hist) >= 20 else hist['Volume'].mean()
    vol_today = hist['Volume'].iloc[-1]
    vol_5d = hist['Volume'].iloc[-5:].mean()
    
    rel_vol = vol_today / vol_20d if vol_20d > 0 else 1
    vol_trend = vol_5d / vol_20d if vol_20d > 0 else 1
    
    return {
        "relative_volume": float(rel_vol),
        "volume_trend": float(vol_trend),
        "is_surge": bool(rel_vol > 1.5),
        "is_declining": bool(vol_trend < 0.7),
        "is_confirming": bool(rel_vol > 1.2)  # Volume confirms move
    }


def calculate_trend(prices):
    """Trend context using MA crossover"""
    if len(prices) < 50:
        return {"trend": "unknown", "strength": 0}
    
    sma_10 = prices.rolling(10).mean().iloc[-1]
    sma_20 = prices.rolling(20).mean().iloc[-1]
    sma_50 = prices.rolling(50).mean().iloc[-1]
    current = prices.iloc[-1]
    
    if current > sma_10 > sma_20 > sma_50:
        trend = "strong_up"
        strength = 1.0
    elif current > sma_20 > sma_50:
        trend = "up"
        strength = 0.6
    elif current < sma_10 < sma_20 < sma_50:
        trend = "strong_down"
        strength = -1.0
    elif current < sma_20 < sma_50:
        trend = "down"
        strength = -0.6
    else:
        trend = "sideways"
        strength = 0
    
    return {
        "trend": trend,
        "strength": strength,
        "sma_10": float(sma_10),
        "sma_20": float(sma_20),
        "sma_50": float(sma_50),
        "above_sma_50": bool(current > sma_50)
    }


# === COMBINED SIGNAL ENGINE ===

def generate_combined_signal(symbol, fear_greed=None, vix=None):
    """
    Generate combined signal using multiple factors
    
    Signal strength -1 to +1:
    - Positive = bullish
    - Negative = bearish
    - Near zero = no trade
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo", interval="1d")
        
        if len(hist) < 30:
            return None
        
        prices = hist['Close']
        current_price = float(prices.iloc[-1])
        
        # Get sentiment if not provided
        if fear_greed is None:
            fear_greed = get_fear_greed()
        if vix is None:
            vix = get_vix()
        
        # Calculate all indicators
        bb = calculate_bollinger(prices)
        rsi = calculate_rsi(prices)
        vol = calculate_volume_signal(hist)
        trend = calculate_trend(prices)
        
        # === SIGNAL SCORING ===
        
        signal_score = 0
        reasons = []
        
        # 1. BOLLINGER BAND SCORE (-0.3 to +0.3)
        if bb["position"] < 0.1:  # At lower band
            signal_score += 0.3
            reasons.append("BB oversold")
        elif bb["position"] > 0.9:  # At upper band
            signal_score -= 0.3
            reasons.append("BB overbought")
        elif bb["width"] < 10:  # Squeeze
            reasons.append("BB squeeze (breakout pending)")
        
        # 2. RSI SCORE (-0.25 to +0.25)
        if rsi["is_oversold"]:
            signal_score += 0.25
            reasons.append(f"RSI oversold ({rsi['value']:.0f})")
        elif rsi["is_overbought"]:
            signal_score -= 0.25
            reasons.append(f"RSI overbought ({rsi['value']:.0f})")
        
        # RSI divergence bonus
        if rsi["divergence"] == "bullish":
            signal_score += 0.15
            reasons.append("Bullish RSI divergence")
        elif rsi["divergence"] == "bearish":
            signal_score -= 0.15
            reasons.append("Bearish RSI divergence")
        
        # 3. TREND FILTER (-0.2 to +0.2)
        signal_score += trend["strength"] * 0.2
        if trend["trend"] != "sideways":
            reasons.append(f"Trend: {trend['trend']}")
        
        # 4. VOLUME CONFIRMATION (multiplier)
        if vol["is_confirming"]:
            signal_score *= 1.2
            reasons.append(f"Volume confirms ({vol['relative_volume']:.1f}x)")
        elif vol["is_declining"] and abs(signal_score) > 0.2:
            signal_score *= 0.8
            reasons.append("Weak volume")
        
        # 5. FEAR & GREED FILTER (critical!)
        fg_multiplier = 1.0
        fg_note = None
        
        if fear_greed < 25:  # Extreme Fear
            if signal_score > 0:  # Bullish signal
                fg_multiplier = 1.5
                fg_note = "🟢 Extreme Fear amplifies BUY"
            else:  # Bearish signal in fear = ignore
                fg_multiplier = 0.5
                fg_note = "Fear override: reduce short bias"
        elif fear_greed > 75:  # Extreme Greed
            if signal_score < 0:  # Bearish signal
                fg_multiplier = 1.5
                fg_note = "🔴 Extreme Greed amplifies SELL"
            else:  # Bullish signal in greed = ignore
                fg_multiplier = 0.5
                fg_note = "Greed override: reduce long bias"
        
        signal_score *= fg_multiplier
        if fg_note:
            reasons.append(fg_note)
        
        # 6. VIX ADJUSTMENT
        if vix > 30:  # High volatility
            signal_score *= 0.8  # Reduce position size signal
            reasons.append(f"High VIX ({vix:.0f}) - reduce size")
        
        # === FINAL SIGNAL ===
        
        if signal_score > 0.4:
            action = "STRONG_BUY"
        elif signal_score > 0.2:
            action = "BUY"
        elif signal_score < -0.4:
            action = "STRONG_SELL"
        elif signal_score < -0.2:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Position size suggestion (1-5 scale)
        position_size = min(5, max(1, int(abs(signal_score) * 10)))
        
        return {
            "symbol": symbol,
            "price": current_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_score": round(signal_score, 3),
            "action": action,
            "position_size": position_size,
            "reasons": reasons,
            "indicators": {
                "bollinger": bb,
                "rsi": rsi,
                "volume": vol,
                "trend": trend
            },
            "sentiment": {
                "fear_greed": fear_greed,
                "vix": vix
            }
        }
        
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def scan_universe(symbols, fear_greed=None, vix=None):
    """Scan multiple symbols with combined signals"""
    
    if fear_greed is None:
        fear_greed = get_fear_greed()
    if vix is None:
        vix = get_vix()
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_context": {
            "fear_greed": fear_greed,
            "fear_greed_label": "Extreme Fear" if fear_greed < 25 else "Fear" if fear_greed < 45 else "Neutral" if fear_greed < 55 else "Greed" if fear_greed < 75 else "Extreme Greed",
            "vix": vix,
            "vix_regime": "Crisis" if vix > 30 else "Elevated" if vix > 20 else "Normal" if vix > 12 else "Complacent"
        },
        "signals": [],
        "buys": [],
        "sells": []
    }
    
    print(f"📊 Market Context: F&G={fear_greed} ({results['market_context']['fear_greed_label']}), VIX={vix:.1f}\n")
    
    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        sig = generate_combined_signal(symbol, fear_greed, vix)
        
        if sig and "error" not in sig:
            results["signals"].append(sig)
            
            if sig["action"] in ["BUY", "STRONG_BUY"]:
                results["buys"].append(sig)
                print(f"✅ {sig['action']} (score: {sig['signal_score']:.2f})")
            elif sig["action"] in ["SELL", "STRONG_SELL"]:
                results["sells"].append(sig)
                print(f"🔴 {sig['action']} (score: {sig['signal_score']:.2f})")
            else:
                print(f"⏸️ {sig['action']}")
        else:
            print("SKIP")
    
    # Sort by signal strength
    results["buys"] = sorted(results["buys"], key=lambda x: x["signal_score"], reverse=True)
    results["sells"] = sorted(results["sells"], key=lambda x: x["signal_score"])
    
    # Save results
    with open(COMBINED_SIGNALS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def print_report(results):
    """Print combined signals report"""
    print("\n" + "="*60)
    print("📈 COMBINED SIGNALS REPORT")
    print(f"   {results['timestamp']}")
    print("="*60)
    
    ctx = results["market_context"]
    fg_emoji = "😱" if ctx["fear_greed"] < 25 else "😰" if ctx["fear_greed"] < 45 else "😐" if ctx["fear_greed"] < 55 else "😊" if ctx["fear_greed"] < 75 else "🤑"
    print(f"\n{fg_emoji} Fear & Greed: {ctx['fear_greed']} ({ctx['fear_greed_label']})")
    print(f"📉 VIX: {ctx['vix']:.1f} ({ctx['vix_regime']})")
    
    if results["buys"]:
        print(f"\n✅ BUY SIGNALS ({len(results['buys'])}):")
        for s in results["buys"][:5]:
            print(f"\n   {s['symbol']} @ ${s['price']:.2f}")
            print(f"   Score: {s['signal_score']:.2f} | Size: {s['position_size']}/5")
            print(f"   Reasons: {', '.join(s['reasons'][:3])}")
    
    if results["sells"]:
        print(f"\n🔴 SELL SIGNALS ({len(results['sells'])}):")
        for s in results["sells"][:5]:
            print(f"\n   {s['symbol']} @ ${s['price']:.2f}")
            print(f"   Score: {s['signal_score']:.2f}")
            print(f"   Reasons: {', '.join(s['reasons'][:3])}")
    
    if not results["buys"] and not results["sells"]:
        print("\n⏸️ No strong signals - market in wait mode")
    
    print()


# Default universe
UNIVERSE = [
    # Major ETFs
    "SPY", "QQQ", "IWM",
    # Top stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Memes
    "DOGE-USD", "MSTR", "COIN"
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined Signal System")
    parser.add_argument("--symbol", "-s", help="Single symbol to analyze")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    if args.symbol:
        result = generate_combined_signal(args.symbol)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{result['symbol']} @ ${result['price']:.2f}")
            print(f"Signal: {result['action']} (score: {result['signal_score']:.2f})")
            print(f"Position size: {result['position_size']}/5")
            print(f"Reasons: {', '.join(result['reasons'])}")
    else:
        results = scan_universe(UNIVERSE)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_report(results)
