#!/usr/bin/env python3
"""
Sentiment Data Collection

Multiple sources for market sentiment:
1. Fear & Greed Index (CNN)
2. Options flow (put/call ratios)
3. News sentiment (via web scraping)
4. Social sentiment indicators
"""

import json
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error
import re
import yfinance as yf

BASE_DIR = Path(__file__).parent
SENTIMENT_LOG = BASE_DIR / "sentiment_log.jsonl"

def get_fear_greed_index():
    """
    Scrape CNN Fear & Greed Index
    Returns: 0-100 (0 = extreme fear, 100 = extreme greed)
    """
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            if "fear_and_greed" in data:
                fg = data["fear_and_greed"]
                return {
                    "value": fg.get("score", 50),
                    "rating": fg.get("rating", "Neutral"),
                    "previous_close": fg.get("previous_close", None),
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        pass
    
    # Fallback: estimate from VIX
    try:
        vix = yf.Ticker("^VIX")
        vix_val = float(vix.history(period="1d")['Close'].iloc[-1])
        
        # Simple mapping: VIX 10 = 80 (greed), VIX 30 = 20 (fear)
        # Linear interpolation
        estimated = max(0, min(100, 100 - (vix_val - 10) * 3))
        
        return {
            "value": round(estimated),
            "rating": "Extreme Fear" if estimated < 25 else "Fear" if estimated < 45 else "Neutral" if estimated < 55 else "Greed" if estimated < 75 else "Extreme Greed",
            "source": "VIX_estimate",
            "vix": vix_val,
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {"error": "Could not get Fear & Greed data"}

def get_options_sentiment():
    """Get options market sentiment"""
    try:
        spy = yf.Ticker("SPY")
        
        expirations = spy.options[:3]
        
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        
        for exp in expirations:
            chain = spy.option_chain(exp)
            total_call_vol += chain.calls['volume'].sum()
            total_put_vol += chain.puts['volume'].sum()
            total_call_oi += chain.calls['openInterest'].sum()
            total_put_oi += chain.puts['openInterest'].sum()
        
        pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else 1
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1
        
        # Interpret
        if pcr_vol > 1.3:
            sentiment = "Very Bearish"
        elif pcr_vol > 1.1:
            sentiment = "Bearish"
        elif pcr_vol > 0.9:
            sentiment = "Neutral"
        elif pcr_vol > 0.7:
            sentiment = "Bullish"
        else:
            sentiment = "Very Bullish"
        
        return {
            "put_call_ratio_volume": round(pcr_vol, 3),
            "put_call_ratio_oi": round(pcr_oi, 3),
            "sentiment": sentiment,
            "call_volume": int(total_call_vol),
            "put_volume": int(total_put_vol),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

def get_vix_term_structure():
    """
    VIX term structure (contango vs backwardation)
    Contango (VIX < VIX futures) = complacent
    Backwardation (VIX > VIX futures) = fear/hedging demand
    """
    try:
        vix = yf.Ticker("^VIX")
        vix_val = float(vix.history(period="1d")['Close'].iloc[-1])
        
        # VIX3M (3-month VIX)
        vix3m = yf.Ticker("^VIX3M")
        vix3m_val = float(vix3m.history(period="1d")['Close'].iloc[-1])
        
        term_spread = vix3m_val - vix_val
        
        if term_spread > 3:
            structure = "Strong Contango (complacent)"
        elif term_spread > 0:
            structure = "Contango (normal)"
        elif term_spread > -3:
            structure = "Flat/Slight Backwardation"
        else:
            structure = "Strong Backwardation (fear)"
        
        return {
            "vix": round(vix_val, 2),
            "vix3m": round(vix3m_val, 2),
            "term_spread": round(term_spread, 2),
            "structure": structure,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

def get_breadth_indicators():
    """
    Market breadth indicators
    - Advance/Decline (approximated via sector ETFs)
    - New highs vs new lows
    """
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"]
    
    advancing = 0
    declining = 0
    
    for sector in sectors:
        try:
            ticker = yf.Ticker(sector)
            hist = ticker.history(period="5d")
            if len(hist) >= 2:
                change = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                if change > 0.1:
                    advancing += 1
                elif change < -0.1:
                    declining += 1
        except:
            pass
    
    total = advancing + declining
    if total > 0:
        ad_ratio = advancing / total
        
        if ad_ratio > 0.8:
            breadth = "Very Strong (broad rally)"
        elif ad_ratio > 0.6:
            breadth = "Strong"
        elif ad_ratio > 0.4:
            breadth = "Mixed"
        elif ad_ratio > 0.2:
            breadth = "Weak"
        else:
            breadth = "Very Weak (broad selloff)"
    else:
        breadth = "Unknown"
        ad_ratio = 0.5
    
    return {
        "advancing_sectors": advancing,
        "declining_sectors": declining,
        "ad_ratio": round(ad_ratio, 2),
        "breadth": breadth,
        "timestamp": datetime.now().isoformat()
    }

def get_all_sentiment():
    """Collect all sentiment data"""
    
    fear_greed = get_fear_greed_index()
    options = get_options_sentiment()
    vix_term = get_vix_term_structure()
    breadth = get_breadth_indicators()
    
    # Composite sentiment score (0-100)
    scores = []
    
    if "value" in fear_greed:
        scores.append(fear_greed["value"])
    
    if "put_call_ratio_volume" in options:
        # Convert PCR to 0-100 scale (lower PCR = higher score)
        pcr = options["put_call_ratio_volume"]
        pcr_score = max(0, min(100, (1.5 - pcr) * 100))
        scores.append(pcr_score)
    
    if "ad_ratio" in breadth:
        scores.append(breadth["ad_ratio"] * 100)
    
    composite = sum(scores) / len(scores) if scores else 50
    
    if composite < 25:
        overall = "EXTREME_FEAR"
    elif composite < 40:
        overall = "FEAR"
    elif composite < 60:
        overall = "NEUTRAL"
    elif composite < 75:
        overall = "GREED"
    else:
        overall = "EXTREME_GREED"
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "composite_score": round(composite, 1),
        "overall_sentiment": overall,
        "components": {
            "fear_greed": fear_greed,
            "options": options,
            "vix_term_structure": vix_term,
            "breadth": breadth
        }
    }
    
    # Log
    with open(SENTIMENT_LOG, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")
    
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Market Sentiment")
    parser.add_argument("--json", action="store_true")
    
    args = parser.parse_args()
    
    sentiment = get_all_sentiment()
    
    if args.json:
        print(json.dumps(sentiment, indent=2))
    else:
        print("\n" + "="*50)
        print("MARKET SENTIMENT")
        print("="*50)
        
        print(f"\n📊 Composite Score: {sentiment['composite_score']}/100")
        print(f"📈 Overall: {sentiment['overall_sentiment']}")
        
        c = sentiment["components"]
        
        if "value" in c["fear_greed"]:
            print(f"\n😨 Fear & Greed: {c['fear_greed']['value']} ({c['fear_greed']['rating']})")
        
        if "put_call_ratio_volume" in c["options"]:
            print(f"📉 Put/Call Ratio: {c['options']['put_call_ratio_volume']} ({c['options']['sentiment']})")
        
        if "structure" in c["vix_term_structure"]:
            print(f"📈 VIX Term: {c['vix_term_structure']['structure']}")
        
        if "breadth" in c["breadth"]:
            print(f"📊 Breadth: {c['breadth']['breadth']} ({c['breadth']['advancing_sectors']}/{c['breadth']['declining_sectors']+c['breadth']['advancing_sectors']} advancing)")
        
        print()

if __name__ == "__main__":
    main()
