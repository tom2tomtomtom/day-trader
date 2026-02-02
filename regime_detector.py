#!/usr/bin/env python3
"""
Market Behavior & Regime Detection Model — GLOBAL EDITION

NOT predicting price direction — detecting WHAT the market is doing:
- Volatility regime (calm/normal/volatile/crisis)
- Trend regime (strong trend/weak trend/ranging)
- Risk regime (risk-on/risk-off/neutral)
- Sentiment regime (euphoria/fear/neutral)
- Flow regime (accumulation/distribution/neutral)
- Regional regimes (US, Europe, Asia-Pacific)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
BASE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = BASE_DIR / "regime_state.json"
HISTORY_PATH = BASE_DIR / "regime_history.jsonl"

# ============================================================
# MARKET HOURS & REGIONS
# ============================================================

MARKET_SCHEDULES = {
    # (open_hour_utc, close_hour_utc, weekdays_only)
    "US": {"open": 14, "close": 21, "tz_offset": 0},      # NYSE/NASDAQ: 9:30-16:00 ET = 14:30-21:00 UTC
    "Europe": {"open": 8, "close": 16, "tz_offset": 0},   # LSE/Euronext: ~8:00-16:30 UTC
    "Japan": {"open": 0, "close": 6, "tz_offset": 0},     # TSE: 9:00-15:00 JST = 0:00-6:00 UTC
    "HongKong": {"open": 1, "close": 8, "tz_offset": 0},  # HKEX: 9:30-16:00 HKT = 1:30-8:00 UTC
    "Australia": {"open": 23, "close": 5, "tz_offset": 0}, # ASX: 10:00-16:00 AEDT = 23:00-05:00 UTC (crosses midnight)
    "Korea": {"open": 0, "close": 6, "tz_offset": 0},     # KRX: 9:00-15:30 KST = 0:00-6:30 UTC
}

def get_market_status():
    """Check which markets are currently open"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    status = {}
    for market, schedule in MARKET_SCHEDULES.items():
        open_h = schedule["open"]
        close_h = schedule["close"]
        
        # Handle markets that cross midnight (like ASX)
        if open_h > close_h:
            is_open = hour >= open_h or hour < close_h
        else:
            is_open = open_h <= hour < close_h
        
        # Weekends closed
        if weekday >= 5:
            is_open = False
        
        status[market] = {
            "open": is_open,
            "hours": f"{open_h:02d}:00-{close_h:02d}:00 UTC"
        }
    
    return status

def get_freshness_weight(region):
    """Weight data based on how recently that market was open"""
    status = get_market_status()
    if status.get(region, {}).get("open", False):
        return 1.0  # Currently open - full weight
    
    # Could enhance this to calculate hours since close
    # For now, give slightly lower weight to closed markets
    return 0.8

# ============================================================
# DATA FETCHERS
# ============================================================

# Tickers organized by region
TICKERS_BY_REGION = {
    "US": {
        # Core indices
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IWM": "Russell 2000 (small caps)",
        "DIA": "Dow Jones",
        # Volatility
        "^VIX": "VIX (fear index)",
        # Bonds/Rates
        "TLT": "20+ Year Treasury",
        "IEF": "7-10 Year Treasury",
        "HYG": "High Yield Corporate Bonds",
        "LQD": "Investment Grade Bonds",
        # Sectors (for rotation)
        "XLF": "Financials",
        "XLK": "Technology",
        "XLE": "Energy",
        "XLU": "Utilities (defensive)",
        "XLP": "Consumer Staples (defensive)",
        # Risk indicators
        "GLD": "Gold",
        "UUP": "US Dollar",
    },
    "Europe": {
        "^STOXX50E": "Euro Stoxx 50",
        "EWG": "Germany (DAX proxy)",
        "EWU": "UK (FTSE proxy)",
        "FEZ": "Eurozone ETF",
        "EWQ": "France (CAC proxy)",
        "EWI": "Italy",
        "EWP": "Spain",
    },
    "Asia_Japan": {
        "^N225": "Nikkei 225",
        "EWJ": "Japan ETF",
        "FXY": "Japanese Yen",
        "DXJ": "Japan Hedged Equity",
    },
    "Asia_China": {
        "^HSI": "Hang Seng (HK)",
        "FXI": "China Large-Cap",
        "KWEB": "China Internet",
        "MCHI": "MSCI China",
        "GXC": "S&P China",
    },
    "Asia_Pacific": {
        "^AXJO": "ASX 200 (Australia)",
        "EWA": "Australia ETF",
        "^KS11": "KOSPI (Korea)",
        "EWY": "Korea ETF",
        "EWT": "Taiwan ETF",
        "EWS": "Singapore ETF",
        "INDA": "India ETF",
        "^STI": "Straits Times (Singapore)",
    },
    "Global": {
        # Crypto (24/7)
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        # Global risk proxies
        "VEU": "All-World ex-US",
        "ACWI": "All Country World",
        "EEM": "Emerging Markets",
        "EFA": "EAFE (Developed ex-US)",
    },
}

def get_market_data(regions=None):
    """Fetch comprehensive market data, optionally filtered by region"""
    if regions is None:
        regions = list(TICKERS_BY_REGION.keys())
    
    # Flatten tickers for selected regions
    tickers = {}
    for region in regions:
        if region in TICKERS_BY_REGION:
            for ticker, name in TICKERS_BY_REGION[region].items():
                tickers[ticker] = {"name": name, "region": region}
    
    data = {}
    market_status = get_market_status()
    
    for ticker, info in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3mo", interval="1d")
            if not hist.empty:
                region = info["region"]
                # Map region to market status key
                market_key = {
                    "US": "US",
                    "Europe": "Europe",
                    "Asia_Japan": "Japan",
                    "Asia_China": "HongKong",
                    "Asia_Pacific": "Australia",
                    "Global": None,  # 24/7 or global ETFs
                }.get(region)
                
                is_market_open = market_status.get(market_key, {}).get("open", False) if market_key else True
                freshness = 1.0 if is_market_open else 0.8
                
                data[ticker] = {
                    "name": info["name"],
                    "region": region,
                    "market_open": is_market_open,
                    "freshness_weight": freshness,
                    "current": float(hist['Close'].iloc[-1]),
                    "prev_close": float(hist['Close'].iloc[-2]) if len(hist) > 1 else None,
                    "change_1d": float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100) if len(hist) > 1 else 0,
                    "change_5d": float((hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100) if len(hist) > 5 else 0,
                    "change_20d": float((hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100) if len(hist) > 20 else 0,
                    "high_20d": float(hist['Close'].tail(20).max()),
                    "low_20d": float(hist['Close'].tail(20).min()),
                    "avg_volume_20d": float(hist['Volume'].tail(20).mean()),
                    "volume_ratio": float(hist['Volume'].iloc[-1] / hist['Volume'].tail(20).mean()) if hist['Volume'].tail(20).mean() > 0 else 1,
                    "volatility_20d": float(hist['Close'].tail(20).pct_change().std() * np.sqrt(252) * 100),  # Annualized
                    "history": hist
                }
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    return data

def get_options_data(symbol="SPY"):
    """Get options market data for sentiment"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get all expiration dates
        expirations = ticker.options[:3]  # Next 3 expirations
        
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        
        for exp in expirations:
            opt = ticker.option_chain(exp)
            total_call_volume += opt.calls['volume'].sum()
            total_put_volume += opt.puts['volume'].sum()
            total_call_oi += opt.calls['openInterest'].sum()
            total_put_oi += opt.puts['openInterest'].sum()
        
        put_call_ratio_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 1
        put_call_ratio_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1
        
        return {
            "put_call_ratio_volume": round(put_call_ratio_volume, 3),
            "put_call_ratio_oi": round(put_call_ratio_oi, 3),
            "total_call_volume": int(total_call_volume),
            "total_put_volume": int(total_put_volume),
            "interpretation": "bearish" if put_call_ratio_volume > 1.2 else "bullish" if put_call_ratio_volume < 0.7 else "neutral"
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# REGIME DETECTION
# ============================================================

def detect_volatility_regime(data):
    """
    Volatility Regime:
    - CRISIS: VIX > 30, realized vol spiking
    - HIGH: VIX 20-30
    - NORMAL: VIX 12-20
    - LOW: VIX < 12 (complacency)
    """
    vix = data.get("^VIX", {}).get("current", 15)
    spy_vol = data.get("SPY", {}).get("volatility_20d", 15)
    
    # VIX vs realized vol spread (when VIX >> realized, fear is elevated)
    vol_spread = vix - spy_vol
    
    if vix > 30:
        regime = "CRISIS"
        confidence = min(1.0, (vix - 30) / 20 + 0.7)
    elif vix > 20:
        regime = "HIGH"
        confidence = 0.6 + (vix - 20) / 20
    elif vix > 12:
        regime = "NORMAL"
        confidence = 0.7
    else:
        regime = "LOW_COMPLACENT"
        confidence = 0.6 + (12 - vix) / 12
    
    return {
        "regime": regime,
        "confidence": round(confidence, 2),
        "vix": round(vix, 2),
        "realized_vol": round(spy_vol, 2),
        "vol_spread": round(vol_spread, 2),
        "note": "VIX > realized = fear premium" if vol_spread > 5 else "VIX ≈ realized = fair" if abs(vol_spread) < 5 else "VIX < realized = complacent"
    }

def detect_trend_regime(data):
    """
    Trend Regime:
    - STRONG_UP: Multiple indices trending up, above MAs
    - WEAK_UP: Mixed signals, slight uptrend
    - RANGING: Choppy, no clear direction
    - WEAK_DOWN: Mixed signals, slight downtrend
    - STRONG_DOWN: Multiple indices trending down
    """
    indices = ["SPY", "QQQ", "IWM", "DIA"]
    
    scores = []
    for idx in indices:
        if idx in data:
            d = data[idx]
            # Score based on multiple timeframes
            score = 0
            if d["change_1d"] > 0: score += 1
            if d["change_5d"] > 0: score += 2
            if d["change_20d"] > 0: score += 3
            if d["current"] > d["high_20d"] * 0.98: score += 2  # Near 20d high
            if d["current"] < d["low_20d"] * 1.02: score -= 2  # Near 20d low
            scores.append(score)
    
    avg_score = np.mean(scores) if scores else 0
    
    if avg_score > 5:
        regime = "STRONG_UPTREND"
    elif avg_score > 2:
        regime = "WEAK_UPTREND"
    elif avg_score > -2:
        regime = "RANGING"
    elif avg_score > -5:
        regime = "WEAK_DOWNTREND"
    else:
        regime = "STRONG_DOWNTREND"
    
    # Calculate trend alignment (are indices moving together?)
    if len(scores) > 1:
        alignment = 1 - (np.std(scores) / (np.mean(np.abs(scores)) + 0.01))
    else:
        alignment = 0.5
    
    return {
        "regime": regime,
        "score": round(avg_score, 2),
        "alignment": round(max(0, alignment), 2),
        "indices": {idx: data.get(idx, {}).get("change_5d", 0) for idx in indices}
    }

def detect_risk_regime(data):
    """
    Risk Regime (risk-on vs risk-off):
    - RISK_ON: Equities up, junk bonds up, defensives lagging, yen weak
    - RISK_OFF: Treasuries up, gold up, yen strong, defensives outperforming
    """
    signals = {}
    
    # High yield vs investment grade spread behavior
    hyg = data.get("HYG", {})
    lqd = data.get("LQD", {})
    if hyg and lqd and hyg.get("change_5d") is not None and lqd.get("change_5d") is not None:
        credit_spread_signal = hyg["change_5d"] - lqd["change_5d"]
        signals["credit"] = "risk_on" if credit_spread_signal > 0.5 else "risk_off" if credit_spread_signal < -0.5 else "neutral"
    
    # Defensive vs cyclical sectors
    xlp = data.get("XLP", {}).get("change_5d", 0)  # Consumer staples (defensive)
    xlu = data.get("XLU", {}).get("change_5d", 0)  # Utilities (defensive)
    xlk = data.get("XLK", {}).get("change_5d", 0)  # Tech (cyclical)
    xlf = data.get("XLF", {}).get("change_5d", 0)  # Financials (cyclical)
    
    defensive_avg = (xlp + xlu) / 2
    cyclical_avg = (xlk + xlf) / 2
    rotation_signal = cyclical_avg - defensive_avg
    signals["rotation"] = "risk_on" if rotation_signal > 1 else "risk_off" if rotation_signal < -1 else "neutral"
    
    # Gold behavior
    gld = data.get("GLD", {}).get("change_5d", 0)
    signals["gold"] = "risk_off" if gld > 2 else "risk_on" if gld < -1 else "neutral"
    
    # Small cap vs large cap (risk appetite)
    iwm = data.get("IWM", {}).get("change_5d", 0)
    spy = data.get("SPY", {}).get("change_5d", 0)
    small_cap_signal = iwm - spy
    signals["small_cap"] = "risk_on" if small_cap_signal > 1 else "risk_off" if small_cap_signal < -1 else "neutral"
    
    # Aggregate
    risk_on_count = sum(1 for v in signals.values() if v == "risk_on")
    risk_off_count = sum(1 for v in signals.values() if v == "risk_off")
    
    if risk_on_count >= 3:
        regime = "RISK_ON"
    elif risk_off_count >= 3:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"
    
    return {
        "regime": regime,
        "signals": signals,
        "risk_on_signals": risk_on_count,
        "risk_off_signals": risk_off_count
    }

def detect_sentiment_regime(data, options_data):
    """
    Sentiment Regime:
    - EUPHORIA: Low VIX + low put/call + high volume + new highs
    - FEAR: High VIX + high put/call + selling pressure
    - NEUTRAL: Mixed signals
    """
    vix = data.get("^VIX", {}).get("current", 15)
    put_call = options_data.get("put_call_ratio_volume", 1.0)
    spy_vol_ratio = data.get("SPY", {}).get("volume_ratio", 1.0)
    
    # Fear/Greed scoring
    score = 0
    
    # VIX component
    if vix < 12:
        score += 2  # Complacent/greedy
    elif vix > 25:
        score -= 2  # Fearful
    
    # Put/Call
    if put_call < 0.7:
        score += 2  # Bullish options activity
    elif put_call > 1.2:
        score -= 2  # Bearish options activity
    
    # Volume (high volume on up days = bullish)
    spy_change = data.get("SPY", {}).get("change_1d", 0)
    if spy_vol_ratio > 1.5 and spy_change > 0:
        score += 1
    elif spy_vol_ratio > 1.5 and spy_change < 0:
        score -= 1
    
    if score >= 3:
        regime = "EUPHORIA"
    elif score <= -3:
        regime = "FEAR"
    elif score > 0:
        regime = "GREED"
    elif score < 0:
        regime = "CAUTION"
    else:
        regime = "NEUTRAL"
    
    return {
        "regime": regime,
        "score": score,
        "components": {
            "vix_signal": "complacent" if vix < 12 else "fearful" if vix > 25 else "normal",
            "put_call": put_call,
            "volume_signal": "high" if spy_vol_ratio > 1.5 else "low" if spy_vol_ratio < 0.7 else "normal"
        }
    }

def detect_flow_regime(data):
    """
    Flow Regime (accumulation vs distribution):
    Based on volume patterns and price action
    """
    spy = data.get("SPY", {})
    
    if "history" not in spy:
        return {"regime": "UNKNOWN", "note": "No historical data"}
    
    hist = spy["history"].tail(20)
    
    # Calculate accumulation/distribution
    up_volume = hist[hist['Close'] > hist['Open']]['Volume'].sum()
    down_volume = hist[hist['Close'] < hist['Open']]['Volume'].sum()
    
    ad_ratio = up_volume / down_volume if down_volume > 0 else 2
    
    if ad_ratio > 1.5:
        regime = "ACCUMULATION"
    elif ad_ratio < 0.67:
        regime = "DISTRIBUTION"
    else:
        regime = "NEUTRAL"
    
    return {
        "regime": regime,
        "up_volume_pct": round(up_volume / (up_volume + down_volume) * 100, 1) if (up_volume + down_volume) > 0 else 50,
        "ad_ratio": round(ad_ratio, 2)
    }

def detect_regional_regimes(data):
    """
    Detect regime by region — useful for global rotation and lead/lag signals
    """
    regions = {}
    
    # Define key tickers per region for regime detection
    region_indices = {
        "US": ["SPY", "QQQ", "IWM"],
        "Europe": ["^STOXX50E", "EWG", "EWU"],
        "Japan": ["^N225", "EWJ"],
        "China": ["^HSI", "FXI", "KWEB"],
        "Asia_Pacific": ["^AXJO", "EWA", "^KS11", "EWY"],
        "Emerging": ["EEM", "INDA", "EWT"],
    }
    
    market_status = get_market_status()
    
    for region, tickers in region_indices.items():
        scores = []
        changes_1d = []
        changes_5d = []
        available = []
        
        for ticker in tickers:
            if ticker in data:
                d = data[ticker]
                score = 0
                if d.get("change_1d", 0) > 0: score += 1
                if d.get("change_5d", 0) > 0: score += 2
                if d.get("change_20d", 0) > 0: score += 3
                scores.append(score * d.get("freshness_weight", 1.0))
                changes_1d.append(d.get("change_1d", 0))
                changes_5d.append(d.get("change_5d", 0))
                available.append(ticker)
        
        if not scores:
            regions[region] = {"regime": "NO_DATA", "tickers": []}
            continue
        
        avg_score = np.mean(scores)
        avg_1d = np.mean(changes_1d)
        avg_5d = np.mean(changes_5d)
        
        if avg_score > 4:
            regime = "BULLISH"
        elif avg_score > 2:
            regime = "SLIGHTLY_BULLISH"
        elif avg_score > -2:
            regime = "NEUTRAL"
        elif avg_score > -4:
            regime = "SLIGHTLY_BEARISH"
        else:
            regime = "BEARISH"
        
        # Check if market is currently open
        market_map = {
            "US": "US",
            "Europe": "Europe", 
            "Japan": "Japan",
            "China": "HongKong",
            "Asia_Pacific": "Australia",
            "Emerging": None,
        }
        market_key = market_map.get(region)
        is_open = market_status.get(market_key, {}).get("open", False) if market_key else False
        
        regions[region] = {
            "regime": regime,
            "score": round(avg_score, 2),
            "change_1d": round(avg_1d, 2),
            "change_5d": round(avg_5d, 2),
            "market_open": is_open,
            "tickers": available
        }
    
    # Global risk-on/off assessment across all regions
    bullish_regions = sum(1 for r in regions.values() if r.get("regime") in ["BULLISH", "SLIGHTLY_BULLISH"])
    bearish_regions = sum(1 for r in regions.values() if r.get("regime") in ["BEARISH", "SLIGHTLY_BEARISH"])
    
    if bullish_regions >= 4:
        global_regime = "GLOBAL_RISK_ON"
    elif bearish_regions >= 4:
        global_regime = "GLOBAL_RISK_OFF"
    else:
        global_regime = "MIXED"
    
    return {
        "regions": regions,
        "global_regime": global_regime,
        "bullish_count": bullish_regions,
        "bearish_count": bearish_regions
    }

def get_global_summary(data):
    """Quick summary of all markets for display"""
    market_status = get_market_status()
    
    summary = []
    key_tickers = {
        "🇺🇸 US": "SPY",
        "🇪🇺 Europe": "^STOXX50E",
        "🇯🇵 Japan": "^N225",
        "🇭🇰 HK/China": "^HSI",
        "🇦🇺 Australia": "^AXJO",
        "🇰🇷 Korea": "^KS11",
        "🌍 Emerging": "EEM",
        "₿ Bitcoin": "BTC-USD",
    }
    
    for label, ticker in key_tickers.items():
        if ticker in data:
            d = data[ticker]
            change = d.get("change_1d", 0)
            arrow = "🟢" if change > 0.5 else "🔴" if change < -0.5 else "⚪"
            open_indicator = "●" if d.get("market_open", False) else "○"
            summary.append(f"{open_indicator} {label}: {change:+.2f}%")
    
    return summary

# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_market(regions=None):
    """Complete market behavior analysis — now global!"""
    print("Fetching global market data...")
    data = get_market_data(regions)
    print(f"  Loaded {len(data)} tickers across all regions")
    
    print("Checking market hours...")
    market_status = get_market_status()
    open_markets = [m for m, s in market_status.items() if s["open"]]
    print(f"  Open now: {', '.join(open_markets) if open_markets else 'None (weekend/off-hours)'}")
    
    print("Fetching options data...")
    options = get_options_data("SPY")
    
    print("Detecting regimes...")
    
    volatility = detect_volatility_regime(data)
    trend = detect_trend_regime(data)
    risk = detect_risk_regime(data)
    sentiment = detect_sentiment_regime(data, options)
    flow = detect_flow_regime(data)
    regional = detect_regional_regimes(data)
    
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_status": market_status,
        "regimes": {
            "volatility": volatility,
            "trend": trend,
            "risk": risk,
            "sentiment": sentiment,
            "flow": flow,
            "regional": regional
        },
        "summary": {
            "volatility": volatility["regime"],
            "trend": trend["regime"],
            "risk": risk["regime"],
            "sentiment": sentiment["regime"],
            "flow": flow["regime"],
            "global": regional["global_regime"]
        },
        "key_data": {
            "vix": data.get("^VIX", {}).get("current"),
            "spy_1d": data.get("SPY", {}).get("change_1d"),
            "spy_5d": data.get("SPY", {}).get("change_5d"),
            "nikkei_1d": data.get("^N225", {}).get("change_1d"),
            "hsi_1d": data.get("^HSI", {}).get("change_1d"),
            "asx_1d": data.get("^AXJO", {}).get("change_1d"),
            "stoxx_1d": data.get("^STOXX50E", {}).get("change_1d"),
            "put_call": options.get("put_call_ratio_volume"),
            "btc_1d": data.get("BTC-USD", {}).get("change_1d")
        },
        "global_summary": get_global_summary(data)
    }
    
    # Generate actionable interpretation
    interpretation = generate_interpretation(analysis)
    analysis["interpretation"] = interpretation
    
    # Save state
    save_state(analysis)
    
    return analysis

def generate_interpretation(analysis):
    """Generate human-readable interpretation"""
    regimes = analysis["summary"]
    regional = analysis["regimes"].get("regional", {})
    
    lines = []
    
    # Global regime first
    global_regime = regimes.get("global", "MIXED")
    if global_regime == "GLOBAL_RISK_ON":
        lines.append("🌍 GLOBAL RISK-ON: Most regions bullish, favorable for equities")
    elif global_regime == "GLOBAL_RISK_OFF":
        lines.append("🌍 GLOBAL RISK-OFF: Most regions bearish, favor defensives")
    else:
        lines.append("🌍 GLOBAL MIXED: Regional divergence, be selective by geography")
    
    # Regional highlights
    region_data = regional.get("regions", {})
    bullish_regions = [r for r, d in region_data.items() if d.get("regime") in ["BULLISH", "SLIGHTLY_BULLISH"]]
    bearish_regions = [r for r, d in region_data.items() if d.get("regime") in ["BEARISH", "SLIGHTLY_BEARISH"]]
    
    if bullish_regions:
        lines.append(f"  📈 Leading: {', '.join(bullish_regions)}")
    if bearish_regions:
        lines.append(f"  📉 Lagging: {', '.join(bearish_regions)}")
    
    lines.append("")
    
    # Volatility context
    if regimes["volatility"] == "CRISIS":
        lines.append("⚠️ CRISIS MODE: Extreme volatility, expect wild swings")
    elif regimes["volatility"] == "HIGH":
        lines.append("🔶 Elevated volatility - size positions smaller")
    elif regimes["volatility"] == "LOW_COMPLACENT":
        lines.append("😴 Low volatility - complacency risk, breakout potential")
    
    # Trend context
    if "STRONG" in regimes["trend"]:
        direction = "UP" if "UP" in regimes["trend"] else "DOWN"
        lines.append(f"📈 Strong trend {direction} - favor trend-following")
    elif regimes["trend"] == "RANGING":
        lines.append("↔️ Ranging market - mean reversion strategies")
    
    # Risk context
    if regimes["risk"] == "RISK_ON":
        lines.append("🟢 Risk-on environment - favor growth/cyclicals")
    elif regimes["risk"] == "RISK_OFF":
        lines.append("🔴 Risk-off environment - favor defensives/bonds")
    
    # Sentiment context
    if regimes["sentiment"] == "EUPHORIA":
        lines.append("🎉 Euphoria detected - contrarian caution (tops form in euphoria)")
    elif regimes["sentiment"] == "FEAR":
        lines.append("😱 Fear detected - contrarian opportunity (bottoms form in fear)")
    
    # Composite view
    bullish_signals = sum([
        regimes["trend"] in ["STRONG_UPTREND", "WEAK_UPTREND"],
        regimes["risk"] == "RISK_ON",
        regimes["sentiment"] in ["GREED", "NEUTRAL"],
        regimes["flow"] == "ACCUMULATION",
        global_regime == "GLOBAL_RISK_ON"
    ])
    
    lines.append("")
    if bullish_signals >= 4:
        lines.append("📊 OVERALL: Strongly bullish environment")
    elif bullish_signals >= 3:
        lines.append("📊 OVERALL: Bullish environment")
    elif bullish_signals <= 1:
        lines.append("📊 OVERALL: Bearish/cautious environment")
    else:
        lines.append("📊 OVERALL: Mixed signals - be selective")
    
    return "\n".join(lines)

def save_state(analysis):
    """Save current state and append to history"""
    # Current state (remove pandas history objects before saving)
    clean_analysis = {}
    for k, v in analysis.items():
        if k == "regimes":
            clean_regimes = {}
            for rk, rv in v.items():
                if isinstance(rv, dict):
                    clean_regimes[rk] = {kk: vv for kk, vv in rv.items() if kk != "history" and not isinstance(vv, pd.DataFrame)}
                else:
                    clean_regimes[rk] = rv
            clean_analysis[k] = clean_regimes
        else:
            clean_analysis[k] = v
    
    with open(STATE_PATH, "w") as f:
        json.dump(clean_analysis, f, indent=2, default=str)
    
    # Append to history
    with open(HISTORY_PATH, "a") as f:
        compact = {
            "timestamp": analysis["timestamp"],
            "regimes": analysis["summary"],
            "vix": analysis["key_data"].get("vix"),
            "spy_1d": analysis["key_data"].get("spy_1d"),
            "asx_1d": analysis["key_data"].get("asx_1d"),
            "nikkei_1d": analysis["key_data"].get("nikkei_1d")
        }
        f.write(json.dumps(compact) + "\n")

def load_history(days=30):
    """Load regime history for learning"""
    if not HISTORY_PATH.exists():
        return []
    
    history = []
    with open(HISTORY_PATH) as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except:
                pass
    
    return history[-days*24:]  # Assume hourly snapshots

# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Market Regime Detector — Global Edition")
    parser.add_argument("command", choices=["analyze", "status", "history", "markets"], nargs="?", default="analyze")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--region", type=str, help="Filter by region (US, Europe, Asia_Japan, Asia_China, Asia_Pacific, Global)")
    
    args = parser.parse_args()
    
    if args.command == "markets":
        # Quick market status check
        status = get_market_status()
        print("\n🌍 GLOBAL MARKET STATUS")
        print("=" * 40)
        for market, info in status.items():
            icon = "🟢" if info["open"] else "🔴"
            print(f"  {icon} {market:12} {info['hours']} {'(OPEN)' if info['open'] else ''}")
        print()
        return
    
    if args.command == "analyze":
        regions = [args.region] if args.region else None
        analysis = analyze_market(regions)
        
        if args.json:
            # Remove history objects for JSON output
            clean = {k: v for k, v in analysis.items()}
            print(json.dumps(clean, indent=2, default=str))
        else:
            print("\n" + "="*60)
            print("🌍 GLOBAL MARKET REGIME ANALYSIS")
            print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
            print("="*60)
            
            # Market status
            print("\n⏰ MARKET HOURS:")
            for market, info in analysis["market_status"].items():
                icon = "🟢" if info["open"] else "⚪"
                print(f"  {icon} {market}")
            
            # Global summary
            print("\n🌐 GLOBAL SNAPSHOT:")
            for line in analysis.get("global_summary", []):
                print(f"  {line}")
            
            # Regional regimes
            regional = analysis["regimes"].get("regional", {})
            if regional:
                print("\n🗺️ REGIONAL REGIMES:")
                for region, data in regional.get("regions", {}).items():
                    regime = data.get("regime", "?")
                    change = data.get("change_1d", 0)
                    icon = "🟢" if "BULLISH" in regime else "🔴" if "BEARISH" in regime else "⚪"
                    open_ind = "●" if data.get("market_open") else "○"
                    print(f"  {open_ind} {icon} {region:12} {regime:18} ({change:+.2f}% 1d)")
            
            print("\n📊 CORE REGIMES:")
            for regime_type, regime_value in analysis["summary"].items():
                if regime_type != "global":  # Already shown above
                    print(f"  {regime_type.upper():12} → {regime_value}")
            
            print("\n📈 KEY DATA:")
            for key, value in analysis["key_data"].items():
                if value is not None:
                    label = key.replace("_", " ").upper()
                    print(f"  {label:12} → {value:.2f}" if isinstance(value, float) else f"  {label:12} → {value}")
            
            print("\n💡 INTERPRETATION:")
            print(analysis["interpretation"])
            print()
    
    elif args.command == "status":
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No state file. Run 'analyze' first.")
    
    elif args.command == "history":
        history = load_history()
        for h in history[-10:]:
            global_r = h['regimes'].get('global', 'N/A')
            print(f"{h['timestamp']}: Vol={h['regimes']['volatility']}, Trend={h['regimes']['trend']}, Global={global_r}")

if __name__ == "__main__":
    main()
