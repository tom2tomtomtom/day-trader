#!/usr/bin/env python3
"""
Market Behavior Model - Unified System

Combines:
1. Regime detection (what is the market doing?)
2. Historical pattern matching (what happened next in similar regimes?)
3. Actionable signals based on learned patterns
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
PATTERNS_PATH = BASE_DIR / "learned_patterns.json"
STATE_PATH = BASE_DIR / "current_state.json"
LOG_PATH = BASE_DIR / "signal_log.jsonl"

# ============================================================
# REGIME DETECTION
# ============================================================

def get_current_regime():
    """Detect current market regime"""
    
    # Fetch data
    spy = yf.Ticker("SPY")
    vix = yf.Ticker("^VIX")
    
    spy_hist = spy.history(period="3mo", interval="1d")
    vix_hist = vix.history(period="1mo", interval="1d")
    
    current_spy = float(spy_hist['Close'].iloc[-1])
    current_vix = float(vix_hist['Close'].iloc[-1])
    
    sma_20 = float(spy_hist['Close'].rolling(20).mean().iloc[-1])
    sma_50 = float(spy_hist['Close'].rolling(50).mean().iloc[-1])
    ret_5d = float((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-5] - 1) * 100)
    ret_1d = float((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-2] - 1) * 100)
    
    # Volatility regime
    if current_vix > 30:
        vol_regime = "CRISIS"
    elif current_vix > 20:
        vol_regime = "HIGH"
    elif current_vix > 12:
        vol_regime = "NORMAL"
    else:
        vol_regime = "LOW_COMPLACENT"
    
    # Trend regime
    if current_spy > sma_20 and sma_20 > sma_50:
        trend_regime = "STRONG_UPTREND"
    elif current_spy > sma_50:
        trend_regime = "WEAK_UPTREND"
    elif current_spy < sma_20 and sma_20 < sma_50:
        trend_regime = "STRONG_DOWNTREND"
    elif current_spy < sma_50:
        trend_regime = "WEAK_DOWNTREND"
    else:
        trend_regime = "RANGING"
    
    # Risk regime
    if ret_5d > 2:
        risk_regime = "RISK_ON"
    elif ret_5d < -2:
        risk_regime = "RISK_OFF"
    else:
        risk_regime = "NEUTRAL"
    
    return {
        "volatility": vol_regime,
        "trend": trend_regime,
        "risk": risk_regime,
        "regime_key": f"{vol_regime}|{trend_regime}|{risk_regime}",
        "data": {
            "spy": current_spy,
            "vix": current_vix,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ret_1d": ret_1d,
            "ret_5d": ret_5d
        }
    }

# ============================================================
# PATTERN MATCHING
# ============================================================

def load_patterns():
    """Load learned patterns"""
    if PATTERNS_PATH.exists():
        with open(PATTERNS_PATH) as f:
            return json.load(f)
    return {"regime_forward_returns": {}}

def get_expected_returns(regime_key):
    """Get expected forward returns for a regime"""
    patterns = load_patterns()
    
    if regime_key in patterns["regime_forward_returns"]:
        return patterns["regime_forward_returns"][regime_key]
    
    # Try partial matches
    vol, trend, risk = regime_key.split("|")
    
    partial_matches = []
    for key, data in patterns["regime_forward_returns"].items():
        k_vol, k_trend, k_risk = key.split("|")
        
        # Score similarity
        score = 0
        if k_vol == vol: score += 3
        if k_trend == trend: score += 2
        if k_risk == risk: score += 1
        
        if score >= 3:  # At least vol match + something else
            partial_matches.append((score, data))
    
    if partial_matches:
        # Weighted average by similarity score
        total_weight = sum(m[0] * m[1]["count"] for m in partial_matches)
        if total_weight > 0:
            avg_1d = sum(m[0] * m[1]["count"] * m[1]["avg_1d"] for m in partial_matches) / total_weight
            avg_5d = sum(m[0] * m[1]["count"] * m[1]["avg_5d"] for m in partial_matches) / total_weight
            avg_20d = sum(m[0] * m[1]["count"] * m[1]["avg_20d"] for m in partial_matches) / total_weight
            return {
                "avg_1d": round(avg_1d, 3),
                "avg_5d": round(avg_5d, 3),
                "avg_20d": round(avg_20d, 3),
                "count": sum(m[1]["count"] for m in partial_matches),
                "match_type": "partial"
            }
    
    return None

# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal():
    """Generate trading signal based on regime and patterns"""
    
    regime = get_current_regime()
    expected = get_expected_returns(regime["regime_key"])
    
    signal = {
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "expected_returns": expected
    }
    
    # Generate actionable signal
    if expected:
        # Contrarian logic: extreme regimes often mean-revert
        if regime["volatility"] in ["CRISIS", "HIGH"] and "DOWNTREND" in regime["trend"]:
            if expected["avg_20d"] > 5:
                signal["action"] = "STRONG_BUY"
                signal["confidence"] = "HIGH"
                signal["rationale"] = f"Fear regime with historical +{expected['avg_20d']:.1f}% avg return"
            else:
                signal["action"] = "BUY"
                signal["confidence"] = "MEDIUM"
                signal["rationale"] = "Elevated fear often leads to bounces"
        
        elif regime["volatility"] == "LOW_COMPLACENT" and regime["trend"] == "STRONG_UPTREND":
            signal["action"] = "REDUCE"
            signal["confidence"] = "MEDIUM"
            signal["rationale"] = "Complacency + extended uptrend = lower forward returns"
        
        elif regime["volatility"] == "NORMAL" and regime["trend"] == "STRONG_UPTREND":
            signal["action"] = "HOLD"
            signal["confidence"] = "MEDIUM"
            signal["rationale"] = f"Healthy uptrend, expected +{expected['avg_20d']:.1f}% (modest)"
        
        elif "DOWNTREND" in regime["trend"] and regime["risk"] == "RISK_OFF":
            signal["action"] = "ACCUMULATE"
            signal["confidence"] = "MEDIUM"
            signal["rationale"] = "Risk-off selling often overdone"
        
        else:
            signal["action"] = "NEUTRAL"
            signal["confidence"] = "LOW"
            signal["rationale"] = "Mixed signals, no clear edge"
    
    else:
        signal["action"] = "NEUTRAL"
        signal["confidence"] = "LOW"
        signal["rationale"] = "Insufficient pattern data for this regime"
    
    # Risk management overlay
    if regime["volatility"] == "CRISIS":
        signal["position_size"] = "SMALL"
        signal["risk_note"] = "High vol = reduce position sizes"
    elif regime["volatility"] == "HIGH":
        signal["position_size"] = "REDUCED"
        signal["risk_note"] = "Elevated vol = careful sizing"
    else:
        signal["position_size"] = "NORMAL"
        signal["risk_note"] = None
    
    return signal

def log_signal(signal):
    """Log signal for tracking"""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(signal, default=str) + "\n")

# ============================================================
# MAIN
# ============================================================

def analyze():
    """Full market analysis"""
    signal = generate_signal()
    log_signal(signal)
    
    # Save current state
    with open(STATE_PATH, "w") as f:
        json.dump(signal, f, indent=2, default=str)
    
    return signal

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Market Behavior Model")
    parser.add_argument("command", choices=["analyze", "signal", "status"], nargs="?", default="analyze")
    parser.add_argument("--json", action="store_true")
    
    args = parser.parse_args()
    
    if args.command in ["analyze", "signal"]:
        signal = analyze()
        
        if args.json:
            print(json.dumps(signal, indent=2, default=str))
        else:
            print("\n" + "="*60)
            print("MARKET BEHAVIOR MODEL")
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("="*60)
            
            r = signal["regime"]
            print(f"\n📊 CURRENT REGIME:")
            print(f"   Volatility: {r['volatility']} (VIX: {r['data']['vix']:.1f})")
            print(f"   Trend:      {r['trend']}")
            print(f"   Risk:       {r['risk']} (5d: {r['data']['ret_5d']:+.2f}%)")
            print(f"   SPY:        ${r['data']['spy']:.2f}")
            
            if signal.get("expected_returns"):
                e = signal["expected_returns"]
                print(f"\n📈 EXPECTED RETURNS (based on {e.get('count', '?')} similar periods):")
                print(f"   1-day:  {e['avg_1d']:+.2f}%")
                print(f"   5-day:  {e['avg_5d']:+.2f}%")
                print(f"   20-day: {e['avg_20d']:+.2f}%")
            
            print(f"\n🎯 SIGNAL:")
            print(f"   Action:     {signal['action']}")
            print(f"   Confidence: {signal['confidence']}")
            print(f"   Rationale:  {signal['rationale']}")
            
            if signal.get("position_size"):
                print(f"\n⚠️  RISK:")
                print(f"   Position size: {signal['position_size']}")
                if signal.get("risk_note"):
                    print(f"   Note: {signal['risk_note']}")
            
            print()
    
    elif args.command == "status":
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("No state. Run 'analyze' first.")

if __name__ == "__main__":
    main()
