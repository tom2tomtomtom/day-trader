#!/usr/bin/env python3
"""
Regime Learning Module

Tracks what happens after each regime is detected:
- When we see FEAR → what does market do next 1/5/20 days?
- When we see RISK_OFF → which sectors outperform?
- Build predictive patterns from regime transitions
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "regime_history.jsonl"
OUTCOMES_PATH = BASE_DIR / "regime_outcomes.json"
PATTERNS_PATH = BASE_DIR / "learned_patterns.json"

def load_history():
    """Load regime history"""
    if not HISTORY_PATH.exists():
        return []
    
    history = []
    with open(HISTORY_PATH) as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except:
                pass
    return history

def load_outcomes():
    """Load tracked outcomes"""
    if OUTCOMES_PATH.exists():
        with open(OUTCOMES_PATH) as f:
            return json.load(f)
    return {"pending": [], "completed": []}

def save_outcomes(outcomes):
    with open(OUTCOMES_PATH, "w") as f:
        json.dump(outcomes, f, indent=2)

def load_patterns():
    """Load learned patterns"""
    if PATTERNS_PATH.exists():
        with open(PATTERNS_PATH) as f:
            return json.load(f)
    return {
        "regime_transitions": {},
        "regime_forward_returns": {},
        "best_signals": []
    }

def save_patterns(patterns):
    with open(PATTERNS_PATH, "w") as f:
        json.dump(patterns, f, indent=2)

def record_regime_snapshot(regime_summary, spy_price):
    """Record a regime snapshot for future evaluation"""
    outcomes = load_outcomes()
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "regimes": regime_summary,
        "spy_price_at_signal": spy_price,
        "lookback_periods": {
            "1d": None,
            "5d": None,
            "20d": None
        },
        "status": "pending"
    }
    
    outcomes["pending"].append(snapshot)
    save_outcomes(outcomes)
    return snapshot

def evaluate_pending_outcomes():
    """Check pending outcomes that are now old enough to evaluate"""
    outcomes = load_outcomes()
    patterns = load_patterns()
    
    now = datetime.now()
    spy = yf.Ticker("SPY")
    current_price = float(spy.history(period="1d")['Close'].iloc[-1])
    
    still_pending = []
    newly_completed = []
    
    for snap in outcomes["pending"]:
        snap_time = datetime.fromisoformat(snap["timestamp"])
        days_elapsed = (now - snap_time).days
        
        if days_elapsed >= 1 and snap["lookback_periods"]["1d"] is None:
            # Calculate 1-day return
            ret_1d = (current_price / snap["spy_price_at_signal"] - 1) * 100
            snap["lookback_periods"]["1d"] = round(ret_1d, 3)
        
        if days_elapsed >= 5 and snap["lookback_periods"]["5d"] is None:
            ret_5d = (current_price / snap["spy_price_at_signal"] - 1) * 100
            snap["lookback_periods"]["5d"] = round(ret_5d, 3)
        
        if days_elapsed >= 20 and snap["lookback_periods"]["20d"] is None:
            ret_20d = (current_price / snap["spy_price_at_signal"] - 1) * 100
            snap["lookback_periods"]["20d"] = round(ret_20d, 3)
            snap["status"] = "completed"
            
            # Update patterns
            update_patterns_from_outcome(patterns, snap)
            newly_completed.append(snap)
        else:
            still_pending.append(snap)
    
    outcomes["pending"] = still_pending
    outcomes["completed"].extend(newly_completed)
    
    save_outcomes(outcomes)
    save_patterns(patterns)
    
    return {
        "evaluated": len(newly_completed),
        "still_pending": len(still_pending),
        "total_completed": len(outcomes["completed"])
    }

def update_patterns_from_outcome(patterns, outcome):
    """Update learned patterns from a completed outcome"""
    regimes = outcome["regimes"]
    returns = outcome["lookback_periods"]
    
    # Composite regime key
    regime_key = f"{regimes['volatility']}|{regimes['trend']}|{regimes['risk']}"
    
    if regime_key not in patterns["regime_forward_returns"]:
        patterns["regime_forward_returns"][regime_key] = {
            "count": 0,
            "avg_1d": 0,
            "avg_5d": 0,
            "avg_20d": 0,
            "outcomes": []
        }
    
    p = patterns["regime_forward_returns"][regime_key]
    n = p["count"]
    
    # Update running averages
    p["avg_1d"] = (p["avg_1d"] * n + returns["1d"]) / (n + 1)
    p["avg_5d"] = (p["avg_5d"] * n + returns["5d"]) / (n + 1)
    p["avg_20d"] = (p["avg_20d"] * n + returns["20d"]) / (n + 1)
    p["count"] = n + 1
    p["outcomes"].append(returns)

def get_pattern_insights():
    """Get actionable insights from learned patterns"""
    patterns = load_patterns()
    
    if not patterns["regime_forward_returns"]:
        return {"message": "Not enough data yet. Need completed 20-day outcomes."}
    
    insights = []
    
    # Find best and worst regimes
    sorted_by_20d = sorted(
        patterns["regime_forward_returns"].items(),
        key=lambda x: x[1]["avg_20d"],
        reverse=True
    )
    
    for regime_key, data in sorted_by_20d[:3]:
        if data["count"] >= 3:  # Need enough samples
            insights.append({
                "regime": regime_key,
                "type": "bullish_pattern",
                "avg_20d_return": round(data["avg_20d"], 2),
                "sample_size": data["count"]
            })
    
    for regime_key, data in sorted_by_20d[-3:]:
        if data["count"] >= 3:
            insights.append({
                "regime": regime_key,
                "type": "bearish_pattern",
                "avg_20d_return": round(data["avg_20d"], 2),
                "sample_size": data["count"]
            })
    
    return {
        "total_patterns": len(patterns["regime_forward_returns"]),
        "insights": insights,
        "raw_patterns": {k: {
            "avg_1d": round(v["avg_1d"], 2),
            "avg_5d": round(v["avg_5d"], 2),
            "avg_20d": round(v["avg_20d"], 2),
            "count": v["count"]
        } for k, v in patterns["regime_forward_returns"].items()}
    }

def backfill_with_historical_data():
    """
    Backfill learning data using historical market data
    This lets us learn from past regimes without waiting
    """
    spy = yf.Ticker("SPY")
    vix = yf.Ticker("^VIX")
    
    # Get 1 year of daily data
    spy_hist = spy.history(period="1y", interval="1d")
    vix_hist = vix.history(period="1y", interval="1d")
    
    # Align indexes
    combined = pd.DataFrame({
        'spy_close': spy_hist['Close'],
        'spy_volume': spy_hist['Volume'],
        'vix': vix_hist['Close']
    }).dropna()
    
    # Calculate features
    combined['spy_ret_1d'] = combined['spy_close'].pct_change() * 100
    combined['spy_ret_5d'] = combined['spy_close'].pct_change(5) * 100
    combined['spy_ret_20d'] = combined['spy_close'].pct_change(20) * 100
    combined['spy_vol_20d'] = combined['spy_ret_1d'].rolling(20).std() * np.sqrt(252)
    combined['sma_20'] = combined['spy_close'].rolling(20).mean()
    combined['sma_50'] = combined['spy_close'].rolling(50).mean()
    
    # Forward returns (what we're trying to predict)
    combined['fwd_1d'] = combined['spy_close'].shift(-1) / combined['spy_close'] * 100 - 100
    combined['fwd_5d'] = combined['spy_close'].shift(-5) / combined['spy_close'] * 100 - 100
    combined['fwd_20d'] = combined['spy_close'].shift(-20) / combined['spy_close'] * 100 - 100
    
    patterns = load_patterns()
    
    for idx in range(50, len(combined) - 20):
        row = combined.iloc[idx]
        
        # Determine regimes from historical data
        vix_val = row['vix']
        if vix_val > 30:
            vol_regime = "CRISIS"
        elif vix_val > 20:
            vol_regime = "HIGH"
        elif vix_val > 12:
            vol_regime = "NORMAL"
        else:
            vol_regime = "LOW_COMPLACENT"
        
        # Trend regime
        if row['spy_close'] > row['sma_20'] > row['sma_50']:
            trend_regime = "STRONG_UPTREND"
        elif row['spy_close'] > row['sma_50']:
            trend_regime = "WEAK_UPTREND"
        elif row['spy_close'] < row['sma_20'] < row['sma_50']:
            trend_regime = "STRONG_DOWNTREND"
        elif row['spy_close'] < row['sma_50']:
            trend_regime = "WEAK_DOWNTREND"
        else:
            trend_regime = "RANGING"
        
        # Simplified risk regime based on recent performance
        if row['spy_ret_5d'] > 2:
            risk_regime = "RISK_ON"
        elif row['spy_ret_5d'] < -2:
            risk_regime = "RISK_OFF"
        else:
            risk_regime = "NEUTRAL"
        
        regime_key = f"{vol_regime}|{trend_regime}|{risk_regime}"
        
        if regime_key not in patterns["regime_forward_returns"]:
            patterns["regime_forward_returns"][regime_key] = {
                "count": 0,
                "avg_1d": 0,
                "avg_5d": 0,
                "avg_20d": 0,
                "outcomes": []
            }
        
        p = patterns["regime_forward_returns"][regime_key]
        n = p["count"]
        
        fwd_1d = row['fwd_1d']
        fwd_5d = row['fwd_5d']
        fwd_20d = row['fwd_20d']
        
        if pd.notna(fwd_20d):
            p["avg_1d"] = (p["avg_1d"] * n + fwd_1d) / (n + 1)
            p["avg_5d"] = (p["avg_5d"] * n + fwd_5d) / (n + 1)
            p["avg_20d"] = (p["avg_20d"] * n + fwd_20d) / (n + 1)
            p["count"] = n + 1
    
    save_patterns(patterns)
    
    return {
        "patterns_learned": len(patterns["regime_forward_returns"]),
        "total_observations": sum(p["count"] for p in patterns["regime_forward_returns"].values())
    }

# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regime Learning")
    parser.add_argument("command", choices=["backfill", "evaluate", "insights", "record"])
    
    args = parser.parse_args()
    
    if args.command == "backfill":
        print("Backfilling with historical data...")
        result = backfill_with_historical_data()
        print(json.dumps(result, indent=2))
    
    elif args.command == "evaluate":
        result = evaluate_pending_outcomes()
        print(json.dumps(result, indent=2))
    
    elif args.command == "insights":
        insights = get_pattern_insights()
        print(json.dumps(insights, indent=2))
    
    elif args.command == "record":
        # Would be called after analyze
        print("Use regime_detector.py to analyze and auto-record")

if __name__ == "__main__":
    main()
