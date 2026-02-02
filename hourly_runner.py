#!/usr/bin/env python3
"""
Hourly Global Market Monitor & Auto-Trader

Runs every hour when ANY major market is open:
1. Detects current regime (global + regional)
2. Matches against learned patterns
3. Generates signal
4. Executes paper trades based on signal
5. Logs everything for learning
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from model import get_current_regime, get_expected_returns, generate_signal
from paper_trader import execute_trade, get_status, load_portfolio, get_current_price
from learner import record_regime_snapshot, evaluate_pending_outcomes
from regime_detector import get_market_status, analyze_market

RUN_LOG = BASE_DIR / "hourly_log.jsonl"

# Position sizing rules
MAX_POSITION_PCT = 0.25  # Max 25% in single position
MIN_TRADE_AMOUNT = 1000  # Minimum trade size

# Regional ETF mapping - what to trade when each region is active
REGIONAL_ETFS = {
    "US": "SPY",
    "Europe": "FEZ",
    "Japan": "EWJ",
    "HongKong": "FXI",  # China proxy
    "Australia": "EWA",
    "Korea": "EWY",
}

def get_active_markets():
    """Get list of currently open markets"""
    status = get_market_status()
    return [market for market, info in status.items() if info["open"]]

def is_any_market_open():
    """Check if any major market is open"""
    return len(get_active_markets()) > 0

def log_run(data):
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")

def check_position_limits(portfolio, symbol):
    """Check if we can add to a position"""
    portfolio_value = portfolio["cash"]
    for sym, pos in portfolio["positions"].items():
        price = get_current_price(sym)
        if price:
            portfolio_value += price * pos["shares"]
    
    if portfolio_value <= 0:
        return False, "Portfolio value is zero"
    
    # Check current position size
    if symbol in portfolio["positions"]:
        price = get_current_price(symbol)
        if price:
            current_value = price * portfolio["positions"][symbol]["shares"]
            current_pct = current_value / portfolio_value
            if current_pct >= MAX_POSITION_PCT:
                return False, f"Position already at {current_pct:.1%} (max {MAX_POSITION_PCT:.0%})"
    
    return True, None

def get_best_trading_opportunity(global_analysis, portfolio):
    """
    Determine best trading opportunity based on:
    1. Which markets are open
    2. Regional regime strength
    3. Current positions
    """
    active_markets = get_active_markets()
    regional = global_analysis.get("regimes", {}).get("regional", {})
    regions = regional.get("regions", {})
    
    opportunities = []
    
    for market in active_markets:
        etf = REGIONAL_ETFS.get(market)
        if not etf:
            continue
            
        # Map market to region key
        region_map = {
            "US": "US",
            "Europe": "Europe",
            "Japan": "Japan",
            "HongKong": "China",
            "Australia": "Asia_Pacific",
            "Korea": "Asia_Pacific",
        }
        region_key = region_map.get(market, market)
        
        region_data = regions.get(region_key, {})
        regime = region_data.get("regime", "NEUTRAL")
        score = region_data.get("score", 0)
        change_1d = region_data.get("change_1d", 0)
        
        # Score the opportunity
        opp_score = score
        
        # Bonus for strong signals
        if regime in ["BULLISH"]:
            opp_score += 2
        elif regime in ["BEARISH"]:
            opp_score -= 2
        
        # Prefer markets we don't already have positions in (diversification)
        if etf not in portfolio.get("positions", {}):
            opp_score += 1
        
        opportunities.append({
            "market": market,
            "etf": etf,
            "region": region_key,
            "regime": regime,
            "score": opp_score,
            "change_1d": change_1d
        })
    
    # Sort by score (highest first for buys)
    opportunities.sort(key=lambda x: x["score"], reverse=True)
    
    return opportunities

def run_hourly():
    """Main hourly run - now global!"""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Check which markets are open
    active_markets = get_active_markets()
    any_market_open = len(active_markets) > 0
    
    # Run global analysis
    print("Running global market analysis...")
    global_analysis = analyze_market()
    
    # Get signal from the model (US-focused for core signal)
    signal = generate_signal()
    regime = signal["regime"]
    
    # Record for learning
    spy_price = regime["data"].get("spy", 0)
    record_regime_snapshot(signal["regime"], spy_price)
    
    # Evaluate any pending outcomes
    eval_result = evaluate_pending_outcomes()
    
    # Get portfolio status
    portfolio = load_portfolio()
    status = get_status()
    
    # Get regional opportunities
    opportunities = get_best_trading_opportunity(global_analysis, portfolio)
    
    run_data = {
        "timestamp": timestamp,
        "active_markets": active_markets,
        "any_market_open": any_market_open,
        "global_regime": global_analysis.get("summary", {}).get("global", "UNKNOWN"),
        "us_regime": regime["regime_key"],
        "us_signal": signal["action"],
        "confidence": signal["confidence"],
        "spy_price": spy_price,
        "vix": regime["data"].get("vix", 0),
        "portfolio_value": status["portfolio_value"],
        "cash": status["cash"],
        "positions": list(portfolio["positions"].keys()),
        "regional_snapshot": {
            opp["market"]: f"{opp['regime']} ({opp['change_1d']:+.2f}%)"
            for opp in opportunities[:4]
        },
        "trades_executed": []
    }
    
    # Execute trades when markets are open
    if any_market_open and opportunities:
        best_opp = opportunities[0]
        
        # Determine action based on regime and signal
        if best_opp["regime"] in ["BULLISH", "SLIGHTLY_BULLISH"]:
            action = "BUY"
            pct = 0.10 if best_opp["regime"] == "BULLISH" else 0.05
        elif best_opp["regime"] in ["BEARISH", "SLIGHTLY_BEARISH"]:
            # Check if we have position to reduce
            if best_opp["etf"] in portfolio["positions"]:
                action = "SELL"
                pct = 0.50
            else:
                action = None
        else:
            action = None
        
        # Also consider core US signal if US is open
        if "US" in active_markets:
            if signal["action"] == "STRONG_BUY" and signal["confidence"] == "HIGH":
                can_trade, reason = check_position_limits(portfolio, "SPY")
                if can_trade and portfolio["cash"] > MIN_TRADE_AMOUNT:
                    trade_result = execute_trade(
                        "BUY", "SPY", 0.20,
                        f"STRONG_BUY signal ({signal['rationale']})",
                        regime
                    )
                    if trade_result and "error" not in trade_result:
                        run_data["trades_executed"].append(trade_result)
            
            elif signal["action"] == "BUY":
                can_trade, reason = check_position_limits(portfolio, "SPY")
                if can_trade and portfolio["cash"] > MIN_TRADE_AMOUNT:
                    trade_result = execute_trade(
                        "BUY", "SPY", 0.10,
                        f"BUY signal ({signal['rationale']})",
                        regime
                    )
                    if trade_result and "error" not in trade_result:
                        run_data["trades_executed"].append(trade_result)
            
            elif signal["action"] == "REDUCE" and "SPY" in portfolio["positions"]:
                trade_result = execute_trade(
                    "SELL", "SPY", 0.5,
                    f"REDUCE signal ({signal['rationale']})",
                    regime
                )
                if trade_result and "error" not in trade_result:
                    run_data["trades_executed"].append(trade_result)
        
        # Trade regional ETF if not US or if best opportunity is elsewhere
        if action and best_opp["market"] != "US":
            etf = best_opp["etf"]
            can_trade, reason = check_position_limits(portfolio, etf)
            
            if action == "BUY" and can_trade and portfolio["cash"] > MIN_TRADE_AMOUNT:
                trade_result = execute_trade(
                    "BUY", etf, pct,
                    f"{best_opp['regime']} in {best_opp['market']} ({best_opp['change_1d']:+.2f}% today)",
                    {"regional": best_opp}
                )
                if trade_result and "error" not in trade_result:
                    run_data["trades_executed"].append(trade_result)
            
            elif action == "SELL" and etf in portfolio["positions"]:
                trade_result = execute_trade(
                    "SELL", etf, pct,
                    f"{best_opp['regime']} in {best_opp['market']}",
                    {"regional": best_opp}
                )
                if trade_result and "error" not in trade_result:
                    run_data["trades_executed"].append(trade_result)
    
    log_run(run_data)
    
    return run_data

def print_summary(run_data):
    """Print human-readable summary"""
    print("\n" + "="*60)
    print(f"🌍 GLOBAL HOURLY RUN: {run_data['timestamp']}")
    print("="*60)
    
    # Market status
    active = run_data.get("active_markets", [])
    if active:
        print(f"\n🟢 Markets Open: {', '.join(active)}")
    else:
        print(f"\n🔴 All Markets Closed")
    
    # Global regime
    print(f"\n🌍 Global: {run_data.get('global_regime', 'N/A')}")
    print(f"🇺🇸 US Regime: {run_data.get('us_regime', 'N/A')}")
    print(f"📊 US Signal: {run_data.get('us_signal', 'N/A')} ({run_data.get('confidence', 'N/A')})")
    
    # Regional snapshot
    regional = run_data.get("regional_snapshot", {})
    if regional:
        print(f"\n🗺️ Regional Snapshot:")
        for market, status in regional.items():
            print(f"   {market}: {status}")
    
    # Key data
    print(f"\n📈 SPY: ${run_data.get('spy_price', 0):.2f} | VIX: {run_data.get('vix', 0):.1f}")
    
    # Portfolio
    print(f"\n💰 Portfolio: ${run_data.get('portfolio_value', 0):,.2f}")
    print(f"   Cash: ${run_data.get('cash', 0):,.2f}")
    positions = run_data.get("positions", [])
    if positions:
        print(f"   Positions: {', '.join(positions)}")
    
    # Trades
    trades = run_data.get("trades_executed", [])
    if trades:
        print(f"\n🔔 TRADES EXECUTED:")
        for t in trades:
            print(f"   {t['action']} {t.get('shares', 0)} {t['symbol']} @ ${t.get('price', 0):.2f}")
            print(f"   Reason: {t.get('reason', 'N/A')}")
    else:
        print(f"\n   No trades this hour")
    
    print()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Global Hourly Market Monitor")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run even if all markets closed")
    
    args = parser.parse_args()
    
    # Check if any market is open (unless forced)
    if not args.force and not is_any_market_open():
        print("All markets closed. Use --force to run anyway.")
        return
    
    run_data = run_hourly()
    
    if args.json:
        print(json.dumps(run_data, indent=2, default=str))
    else:
        print_summary(run_data)

if __name__ == "__main__":
    main()
