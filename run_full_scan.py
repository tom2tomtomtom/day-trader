#!/usr/bin/env python3
"""
Full Market Scan - Runs all scanners and combines results

1. Edge Scanner (WSB, short interest, options flow, sector rotation)
2. Combined Signals (technicals + sentiment)
3. Merge results for final recommendations

Usage: python3 run_full_scan.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# Import scanners
from edge_scanner import (
    scan_for_edge, 
    get_wsb_sentiment,
    get_sector_rotation,
    EDGE_UNIVERSE
)
from combined_signals import scan_universe, UNIVERSE

BASE_DIR = Path(__file__).parent
FULL_SCAN_PATH = BASE_DIR / "full_scan.json"


def merge_signals(combined: dict, edge: dict) -> dict:
    """
    Merge edge signals with combined technical signals
    
    Edge factors boost or dampen technical signals:
    - High SI + bullish technical = amplified
    - WSB momentum + bullish = amplified (but watch for tops)
    - Insider buying + any signal = amplified
    - Bearish options flow + bullish technical = dampened
    """
    
    # Create lookup for edge opportunities
    edge_lookup = {}
    for opp in edge.get("edge_opportunities", []):
        symbol = opp["symbol"]
        if symbol not in edge_lookup:
            edge_lookup[symbol] = opp
    
    # Create lookup for WSB momentum
    wsb_symbols = {s["symbol"]: s for s in edge.get("wsb_momentum", [])}
    
    # Merge signals
    merged = []
    
    for sig in combined.get("signals", []):
        symbol = sig["symbol"]
        
        # Base score from technical analysis
        score = sig.get("signal_score", 0)
        reasons = sig.get("reasons", []).copy()
        
        # Apply edge factors
        edge_boost = 0
        
        # Check edge opportunities
        if symbol in edge_lookup:
            opp = edge_lookup[symbol]
            edge_score = opp.get("edge_score", 0)
            
            # Short squeeze potential
            si = opp.get("short_interest", {})
            if si.get("squeeze_potential") in ["HIGH", "MEDIUM"]:
                if score > 0:  # Bullish technical
                    edge_boost += 0.15
                    reasons.append(f"🚀 Squeeze setup ({si.get('short_percent', 0):.0f}% SI)")
            
            # Options flow
            opt = opp.get("options", {})
            if opt.get("options_signal") in ["VERY_BULLISH", "BULLISH"]:
                if score > 0:
                    edge_boost += 0.1
                    reasons.append(f"📊 Options bullish (P/C: {opt.get('put_call_ratio', 0):.2f})")
            elif opt.get("options_signal") in ["VERY_BEARISH", "BEARISH"]:
                if score > 0:
                    edge_boost -= 0.1  # Dampen bullish signal
                    reasons.append(f"⚠️ Options bearish (P/C: {opt.get('put_call_ratio', 0):.2f})")
            
            if opt.get("unusual_activity"):
                reasons.append("👀 Unusual options activity")
            
            # Earnings momentum
            earn = opp.get("earnings", {})
            if earn.get("earnings_momentum") == "STRONG":
                edge_boost += 0.1
                reasons.append(f"📈 {earn.get('beat_streak', 0)}Q beat streak")
        
        # WSB momentum (double-edged)
        if symbol in wsb_symbols:
            wsb = wsb_symbols[symbol]
            change = wsb.get("mention_change_pct", 0)
            
            if change > 100:
                # Extreme hype - be careful, might be near top
                if score > 0:
                    edge_boost += 0.05  # Small boost
                    reasons.append(f"🦍 WSB exploding (+{change:.0f}%) ⚠️ watch for top")
            elif change > 50:
                if score > 0:
                    edge_boost += 0.1
                    reasons.append(f"🦍 WSB momentum (+{change:.0f}%)")
        
        # Apply boost
        final_score = score + edge_boost
        
        # Recalculate action
        if final_score > 0.5:
            action = "STRONG_BUY"
        elif final_score > 0.25:
            action = "BUY"
        elif final_score < -0.5:
            action = "STRONG_SELL"
        elif final_score < -0.25:
            action = "SELL"
        else:
            action = "HOLD"
        
        merged.append({
            **sig,
            "signal_score": round(final_score, 3),
            "original_score": sig.get("signal_score", 0),
            "edge_boost": round(edge_boost, 3),
            "action": action,
            "reasons": reasons,
            "has_edge": symbol in edge_lookup or symbol in wsb_symbols
        })
    
    return sorted(merged, key=lambda x: x["signal_score"], reverse=True)


def run_full_scan(finnhub_key: str = None):
    """Run complete market scan"""
    
    print("=" * 60)
    print("🔍 FULL MARKET SCAN")
    print("=" * 60)
    print()
    
    # 1. Run edge scanner
    print("📊 Running Edge Scanner...")
    print("-" * 40)
    
    # Combine universes
    all_symbols = list(set(UNIVERSE + list(EDGE_UNIVERSE)))
    edge_results = scan_for_edge(all_symbols[:30], finnhub_key)  # Limit for speed
    
    print()
    
    # 2. Run combined technical signals
    print("📈 Running Technical Scanner...")
    print("-" * 40)
    combined_results = scan_universe(all_symbols[:30])
    
    print()
    
    # 3. Merge results
    print("🔗 Merging signals...")
    merged_signals = merge_signals(combined_results, edge_results)
    
    # Build final report
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_context": combined_results.get("market_context", {}),
        "sector_rotation": edge_results.get("sector_rotation", {}),
        "wsb_trending": edge_results.get("wsb_trending", [])[:10],
        "signals": merged_signals,
        "top_buys": [s for s in merged_signals if s["action"] in ["BUY", "STRONG_BUY"]][:10],
        "top_sells": [s for s in merged_signals if s["action"] in ["SELL", "STRONG_SELL"]][:10],
        "edge_opportunities": edge_results.get("edge_opportunities", [])[:10],
        "squeeze_setups": edge_results.get("squeeze_setups", []),
        "summary": {
            "total_scanned": len(all_symbols[:30]),
            "buys": len([s for s in merged_signals if s["action"] in ["BUY", "STRONG_BUY"]]),
            "sells": len([s for s in merged_signals if s["action"] in ["SELL", "STRONG_SELL"]]),
            "with_edge": len([s for s in merged_signals if s.get("has_edge")]),
            "sector_signal": edge_results.get("sector_rotation", {}).get("rotation_signal", "NEUTRAL"),
            "fear_greed": combined_results.get("market_context", {}).get("fear_greed", 50)
        }
    }
    
    # Save
    with open(FULL_SCAN_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def print_full_report(results: dict):
    """Print comprehensive report"""
    
    print()
    print("=" * 60)
    print("📊 FULL SCAN RESULTS")
    print("=" * 60)
    
    ctx = results.get("market_context", {})
    summary = results.get("summary", {})
    
    # Market context
    fg = ctx.get("fear_greed", 50)
    fg_emoji = "😱" if fg < 25 else "😰" if fg < 45 else "😐" if fg < 55 else "😊" if fg < 75 else "🤑"
    print(f"\n{fg_emoji} Fear & Greed: {fg} ({ctx.get('fear_greed_label', 'Unknown')})")
    print(f"📉 VIX: {ctx.get('vix', 'N/A')}")
    print(f"🔄 Sector: {summary.get('sector_signal', 'NEUTRAL')}")
    
    # Summary stats
    print(f"\n📈 Scanned: {summary.get('total_scanned', 0)} | "
          f"Buys: {summary.get('buys', 0)} | "
          f"Sells: {summary.get('sells', 0)} | "
          f"With Edge: {summary.get('with_edge', 0)}")
    
    # Top buys
    top_buys = results.get("top_buys", [])
    if top_buys:
        print(f"\n✅ TOP BUYS ({len(top_buys)}):")
        for sig in top_buys[:5]:
            edge_flag = "⭐" if sig.get("has_edge") else ""
            boost = f" (+{sig.get('edge_boost', 0):.2f} edge)" if sig.get("edge_boost", 0) > 0 else ""
            print(f"\n   {edge_flag} {sig['symbol']} @ ${sig['price']:.2f}")
            print(f"   Score: {sig['signal_score']:.2f}{boost} | {sig['action']}")
            print(f"   • {', '.join(sig['reasons'][:3])}")
    
    # Squeeze setups
    squeezes = results.get("squeeze_setups", [])
    if squeezes:
        unique_squeezes = list({s["symbol"]: s for s in squeezes}.values())
        print(f"\n🚀 SQUEEZE SETUPS ({len(unique_squeezes)}):")
        for s in unique_squeezes[:5]:
            print(f"   {s['symbol']}: {s.get('short_percent', 0):.1f}% SI, {s.get('days_to_cover', 0):.1f} DTC")
    
    # WSB trending
    wsb = results.get("wsb_trending", [])
    if wsb:
        print(f"\n🦍 WSB HOT:")
        hot = [s for s in wsb if s.get("mention_change_pct", 0) > 50][:5]
        for s in hot:
            print(f"   {s['symbol']}: +{s.get('mention_change_pct', 0):.0f}% mentions")
    
    print()
    print("=" * 60)
    print(f"💾 Saved to: {FULL_SCAN_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Full Market Scanner")
    parser.add_argument("--finnhub-key", "-k", help="Finnhub API key")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    finnhub_key = args.finnhub_key or os.environ.get("FINNHUB_API_KEY")
    
    results = run_full_scan(finnhub_key)
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print_full_report(results)
