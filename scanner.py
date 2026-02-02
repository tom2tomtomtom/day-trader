#!/usr/bin/env python3
"""
Day Trading Scanner - Find potential targets each session

Scans for:
- Gap ups/downs (pre-market movers)
- High relative volume
- Technical setups (breakouts, VWAP plays)
- Sector rotation opportunities
- Momentum stocks
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import yfinance as yf

# Asset categories for market hours
CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD"}
FOREX_SYMBOLS = {"EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"}

def get_market_hours():
    """Determine which markets are currently open/tradeable"""
    now = datetime.now(timezone.utc)
    hour_utc = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    markets = {
        "crypto": True,  # Always open
        "forex": weekday < 5 or (weekday == 6 and hour_utc >= 22),  # Sun 22:00 - Fri 22:00 UTC
        "us_stocks": weekday < 5 and 14 <= hour_utc < 21,  # Mon-Fri 14:30-21:00 UTC (9:30-16:00 ET)
        "us_premarket": weekday < 5 and 9 <= hour_utc < 14,  # 4:00-9:30 ET
        "asia": weekday < 5 and (0 <= hour_utc < 7),  # Tokyo/HK roughly
        "europe": weekday < 5 and (7 <= hour_utc < 16),  # London/Frankfurt roughly
    }
    return markets

def filter_tradeable_symbols(symbols_dict, markets=None):
    """Filter symbols to only those currently tradeable"""
    if markets is None:
        markets = get_market_hours()
    
    tradeable = {}
    for symbol, name in symbols_dict.items():
        if symbol in CRYPTO_SYMBOLS:
            if markets["crypto"]:
                tradeable[symbol] = name
        elif symbol in FOREX_SYMBOLS:
            if markets["forex"]:
                tradeable[symbol] = name
        else:
            # US stocks/ETFs - include if market open or premarket
            if markets["us_stocks"] or markets["us_premarket"]:
                tradeable[symbol] = name
    
    return tradeable

BASE_DIR = Path(__file__).parent
WATCHLIST_PATH = BASE_DIR / "watchlist.json"
SCAN_LOG = BASE_DIR / "scan_log.jsonl"

# Universe to scan
SCAN_UNIVERSE = {
    # === CRYPTO (24/7 - always tradeable) ===
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
    "DOGE-USD": "Dogecoin",
    "ADA-USD": "Cardano",
    "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink",
    
    # === FOREX (24/5 - Sun evening to Fri evening) ===
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
    
    # === US ETFs (Mon-Fri 9:30-16:00 ET) ===
    # Major ETFs (liquid, good for day trading)
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    
    # Sector ETFs
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLC": "Communications",
    "XLY": "Consumer Disc",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    
    # Leveraged (bigger moves)
    "TQQQ": "3x Nasdaq",
    "SQQQ": "-3x Nasdaq",
    "SPXL": "3x S&P",
    "SPXS": "-3x S&P",
    "SOXL": "3x Semis",
    "SOXS": "-3x Semis",
    "LABU": "3x Biotech",
    "LABD": "-3x Biotech",
    
    # Volatility
    "UVXY": "1.5x VIX",
    "SVXY": "-0.5x VIX",
    
    # International (when those markets active)
    "EWJ": "Japan",
    "FXI": "China",
    "EWZ": "Brazil",
    "EWG": "Germany",
    "EWU": "UK",
    "EWA": "Australia",
    "EWY": "Korea",
    "EEM": "Emerging Mkts",
    
    # Crypto proxies
    "BITO": "Bitcoin ETF",
    "MSTR": "MicroStrategy",
    "COIN": "Coinbase",
    
    # Popular day trading stocks (high volume, good moves)
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "BA": "Boeing",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
}

def get_ticker_data(symbol, period="5d", interval="1d"):
    """Fetch data for a single ticker"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            return None
        
        # Get current/last price
        current = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
        
        # Calculate metrics
        change_pct = ((current - prev_close) / prev_close) * 100
        
        # Volume analysis
        avg_volume = hist['Volume'].iloc[:-1].mean() if len(hist) > 1 else hist['Volume'].iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        rel_volume = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Range analysis
        high_5d = hist['High'].max()
        low_5d = hist['Low'].min()
        range_position = (current - low_5d) / (high_5d - low_5d) if high_5d != low_5d else 0.5
        
        # Volatility (ATR proxy)
        if len(hist) > 1:
            daily_ranges = hist['High'] - hist['Low']
            atr = daily_ranges.mean()
            atr_pct = (atr / current) * 100
        else:
            atr_pct = 2.0  # Default
        
        return {
            "symbol": symbol,
            "price": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(current_volume),
            "avg_volume": int(avg_volume),
            "rel_volume": round(rel_volume, 2),
            "high_5d": round(high_5d, 2),
            "low_5d": round(low_5d, 2),
            "range_position": round(range_position, 2),  # 0=at lows, 1=at highs
            "atr_pct": round(atr_pct, 2),  # Average daily range %
        }
    except Exception as e:
        return None

def scan_for_gaps(data_list, min_gap=1.5):
    """Find stocks gapping up or down significantly"""
    gaps = []
    for d in data_list:
        if d and abs(d["change_pct"]) >= min_gap:
            direction = "GAP_UP" if d["change_pct"] > 0 else "GAP_DOWN"
            gaps.append({
                **d,
                "setup": direction,
                "signal_strength": min(abs(d["change_pct"]) / 3, 1.0),  # Normalize to 0-1
                "trade_idea": f"{'Fade' if abs(d['change_pct']) > 4 else 'Momentum'} the gap"
            })
    return sorted(gaps, key=lambda x: abs(x["change_pct"]), reverse=True)

def scan_for_volume_surge(data_list, min_rel_vol=1.5):
    """Find stocks with unusual volume"""
    surges = []
    for d in data_list:
        if d and d["rel_volume"] >= min_rel_vol:
            surges.append({
                **d,
                "setup": "VOLUME_SURGE",
                "signal_strength": min(d["rel_volume"] / 3, 1.0),
                "trade_idea": "Follow the volume - big players moving"
            })
    return sorted(surges, key=lambda x: x["rel_volume"], reverse=True)

def scan_for_breakouts(data_list, threshold=0.98):
    """Find stocks near 5-day highs (potential breakouts)"""
    breakouts = []
    for d in data_list:
        if d and d["range_position"] >= threshold:
            breakouts.append({
                **d,
                "setup": "BREAKOUT",
                "signal_strength": d["range_position"],
                "trade_idea": "Breakout play - buy break of high, stop below"
            })
    return sorted(breakouts, key=lambda x: x["range_position"], reverse=True)

def scan_for_breakdowns(data_list, threshold=0.02):
    """Find stocks near 5-day lows (potential breakdowns or bounce plays)"""
    breakdowns = []
    for d in data_list:
        if d and d["range_position"] <= threshold:
            breakdowns.append({
                **d,
                "setup": "BREAKDOWN",
                "signal_strength": 1 - d["range_position"],
                "trade_idea": "Breakdown/bounce - short break of low OR buy support bounce"
            })
    return sorted(breakdowns, key=lambda x: x["range_position"])

def scan_for_mean_reversion(data_list, overbought=0.9, oversold=0.1):
    """Find extended stocks that might revert"""
    reversions = []
    for d in data_list:
        if d:
            if d["range_position"] >= overbought and d["change_pct"] > 2:
                reversions.append({
                    **d,
                    "setup": "OVERBOUGHT",
                    "signal_strength": d["range_position"],
                    "trade_idea": "Extended - watch for reversal short"
                })
            elif d["range_position"] <= oversold and d["change_pct"] < -2:
                reversions.append({
                    **d,
                    "setup": "OVERSOLD",
                    "signal_strength": 1 - d["range_position"],
                    "trade_idea": "Oversold bounce candidate"
                })
    return reversions

def scan_for_volatility_plays(data_list, min_atr=2.5):
    """Find high volatility stocks (good for day trading)"""
    vol_plays = []
    for d in data_list:
        if d and d["atr_pct"] >= min_atr:
            vol_plays.append({
                **d,
                "setup": "HIGH_VOLATILITY",
                "signal_strength": min(d["atr_pct"] / 5, 1.0),
                "trade_idea": f"Volatile ({d['atr_pct']:.1f}% daily range) - good for scalping"
            })
    return sorted(vol_plays, key=lambda x: x["atr_pct"], reverse=True)

def scan_for_momentum(data_list, min_move=2.0, min_vol=1.3):
    """Find momentum plays - strong move + volume confirmation"""
    momentum = []
    for d in data_list:
        if d and abs(d["change_pct"]) >= min_move and d["rel_volume"] >= min_vol:
            direction = "MOMENTUM_LONG" if d["change_pct"] > 0 else "MOMENTUM_SHORT"
            momentum.append({
                **d,
                "setup": direction,
                "signal_strength": min(abs(d["change_pct"]) / 5, 1.0) * min(d["rel_volume"] / 2, 1.0),
                "trade_idea": f"{'Long' if d['change_pct'] > 0 else 'Short'} momentum - {d['change_pct']:+.1f}% on {d['rel_volume']:.1f}x volume"
            })
    return sorted(momentum, key=lambda x: x["signal_strength"], reverse=True)

def scan_for_relative_strength(data_list, spy_change=0):
    """Find stocks significantly outperforming or underperforming SPY"""
    rs_plays = []
    for d in data_list:
        if d:
            relative = d["change_pct"] - spy_change
            if abs(relative) >= 3:  # At least 3% relative move
                setup = "RELATIVE_STRENGTH" if relative > 0 else "RELATIVE_WEAKNESS"
                rs_plays.append({
                    **d,
                    "setup": setup,
                    "relative_pct": round(relative, 2),
                    "signal_strength": min(abs(relative) / 5, 1.0),
                    "trade_idea": f"{'Outperforming' if relative > 0 else 'Underperforming'} SPY by {relative:+.1f}%"
                })
    return sorted(rs_plays, key=lambda x: abs(x.get("relative_pct", 0)), reverse=True)

def scan_for_reversal_candidates(data_list):
    """Find potential reversal setups - extended + volume exhaustion"""
    reversals = []
    for d in data_list:
        if d:
            # Extended up with declining volume = potential top
            if d["range_position"] > 0.9 and d["rel_volume"] < 0.8 and d["change_pct"] > 1:
                reversals.append({
                    **d,
                    "setup": "REVERSAL_SHORT",
                    "signal_strength": d["range_position"] * (1 - d["rel_volume"]),
                    "trade_idea": "Near highs on weak volume - potential reversal short"
                })
            # Extended down with declining volume = potential bottom
            elif d["range_position"] < 0.1 and d["rel_volume"] < 0.8 and d["change_pct"] < -1:
                reversals.append({
                    **d,
                    "setup": "REVERSAL_LONG",
                    "signal_strength": (1 - d["range_position"]) * (1 - d["rel_volume"]),
                    "trade_idea": "Near lows on weak volume - potential reversal long"
                })
    return sorted(reversals, key=lambda x: x["signal_strength"], reverse=True)

def run_full_scan(universe=None, global_mode=False, tradeable_only=True):
    """Run all scans and compile watchlist
    
    Args:
        universe: Dict of symbols to scan (default: SCAN_UNIVERSE)
        global_mode: If True, includes crypto/forex even outside normal hours
        tradeable_only: If True, only scan assets whose markets are open
    """
    if universe is None:
        universe = SCAN_UNIVERSE
    
    # Filter to tradeable symbols if requested
    markets = get_market_hours()
    if tradeable_only and not global_mode:
        universe = filter_tradeable_symbols(universe, markets)
    
    # Show market status
    print(f"Market Status: Crypto={'✓' if markets['crypto'] else '✗'} | "
          f"Forex={'✓' if markets['forex'] else '✗'} | "
          f"US={'✓' if markets['us_stocks'] else ('PRE' if markets['us_premarket'] else '✗')}")
    print(f"Scanning {len(universe)} symbols...")
    
    # Fetch all data
    all_data = []
    for symbol, name in universe.items():
        print(f"  {symbol}...", end=" ", flush=True)
        data = get_ticker_data(symbol)
        if data:
            data["name"] = name
            all_data.append(data)
            print(f"${data['price']} ({data['change_pct']:+.1f}%)")
        else:
            print("SKIP")
    
    print(f"\nLoaded {len(all_data)} symbols")
    
    # Separate crypto and forex data
    crypto_data = [d for d in all_data if d and d["symbol"] in CRYPTO_SYMBOLS]
    forex_data = [d for d in all_data if d and d["symbol"] in FOREX_SYMBOLS]
    stocks_data = [d for d in all_data if d and d["symbol"] not in CRYPTO_SYMBOLS and d["symbol"] not in FOREX_SYMBOLS]
    
    # Run scans
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scanned": len(all_data),
        "markets": markets,
        "setups": {}
    }
    
    # === CRYPTO SECTION (always included) ===
    if crypto_data:
        # Sort by absolute move
        crypto_sorted = sorted(crypto_data, key=lambda x: abs(x["change_pct"]), reverse=True)
        results["setups"]["crypto"] = [{
            **c,
            "setup": "CRYPTO_" + ("DIP" if c["change_pct"] < -2 else "PUMP" if c["change_pct"] > 2 else "RANGE"),
            "signal_strength": min(abs(c["change_pct"]) / 5, 1.0),
            "trade_idea": f"{'Buy the dip' if c['change_pct'] < -3 else 'Momentum long' if c['change_pct'] > 2 else 'Range trade'} - 24/7 market"
        } for c in crypto_sorted[:5]]
    
    # === FOREX SECTION (always included when open) ===
    if forex_data:
        forex_sorted = sorted(forex_data, key=lambda x: abs(x["change_pct"]), reverse=True)
        results["setups"]["forex"] = [{
            **f,
            "setup": "FX_MOVE",
            "signal_strength": min(abs(f["change_pct"]) / 2, 1.0),
            "trade_idea": f"{'Sell' if f['change_pct'] > 0.5 else 'Buy' if f['change_pct'] < -0.5 else 'Range'} - watch for continuation"
        } for f in forex_sorted[:3]]
    
    # === STOCKS/ETFs ===
    # Gap plays
    gaps = scan_for_gaps(stocks_data)
    if gaps:
        results["setups"]["gaps"] = gaps[:5]  # Top 5
    
    # Volume surges
    volume = scan_for_volume_surge(stocks_data)
    if volume:
        results["setups"]["volume_surges"] = volume[:5]
    
    # Breakouts
    breakouts = scan_for_breakouts(stocks_data)
    if breakouts:
        results["setups"]["breakouts"] = breakouts[:5]
    
    # Breakdowns
    breakdowns = scan_for_breakdowns(stocks_data)
    if breakdowns:
        results["setups"]["breakdowns"] = breakdowns[:5]
    
    # Mean reversion
    reversions = scan_for_mean_reversion(stocks_data)
    if reversions:
        results["setups"]["mean_reversion"] = reversions[:5]
    
    # Volatility plays
    vol_plays = scan_for_volatility_plays(stocks_data)
    if vol_plays:
        results["setups"]["high_volatility"] = vol_plays[:5]
    
    # Momentum plays (strong move + volume)
    momentum = scan_for_momentum(stocks_data)
    if momentum:
        results["setups"]["momentum"] = momentum[:5]
    
    # Relative strength vs SPY
    spy_data = next((d for d in stocks_data if d and d["symbol"] == "SPY"), None)
    spy_change = spy_data["change_pct"] if spy_data else 0
    rs_plays = scan_for_relative_strength(stocks_data, spy_change)
    if rs_plays:
        results["setups"]["relative_strength"] = rs_plays[:5]
    
    # Reversal candidates
    reversals = scan_for_reversal_candidates(all_data)
    if reversals:
        results["setups"]["reversals"] = reversals[:5]
    
    # Build unified watchlist (deduplicated, top opportunities)
    watchlist = {}
    for setup_type, setups in results["setups"].items():
        for s in setups[:3]:  # Top 3 from each category
            sym = s["symbol"]
            if sym not in watchlist:
                watchlist[sym] = {
                    "symbol": sym,
                    "name": s.get("name", sym),
                    "price": s["price"],
                    "change_pct": s["change_pct"],
                    "asset_class": "crypto" if sym in CRYPTO_SYMBOLS else "forex" if sym in FOREX_SYMBOLS else "stock",
                    "setups": [],
                    "trade_ideas": []
                }
            watchlist[sym]["setups"].append(s["setup"])
            watchlist[sym]["trade_ideas"].append(s["trade_idea"])
    
    # Sort: crypto first (24/7), then by number of setups
    def watchlist_sort_key(x):
        # Priority: crypto=2, forex=1, stocks=0, then by setup count
        asset_priority = {"crypto": 2, "forex": 1, "stock": 0}.get(x.get("asset_class", "stock"), 0)
        return (asset_priority, len(x["setups"]), abs(x["change_pct"]))
    
    watchlist_sorted = sorted(watchlist.values(), key=watchlist_sort_key, reverse=True)
    
    results["watchlist"] = watchlist_sorted[:12]  # Top 12 (more room for crypto)
    
    # Save results
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    # Log scan
    with open(SCAN_LOG, "a") as f:
        log_entry = {
            "timestamp": results["timestamp"],
            "scanned": results["scanned"],
            "watchlist_size": len(results["watchlist"]),
            "top_picks": [w["symbol"] for w in results["watchlist"][:5]]
        }
        f.write(json.dumps(log_entry) + "\n")
    
    return results

def print_scan_results(results):
    """Pretty print scan results"""
    print("\n" + "="*60)
    print("🔍 DAY TRADING SCANNER RESULTS")
    print(f"   {results['timestamp']}")
    print("="*60)
    
    # Setup summaries
    for setup_type, setups in results.get("setups", {}).items():
        print(f"\n📊 {setup_type.upper().replace('_', ' ')}:")
        for s in setups[:3]:
            print(f"   {s['symbol']:6} ${s['price']:>8.2f} ({s['change_pct']:+5.1f}%) - {s.get('trade_idea', '')[:40]}")
    
    # Watchlist
    print("\n" + "="*60)
    print("⭐ TODAY'S WATCHLIST")
    print("="*60)
    
    for i, w in enumerate(results.get("watchlist", []), 1):
        setups_str = ", ".join(w["setups"][:2])
        print(f"\n{i}. {w['symbol']} - {w['name']}")
        print(f"   Price: ${w['price']:.2f} ({w['change_pct']:+.1f}%)")
        print(f"   Setups: {setups_str}")
        print(f"   Ideas: {w['trade_ideas'][0][:50]}")
    
    print()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Day Trading Scanner")
    parser.add_argument("command", choices=["scan", "watchlist", "markets"], nargs="?", default="scan")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--global", "-g", dest="global_mode", action="store_true", 
                        help="Scan all assets regardless of market hours (crypto/forex 24/7)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Scan entire universe, not just tradeable assets")
    
    args = parser.parse_args()
    
    if args.command == "markets":
        markets = get_market_hours()
        if args.json:
            print(json.dumps(markets, indent=2))
        else:
            print("\n📊 MARKET STATUS")
            print("="*40)
            print(f"  Crypto:     {'✅ OPEN (24/7)' if markets['crypto'] else '❌ Closed'}")
            print(f"  Forex:      {'✅ OPEN' if markets['forex'] else '❌ Closed'}")
            print(f"  US Stocks:  {'✅ OPEN' if markets['us_stocks'] else ('🌅 PRE-MARKET' if markets['us_premarket'] else '❌ Closed')}")
            print(f"  Asia:       {'✅ OPEN' if markets['asia'] else '❌ Closed'}")
            print(f"  Europe:     {'✅ OPEN' if markets['europe'] else '❌ Closed'}")
            print()
    
    elif args.command == "scan":
        results = run_full_scan(global_mode=args.global_mode, tradeable_only=not args.all)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_scan_results(results)
    
    elif args.command == "watchlist":
        if WATCHLIST_PATH.exists():
            with open(WATCHLIST_PATH) as f:
                results = json.load(f)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print_scan_results(results)
        else:
            print("No watchlist. Run 'scan' first.")

if __name__ == "__main__":
    main()
