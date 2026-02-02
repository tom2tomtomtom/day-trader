#!/usr/bin/env python3
"""
Edge Case Scanner v1.0 🎯

Finds high-probability edge opportunities by scanning:
1. Insider buying clusters (SEC Form 4 - FREE)
2. Congressional trades (Capitol Trades - FREE)  
3. Short squeeze setups (High SI + catalyst)
4. WSB/Reddit momentum (ApeWisdom - FREE)
5. Unusual options activity
6. Sector rotation signals
7. Earnings momentum (beat streaks)

The edge: Most retail watches price. Smart money watches flows.
"""

import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).parent
EDGE_SIGNALS_PATH = BASE_DIR / "edge_signals.json"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Rate limiting
REQUEST_DELAY = 0.5

# ============================================================
# 1. INSIDER TRADING SCANNER (SEC Form 4 via Finnhub/SEC EDGAR)
# ============================================================

def get_insider_sentiment(symbol: str, finnhub_key: str = None) -> dict:
    """
    Check insider buying/selling patterns
    Cluster buys (multiple insiders) = very bullish
    """
    result = {
        "symbol": symbol,
        "insider_signal": "NEUTRAL",
        "net_buys_90d": 0,
        "cluster_buy": False,
        "notable_trades": []
    }
    
    # Try Finnhub if we have a key
    if finnhub_key:
        try:
            url = f"https://finnhub.io/api/v1/stock/insider-transactions"
            params = {"symbol": symbol, "token": finnhub_key}
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if data.get("data"):
                buys = 0
                sells = 0
                buy_value = 0
                sell_value = 0
                recent_buyers = set()
                
                cutoff = datetime.now() - timedelta(days=90)
                
                for tx in data["data"][:50]:  # Last 50 transactions
                    tx_date = datetime.strptime(tx.get("transactionDate", "2000-01-01"), "%Y-%m-%d")
                    if tx_date < cutoff:
                        continue
                    
                    shares = tx.get("share", 0) or 0
                    price = tx.get("transactionPrice", 0) or 0
                    value = abs(shares * price)
                    tx_type = tx.get("transactionCode", "")
                    
                    # P = Purchase, S = Sale, A = Award/Grant
                    if tx_type == "P":
                        buys += 1
                        buy_value += value
                        recent_buyers.add(tx.get("name", "Unknown"))
                        if value > 100000:
                            result["notable_trades"].append({
                                "type": "BUY",
                                "name": tx.get("name"),
                                "value": value,
                                "date": tx.get("transactionDate")
                            })
                    elif tx_type == "S":
                        sells += 1
                        sell_value += value
                
                result["net_buys_90d"] = buys - sells
                result["buy_value"] = buy_value
                result["sell_value"] = sell_value
                
                # Cluster buy = 3+ different insiders buying
                if len(recent_buyers) >= 3:
                    result["cluster_buy"] = True
                    result["insider_signal"] = "STRONG_BUY"
                elif buys > sells * 2 and buy_value > 500000:
                    result["insider_signal"] = "BUY"
                elif sells > buys * 2 and sell_value > 1000000:
                    result["insider_signal"] = "SELL"
                    
        except Exception as e:
            result["error"] = str(e)
    
    return result


# ============================================================
# 2. CONGRESSIONAL TRADING (Quiver Quant API - FREE tier)
# ============================================================

def get_congressional_trades(days: int = 30) -> list:
    """
    Fetch recent congressional stock trades
    Politicians have 45 days to report - but patterns emerge
    
    Uses Quiver Quant public data
    """
    trades = []
    
    try:
        # Quiver Quant has a free tier for congressional trades
        # Alternative: Capitol Trades API
        url = "https://api.quiverquant.com/beta/live/congresstrading"
        headers = {"accept": "application/json"}
        
        # Note: Quiver requires API key for full access
        # Fall back to scraping Capitol Trades if needed
        resp = requests.get(
            "https://www.capitoltrades.com/trades?page=1&pageSize=50",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        
        # For now, return structure for manual enrichment
        # In production, parse the response or use paid API
        
    except Exception as e:
        pass
    
    return trades


def get_congress_stock_mentions() -> dict:
    """
    Track which stocks congress is trading most
    High activity = they know something
    """
    # This would aggregate congressional trade data
    # Return top symbols being bought/sold
    return {
        "top_buys": [],
        "top_sells": [],
        "bipartisan_buys": [],  # Both parties buying = strong signal
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# 3. SHORT SQUEEZE SCANNER
# ============================================================

def get_short_interest(symbol: str) -> dict:
    """
    Get short interest data for squeeze potential
    
    High SI (>20%) + positive catalyst = squeeze setup
    Days to cover > 5 = harder to unwind
    """
    result = {
        "symbol": symbol,
        "short_percent": None,
        "days_to_cover": None,
        "squeeze_potential": "LOW"
    }
    
    try:
        # Yahoo Finance has some short data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        shares_short = info.get("sharesShort", 0)
        shares_outstanding = info.get("sharesOutstanding", 1)
        avg_volume = info.get("averageVolume", 1)
        float_shares = info.get("floatShares", shares_outstanding)
        
        if shares_short and float_shares:
            short_pct = (shares_short / float_shares) * 100
            result["short_percent"] = round(short_pct, 2)
            
            if avg_volume > 0:
                days_to_cover = shares_short / avg_volume
                result["days_to_cover"] = round(days_to_cover, 1)
        
        # Evaluate squeeze potential
        si = result["short_percent"] or 0
        dtc = result["days_to_cover"] or 0
        
        if si > 30 and dtc > 5:
            result["squeeze_potential"] = "HIGH"
        elif si > 20 and dtc > 3:
            result["squeeze_potential"] = "MEDIUM"
        elif si > 15:
            result["squeeze_potential"] = "LOW-MEDIUM"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def scan_squeeze_candidates(symbols: list) -> list:
    """Find stocks with squeeze potential"""
    candidates = []
    
    for symbol in symbols:
        data = get_short_interest(symbol)
        if data.get("squeeze_potential") in ["HIGH", "MEDIUM"]:
            candidates.append(data)
    
    return sorted(candidates, key=lambda x: x.get("short_percent", 0), reverse=True)


# ============================================================
# 4. WSB / REDDIT SENTIMENT (ApeWisdom - FREE)
# ============================================================

def get_wsb_sentiment() -> dict:
    """
    Get WallStreetBets sentiment from ApeWisdom
    
    Rising mentions + positive sentiment = retail momentum
    Falling mentions after spike = hype exhaustion
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trending": [],
        "momentum": [],
        "exhaustion": []
    }
    
    try:
        # ApeWisdom API - free, no key required
        resp = requests.get(
            "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
            timeout=10
        )
        data = resp.json()
        
        if data.get("results"):
            for stock in data["results"][:20]:
                item = {
                    "symbol": stock.get("ticker"),
                    "name": stock.get("name"),
                    "mentions_24h": stock.get("mentions"),
                    "mentions_24h_ago": stock.get("mentions_24h_ago"),
                    "rank": stock.get("rank"),
                    "upvotes": stock.get("upvotes")
                }
                
                # Calculate momentum
                prev = item["mentions_24h_ago"] or 1
                curr = item["mentions_24h"] or 0
                change = ((curr - prev) / prev) * 100 if prev > 0 else 0
                item["mention_change_pct"] = round(change, 1)
                
                result["trending"].append(item)
                
                # Categorize
                if change > 50:
                    result["momentum"].append(item)
                elif change < -30 and curr > 50:
                    result["exhaustion"].append(item)
        
        # Sort by mentions
        result["trending"] = sorted(
            result["trending"], 
            key=lambda x: x.get("mentions_24h", 0), 
            reverse=True
        )
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ============================================================
# 5. OPTIONS FLOW SCANNER
# ============================================================

def get_options_sentiment(symbol: str, finnhub_key: str = None) -> dict:
    """
    Analyze options activity for sentiment
    
    - Put/Call ratio < 0.7 = bullish
    - Put/Call ratio > 1.3 = bearish (or hedging)
    - Unusual volume = someone knows something
    """
    result = {
        "symbol": symbol,
        "put_call_ratio": None,
        "options_signal": "NEUTRAL",
        "unusual_activity": False
    }
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get options chain
        if ticker.options:
            nearest_expiry = ticker.options[0]
            chain = ticker.option_chain(nearest_expiry)
            
            call_volume = chain.calls["volume"].sum()
            put_volume = chain.puts["volume"].sum()
            call_oi = chain.calls["openInterest"].sum()
            put_oi = chain.puts["openInterest"].sum()
            
            if call_volume > 0:
                pc_ratio = put_volume / call_volume
                result["put_call_ratio"] = round(pc_ratio, 2)
                
                if pc_ratio < 0.5:
                    result["options_signal"] = "VERY_BULLISH"
                elif pc_ratio < 0.7:
                    result["options_signal"] = "BULLISH"
                elif pc_ratio > 1.5:
                    result["options_signal"] = "VERY_BEARISH"
                elif pc_ratio > 1.2:
                    result["options_signal"] = "BEARISH"
            
            # Check for unusual activity
            avg_call_vol = chain.calls["volume"].mean()
            avg_put_vol = chain.puts["volume"].mean()
            
            # Any strike with 5x avg volume = unusual
            unusual_calls = (chain.calls["volume"] > avg_call_vol * 5).any()
            unusual_puts = (chain.puts["volume"] > avg_put_vol * 5).any()
            
            result["unusual_activity"] = bool(unusual_calls or unusual_puts)
            
            result["call_volume"] = int(call_volume)
            result["put_volume"] = int(put_volume)
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ============================================================
# 6. SECTOR ROTATION DETECTOR
# ============================================================

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials", 
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communications"
}

def get_sector_rotation() -> dict:
    """
    Detect sector rotation patterns
    
    Money flowing into defensive (XLU, XLP) = risk-off
    Money flowing into growth (XLK, XLY) = risk-on
    """
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rotation_signal": "NEUTRAL",
        "sectors": [],
        "leaders": [],
        "laggards": []
    }
    
    performances = []
    
    for symbol, name in SECTOR_ETFS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if len(hist) >= 5:
                pct_1d = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-2]) - 1) * 100
                pct_5d = ((hist["Close"].iloc[-1] / hist["Close"].iloc[-5]) - 1) * 100
                pct_1mo = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100
                
                performances.append({
                    "symbol": symbol,
                    "name": name,
                    "pct_1d": round(pct_1d, 2),
                    "pct_5d": round(pct_5d, 2),
                    "pct_1mo": round(pct_1mo, 2),
                    "momentum_score": round(pct_1d + pct_5d * 0.5 + pct_1mo * 0.3, 2)
                })
        except:
            continue
    
    # Sort by momentum
    performances = sorted(performances, key=lambda x: x["momentum_score"], reverse=True)
    result["sectors"] = performances
    
    if performances:
        result["leaders"] = performances[:3]
        result["laggards"] = performances[-3:]
        
        # Detect risk-on vs risk-off
        leader_symbols = [s["symbol"] for s in result["leaders"]]
        defensive = {"XLU", "XLP", "XLV"}
        growth = {"XLK", "XLY", "XLC"}
        
        defensive_leading = len(set(leader_symbols) & defensive)
        growth_leading = len(set(leader_symbols) & growth)
        
        if growth_leading >= 2:
            result["rotation_signal"] = "RISK_ON"
        elif defensive_leading >= 2:
            result["rotation_signal"] = "RISK_OFF"
    
    return result


# ============================================================
# 7. EARNINGS MOMENTUM SCANNER
# ============================================================

def get_earnings_momentum(symbol: str) -> dict:
    """
    Check earnings beat streak
    
    Companies that beat consistently tend to continue
    3+ beat streak = momentum
    """
    result = {
        "symbol": symbol,
        "beat_streak": 0,
        "last_surprise_pct": None,
        "earnings_momentum": "NEUTRAL"
    }
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get earnings history
        earnings = ticker.earnings_history
        
        if earnings is not None and len(earnings) > 0:
            streak = 0
            surprises = []
            
            for _, row in earnings.head(8).iterrows():
                surprise = row.get("surprisePercent") or row.get("epsActual", 0) - row.get("epsEstimate", 0)
                
                if isinstance(surprise, (int, float)):
                    surprises.append(surprise)
                    if surprise > 0:
                        streak += 1
                    else:
                        break
            
            result["beat_streak"] = streak
            if surprises:
                result["last_surprise_pct"] = round(surprises[0], 2) if surprises else None
            
            if streak >= 4:
                result["earnings_momentum"] = "STRONG"
            elif streak >= 2:
                result["earnings_momentum"] = "POSITIVE"
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ============================================================
# 8. COMBINED EDGE SCANNER
# ============================================================

def scan_for_edge(symbols: list, finnhub_key: str = None) -> dict:
    """
    Run all edge scanners and combine results
    
    Returns stocks with multiple edge signals converging
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "edge_opportunities": [],
        "wsb_momentum": [],
        "squeeze_setups": [],
        "insider_buys": [],
        "sector_rotation": None,
        "summary": {}
    }
    
    print("🔍 Scanning for edge opportunities...\n")
    
    # 1. WSB Sentiment
    print("  📱 Checking WSB sentiment...")
    wsb = get_wsb_sentiment()
    results["wsb_momentum"] = wsb.get("momentum", [])[:5]
    results["wsb_trending"] = wsb.get("trending", [])[:10]
    
    # 2. Sector Rotation
    print("  🔄 Analyzing sector rotation...")
    rotation = get_sector_rotation()
    results["sector_rotation"] = rotation
    
    # 3. Per-symbol analysis
    print(f"  📊 Analyzing {len(symbols)} symbols...")
    
    for symbol in symbols:
        print(f"    {symbol}...", end=" ", flush=True)
        
        edge_score = 0
        edge_reasons = []
        
        # Short interest
        si = get_short_interest(symbol)
        if si.get("squeeze_potential") in ["HIGH", "MEDIUM"]:
            edge_score += 2 if si["squeeze_potential"] == "HIGH" else 1
            edge_reasons.append(f"Short squeeze potential ({si['short_percent']}% SI)")
            results["squeeze_setups"].append(si)
        
        # Insider trading
        if finnhub_key:
            insider = get_insider_sentiment(symbol, finnhub_key)
            if insider.get("insider_signal") in ["BUY", "STRONG_BUY"]:
                edge_score += 2 if insider["cluster_buy"] else 1
                edge_reasons.append(f"Insider buying ({insider['net_buys_90d']} net)")
                results["insider_buys"].append(insider)
        
        # Options sentiment
        options = get_options_sentiment(symbol)
        if options.get("unusual_activity"):
            edge_score += 1
            edge_reasons.append("Unusual options activity")
        if options.get("options_signal") in ["VERY_BULLISH", "BULLISH"]:
            edge_score += 1
            edge_reasons.append(f"Options bullish (P/C: {options['put_call_ratio']})")
        
        # Earnings momentum
        earnings = get_earnings_momentum(symbol)
        if earnings.get("earnings_momentum") == "STRONG":
            edge_score += 1
            edge_reasons.append(f"Earnings beat streak ({earnings['beat_streak']})")
        
        # Check if in WSB momentum
        wsb_symbols = [w["symbol"] for w in results.get("wsb_momentum", [])]
        if symbol in wsb_symbols:
            edge_score += 1
            edge_reasons.append("WSB momentum rising")
        
        if edge_score >= 2:
            results["edge_opportunities"].append({
                "symbol": symbol,
                "edge_score": edge_score,
                "reasons": edge_reasons,
                "short_interest": si,
                "options": options,
                "earnings": earnings
            })
            print(f"✅ Edge score: {edge_score}")
        else:
            print("○")
    
    # Sort by edge score
    results["edge_opportunities"] = sorted(
        results["edge_opportunities"],
        key=lambda x: x["edge_score"],
        reverse=True
    )
    
    # Summary
    results["summary"] = {
        "total_scanned": len(symbols),
        "edge_opportunities": len(results["edge_opportunities"]),
        "squeeze_setups": len(results["squeeze_setups"]),
        "insider_buys": len(results["insider_buys"]),
        "wsb_momentum": len(results["wsb_momentum"]),
        "sector_signal": rotation.get("rotation_signal", "NEUTRAL")
    }
    
    # Save results
    with open(EDGE_SIGNALS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def print_edge_report(results: dict):
    """Print edge scanner report"""
    print("\n" + "="*60)
    print("🎯 EDGE SCANNER REPORT")
    print("="*60)
    
    # Sector rotation
    rot = results.get("sector_rotation", {})
    signal = rot.get("rotation_signal", "NEUTRAL")
    signal_emoji = "🟢" if signal == "RISK_ON" else "🔴" if signal == "RISK_OFF" else "⚪"
    print(f"\n{signal_emoji} Sector Rotation: {signal}")
    
    if rot.get("leaders"):
        print("   Leaders:", ", ".join([f"{s['symbol']} ({s['pct_5d']:+.1f}%)" for s in rot["leaders"]]))
    if rot.get("laggards"):
        print("   Laggards:", ", ".join([f"{s['symbol']} ({s['pct_5d']:+.1f}%)" for s in rot["laggards"]]))
    
    # WSB momentum
    if results.get("wsb_momentum"):
        print(f"\n🦍 WSB Momentum ({len(results['wsb_momentum'])} stocks):")
        for stock in results["wsb_momentum"][:5]:
            print(f"   {stock['symbol']}: {stock['mentions_24h']} mentions ({stock['mention_change_pct']:+.0f}%)")
    
    # Edge opportunities
    if results.get("edge_opportunities"):
        print(f"\n⭐ TOP EDGE OPPORTUNITIES ({len(results['edge_opportunities'])}):")
        for opp in results["edge_opportunities"][:10]:
            print(f"\n   {opp['symbol']} (Edge Score: {opp['edge_score']})")
            for reason in opp["reasons"]:
                print(f"      • {reason}")
    else:
        print("\n⚪ No strong edge opportunities found")
    
    # Squeeze setups
    if results.get("squeeze_setups"):
        print(f"\n🚀 Squeeze Setups ({len(results['squeeze_setups'])}):")
        for setup in results["squeeze_setups"][:5]:
            print(f"   {setup['symbol']}: {setup['short_percent']}% SI, {setup['days_to_cover']} DTC")
    
    print("\n" + "="*60)


# Default scan universe (add your watchlist)
EDGE_UNIVERSE = [
    # Meme favorites
    "GME", "AMC", "BBBY", "PLTR", "SOFI", "RIVN", "LCID",
    # Tech growth
    "NVDA", "AMD", "TSLA", "SNOW", "CRWD", "NET", "DDOG",
    # Beaten down
    "PYPL", "SQ", "SHOP", "ROKU", "ZM", "DOCU",
    # Biotech (volatile)
    "MRNA", "BNTX", "NVAX",
    # SPACs that de-SPAC'd
    "DWAC", "GRAB", "LCID",
    # Crypto proxies
    "COIN", "MSTR", "RIOT", "MARA",
]


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Edge Case Scanner")
    parser.add_argument("--finnhub-key", "-k", help="Finnhub API key for insider data")
    parser.add_argument("--symbol", "-s", help="Single symbol deep dive")
    parser.add_argument("--wsb", action="store_true", help="Just show WSB data")
    parser.add_argument("--sectors", action="store_true", help="Just show sector rotation")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()
    
    # Try to get Finnhub key from env
    finnhub_key = args.finnhub_key or os.environ.get("FINNHUB_API_KEY")
    
    if args.wsb:
        data = get_wsb_sentiment()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print("\n🦍 WSB Trending:\n")
            for stock in data.get("trending", [])[:15]:
                change = stock.get("mention_change_pct", 0)
                emoji = "🔥" if change > 50 else "📈" if change > 0 else "📉"
                print(f"  {emoji} {stock['symbol']}: {stock['mentions_24h']} mentions ({change:+.0f}%)")
    
    elif args.sectors:
        data = get_sector_rotation()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"\n🔄 Sector Rotation: {data['rotation_signal']}\n")
            for sector in data.get("sectors", []):
                bar = "█" * int(abs(sector["pct_5d"]))
                color = "+" if sector["pct_5d"] > 0 else ""
                print(f"  {sector['symbol']:4} {sector['name']:25} {color}{sector['pct_5d']:5.1f}% {bar}")
    
    elif args.symbol:
        print(f"\n🔍 Deep dive: {args.symbol}\n")
        
        si = get_short_interest(args.symbol)
        print(f"Short Interest: {si.get('short_percent', 'N/A')}%")
        print(f"Days to Cover: {si.get('days_to_cover', 'N/A')}")
        print(f"Squeeze Potential: {si.get('squeeze_potential', 'N/A')}")
        
        opt = get_options_sentiment(args.symbol)
        print(f"\nPut/Call Ratio: {opt.get('put_call_ratio', 'N/A')}")
        print(f"Options Signal: {opt.get('options_signal', 'N/A')}")
        print(f"Unusual Activity: {opt.get('unusual_activity', 'N/A')}")
        
        earn = get_earnings_momentum(args.symbol)
        print(f"\nEarnings Beat Streak: {earn.get('beat_streak', 'N/A')}")
        print(f"Earnings Momentum: {earn.get('earnings_momentum', 'N/A')}")
        
        if finnhub_key:
            ins = get_insider_sentiment(args.symbol, finnhub_key)
            print(f"\nInsider Signal: {ins.get('insider_signal', 'N/A')}")
            print(f"Net Buys (90d): {ins.get('net_buys_90d', 'N/A')}")
            print(f"Cluster Buy: {ins.get('cluster_buy', 'N/A')}")
    
    else:
        results = scan_for_edge(EDGE_UNIVERSE, finnhub_key)
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print_edge_report(results)
