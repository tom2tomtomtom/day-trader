#!/usr/bin/env python3
"""
Day Trading Simulator - Paper trade intraday

Features:
- Track watchlist throughout the day
- Entry/exit signals based on price action
- Paper trade execution with stops and targets
- End of day P&L calculation
- Learning from results
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yfinance as yf

BASE_DIR = Path(__file__).parent
WATCHLIST_PATH = BASE_DIR / "watchlist.json"
POSITIONS_PATH = BASE_DIR / "day_positions.json"
DAY_LOG_PATH = BASE_DIR / "day_trades.jsonl"
DAILY_RESULTS_PATH = BASE_DIR / "daily_results.json"

# Day trading parameters
STARTING_CAPITAL = 100000  # $100k paper account
MAX_POSITION_SIZE = 0.10   # Max 10% per trade
MAX_POSITIONS = 5          # Max 5 concurrent positions
DEFAULT_STOP_PCT = 0.02    # 2% stop loss
DEFAULT_TARGET_PCT = 0.03  # 3% profit target
TRAIL_STOP_PCT = 0.015     # 1.5% trailing stop once in profit

def load_watchlist():
    """Load today's watchlist"""
    if WATCHLIST_PATH.exists():
        with open(WATCHLIST_PATH) as f:
            return json.load(f)
    return {"watchlist": []}

def load_positions():
    """Load current day positions"""
    if POSITIONS_PATH.exists():
        with open(POSITIONS_PATH) as f:
            return json.load(f)
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "capital": STARTING_CAPITAL,
        "cash": STARTING_CAPITAL,
        "positions": {},
        "closed_trades": [],
        "total_trades": 0,
        "winners": 0,
        "losers": 0,
        "gross_pnl": 0
    }

def save_positions(positions):
    """Save current positions"""
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)

def get_current_price(symbol):
    """Get real-time price for symbol"""
    try:
        ticker = yf.Ticker(symbol)
        # Try to get intraday data
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        # Fallback to daily
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

def get_intraday_data(symbol):
    """Get intraday price data for analysis"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="5m")
        if hist.empty:
            return None
        
        return {
            "symbol": symbol,
            "current": float(hist['Close'].iloc[-1]),
            "open": float(hist['Open'].iloc[0]),
            "high": float(hist['High'].max()),
            "low": float(hist['Low'].min()),
            "vwap": float((hist['Close'] * hist['Volume']).sum() / hist['Volume'].sum()) if hist['Volume'].sum() > 0 else float(hist['Close'].mean()),
            "volume": int(hist['Volume'].sum()),
            "bars": len(hist),
            "trend": "UP" if hist['Close'].iloc[-1] > hist['Close'].iloc[0] else "DOWN",
            "from_high_pct": ((hist['Close'].iloc[-1] - hist['High'].max()) / hist['High'].max()) * 100,
            "from_low_pct": ((hist['Close'].iloc[-1] - hist['Low'].min()) / hist['Low'].min()) * 100,
        }
    except Exception as e:
        return None

def check_entry_signal(symbol, watchlist_data, intraday_data):
    """
    Check if we should enter a position
    
    Entry signals:
    - VWAP reclaim (price crosses above VWAP)
    - Breakout (new high of day)
    - Bounce (off low of day with volume)
    - Momentum (strong trend continuation)
    """
    if not intraday_data:
        return None
    
    signals = []
    current = intraday_data["current"]
    vwap = intraday_data["vwap"]
    high = intraday_data["high"]
    low = intraday_data["low"]
    open_price = intraday_data["open"]
    
    # Get setup info from watchlist
    setups = watchlist_data.get("setups", [])
    change_pct = watchlist_data.get("change_pct", 0)
    
    # VWAP reclaim - price above VWAP and trending up
    if current > vwap and intraday_data["trend"] == "UP":
        signals.append({
            "type": "VWAP_RECLAIM",
            "direction": "LONG",
            "strength": 0.6,
            "reason": f"Price ${current:.2f} above VWAP ${vwap:.2f}"
        })
    
    # VWAP rejection - price below VWAP and trending down
    if current < vwap and intraday_data["trend"] == "DOWN":
        signals.append({
            "type": "VWAP_REJECTION",
            "direction": "SHORT",
            "strength": 0.6,
            "reason": f"Price ${current:.2f} below VWAP ${vwap:.2f}"
        })
    
    # Breakout - within 0.2% of high of day
    if intraday_data["from_high_pct"] > -0.2 and change_pct > 0:
        signals.append({
            "type": "BREAKOUT",
            "direction": "LONG",
            "strength": 0.8,
            "reason": f"Testing HOD ${high:.2f}"
        })
    
    # Breakdown - within 0.2% of low of day  
    if intraday_data["from_low_pct"] < 0.2 and change_pct < 0:
        signals.append({
            "type": "BREAKDOWN",
            "direction": "SHORT",
            "strength": 0.8,
            "reason": f"Testing LOD ${low:.2f}"
        })
    
    # Opening range breakout (if early in session)
    if intraday_data["bars"] < 12:  # First hour
        if current > open_price * 1.005:  # 0.5% above open
            signals.append({
                "type": "ORB_LONG",
                "direction": "LONG",
                "strength": 0.7,
                "reason": f"Opening range breakout, above ${open_price:.2f}"
            })
        elif current < open_price * 0.995:  # 0.5% below open
            signals.append({
                "type": "ORB_SHORT",
                "direction": "SHORT",
                "strength": 0.7,
                "reason": f"Opening range breakdown, below ${open_price:.2f}"
            })
    
    # Boost signals if multiple setups from scanner
    if "BREAKOUT" in setups or "GAP_UP" in setups:
        for s in signals:
            if s["direction"] == "LONG":
                s["strength"] = min(s["strength"] + 0.2, 1.0)
    
    if "BREAKDOWN" in setups or "GAP_DOWN" in setups:
        for s in signals:
            if s["direction"] == "SHORT":
                s["strength"] = min(s["strength"] + 0.2, 1.0)
    
    # Return strongest signal
    if signals:
        return max(signals, key=lambda x: x["strength"])
    return None

def execute_entry(positions, symbol, price, direction, signal, capital_pct=None):
    """Execute a paper trade entry"""
    if capital_pct is None:
        capital_pct = MAX_POSITION_SIZE
    
    # Calculate position size
    available = positions["cash"]
    trade_value = min(available * capital_pct, available)
    
    if trade_value < 100:  # Minimum trade size
        return None
    
    shares = int(trade_value / price)
    if shares < 1:
        return None
    
    actual_value = shares * price
    
    # Calculate stops and targets
    if direction == "LONG":
        stop_price = price * (1 - DEFAULT_STOP_PCT)
        target_price = price * (1 + DEFAULT_TARGET_PCT)
    else:  # SHORT
        stop_price = price * (1 + DEFAULT_STOP_PCT)
        target_price = price * (1 - DEFAULT_TARGET_PCT)
    
    # Create position
    position = {
        "symbol": symbol,
        "direction": direction,
        "shares": shares,
        "entry_price": price,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "entry_signal": signal,
        "stop_price": round(stop_price, 2),
        "target_price": round(target_price, 2),
        "highest_price": price if direction == "LONG" else None,
        "lowest_price": price if direction == "SHORT" else None,
        "trailing_stop": None,
        "cost_basis": actual_value
    }
    
    positions["positions"][symbol] = position
    positions["cash"] -= actual_value
    positions["total_trades"] += 1
    
    save_positions(positions)
    
    # Log trade
    log_trade({
        "action": "ENTRY",
        "symbol": symbol,
        "direction": direction,
        "shares": shares,
        "price": price,
        "signal": signal,
        "stop": stop_price,
        "target": target_price
    })
    
    return position

def check_exit_conditions(position, current_price):
    """Check if position should be exited"""
    symbol = position["symbol"]
    direction = position["direction"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    target_price = position["target_price"]
    
    # Calculate P&L
    if direction == "LONG":
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        hit_stop = current_price <= stop_price
        hit_target = current_price >= target_price
        
        # Update trailing stop if in profit
        if current_price > position.get("highest_price", entry_price):
            position["highest_price"] = current_price
            # Activate trailing stop once 1.5% in profit
            if pnl_pct >= 1.5:
                position["trailing_stop"] = current_price * (1 - TRAIL_STOP_PCT)
        
        # Check trailing stop
        if position.get("trailing_stop") and current_price <= position["trailing_stop"]:
            return {"exit": True, "reason": "TRAILING_STOP", "pnl_pct": pnl_pct}
            
    else:  # SHORT
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
        hit_stop = current_price >= stop_price
        hit_target = current_price <= target_price
        
        # Update trailing stop if in profit
        if current_price < position.get("lowest_price", entry_price):
            position["lowest_price"] = current_price
            if pnl_pct >= 1.5:
                position["trailing_stop"] = current_price * (1 + TRAIL_STOP_PCT)
        
        if position.get("trailing_stop") and current_price >= position["trailing_stop"]:
            return {"exit": True, "reason": "TRAILING_STOP", "pnl_pct": pnl_pct}
    
    if hit_stop:
        return {"exit": True, "reason": "STOP_LOSS", "pnl_pct": pnl_pct}
    
    if hit_target:
        return {"exit": True, "reason": "TARGET_HIT", "pnl_pct": pnl_pct}
    
    return {"exit": False, "pnl_pct": pnl_pct}

def execute_exit(positions, symbol, price, reason):
    """Execute a paper trade exit"""
    if symbol not in positions["positions"]:
        return None
    
    position = positions["positions"][symbol]
    direction = position["direction"]
    shares = position["shares"]
    entry_price = position["entry_price"]
    cost_basis = position["cost_basis"]
    
    # Calculate P&L
    exit_value = shares * price
    if direction == "LONG":
        pnl = exit_value - cost_basis
    else:  # SHORT
        pnl = cost_basis - exit_value  # For shorts, profit when price goes down
    
    pnl_pct = (pnl / cost_basis) * 100
    
    # Update positions
    positions["cash"] += exit_value if direction == "LONG" else (cost_basis + pnl)
    positions["gross_pnl"] += pnl
    
    if pnl > 0:
        positions["winners"] += 1
    else:
        positions["losers"] += 1
    
    # Record closed trade
    closed_trade = {
        **position,
        "exit_price": price,
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "exit_reason": reason,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2)
    }
    positions["closed_trades"].append(closed_trade)
    
    # Remove from open positions
    del positions["positions"][symbol]
    
    save_positions(positions)
    
    # Log trade
    log_trade({
        "action": "EXIT",
        "symbol": symbol,
        "direction": direction,
        "shares": shares,
        "entry_price": entry_price,
        "exit_price": price,
        "reason": reason,
        "pnl": pnl,
        "pnl_pct": pnl_pct
    })
    
    return closed_trade

def log_trade(trade_data):
    """Log trade to file"""
    trade_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(DAY_LOG_PATH, "a") as f:
        f.write(json.dumps(trade_data) + "\n")

def run_trading_cycle():
    """Run one trading cycle - check entries and exits"""
    positions = load_positions()
    watchlist_data = load_watchlist()
    
    # Reset if new day
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if positions.get("date") != today:
        positions = {
            "date": today,
            "capital": STARTING_CAPITAL,
            "cash": STARTING_CAPITAL,
            "positions": {},
            "closed_trades": [],
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "gross_pnl": 0
        }
        save_positions(positions)
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entries": [],
        "exits": [],
        "positions_checked": 0,
        "watchlist_checked": 0
    }
    
    # Check existing positions for exits
    for symbol in list(positions["positions"].keys()):
        position = positions["positions"][symbol]
        current_price = get_current_price(symbol)
        
        if current_price:
            results["positions_checked"] += 1
            exit_check = check_exit_conditions(position, current_price)
            
            if exit_check["exit"]:
                closed = execute_exit(positions, symbol, current_price, exit_check["reason"])
                if closed:
                    results["exits"].append({
                        "symbol": symbol,
                        "reason": exit_check["reason"],
                        "pnl": closed["pnl"],
                        "pnl_pct": closed["pnl_pct"]
                    })
    
    # Check watchlist for entries (if we have room)
    if len(positions["positions"]) < MAX_POSITIONS:
        watchlist = watchlist_data.get("watchlist", [])
        
        for item in watchlist:
            symbol = item["symbol"]
            
            # Skip if already in position
            if symbol in positions["positions"]:
                continue
            
            # Skip if no more room
            if len(positions["positions"]) >= MAX_POSITIONS:
                break
            
            results["watchlist_checked"] += 1
            
            # Get intraday data
            intraday = get_intraday_data(symbol)
            if not intraday:
                continue
            
            # Check for entry signal
            signal = check_entry_signal(symbol, item, intraday)
            
            if signal and signal["strength"] >= 0.6:
                entry = execute_entry(
                    positions, 
                    symbol, 
                    intraday["current"],
                    signal["direction"],
                    signal
                )
                if entry:
                    results["entries"].append({
                        "symbol": symbol,
                        "direction": signal["direction"],
                        "price": intraday["current"],
                        "signal": signal["type"],
                        "reason": signal["reason"]
                    })
    
    # Calculate current status
    portfolio_value = positions["cash"]
    for sym, pos in positions["positions"].items():
        price = get_current_price(sym)
        if price:
            portfolio_value += pos["shares"] * price
    
    results["portfolio_value"] = round(portfolio_value, 2)
    results["cash"] = round(positions["cash"], 2)
    results["open_positions"] = len(positions["positions"])
    results["day_pnl"] = round(portfolio_value - STARTING_CAPITAL, 2)
    results["day_pnl_pct"] = round((portfolio_value / STARTING_CAPITAL - 1) * 100, 2)
    results["total_trades"] = positions["total_trades"]
    results["winners"] = positions["winners"]
    results["losers"] = positions["losers"]
    
    return results

def close_all_positions(reason="END_OF_DAY"):
    """Close all open positions (end of day)"""
    positions = load_positions()
    results = []
    
    for symbol in list(positions["positions"].keys()):
        price = get_current_price(symbol)
        if price:
            closed = execute_exit(positions, symbol, price, reason)
            if closed:
                results.append(closed)
    
    return results

def get_daily_summary():
    """Get summary of today's trading"""
    positions = load_positions()
    
    # Calculate open P&L
    open_pnl = 0
    for sym, pos in positions["positions"].items():
        price = get_current_price(sym)
        if price:
            if pos["direction"] == "LONG":
                open_pnl += (price - pos["entry_price"]) * pos["shares"]
            else:
                open_pnl += (pos["entry_price"] - price) * pos["shares"]
    
    total_pnl = positions["gross_pnl"] + open_pnl
    win_rate = positions["winners"] / positions["total_trades"] * 100 if positions["total_trades"] > 0 else 0
    
    return {
        "date": positions["date"],
        "total_trades": positions["total_trades"],
        "winners": positions["winners"],
        "losers": positions["losers"],
        "win_rate": round(win_rate, 1),
        "realized_pnl": round(positions["gross_pnl"], 2),
        "open_pnl": round(open_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / STARTING_CAPITAL) * 100, 2),
        "open_positions": list(positions["positions"].keys()),
        "closed_trades": positions["closed_trades"]
    }

def print_status(results):
    """Print trading status"""
    print("\n" + "="*60)
    print("📈 DAY TRADING STATUS")
    print(f"   {results['timestamp']}")
    print("="*60)
    
    # Portfolio
    pnl_emoji = "🟢" if results["day_pnl"] >= 0 else "🔴"
    print(f"\n💰 Portfolio: ${results['portfolio_value']:,.2f}")
    print(f"   {pnl_emoji} Day P&L: ${results['day_pnl']:+,.2f} ({results['day_pnl_pct']:+.2f}%)")
    print(f"   Cash: ${results['cash']:,.2f}")
    print(f"   Open Positions: {results['open_positions']}")
    
    # Trades
    print(f"\n📊 Trades: {results['total_trades']} total | {results['winners']}W / {results['losers']}L")
    
    # Entries this cycle
    if results["entries"]:
        print(f"\n🔔 NEW ENTRIES:")
        for e in results["entries"]:
            print(f"   {e['direction']} {e['symbol']} @ ${e['price']:.2f}")
            print(f"   Signal: {e['signal']} - {e['reason']}")
    
    # Exits this cycle
    if results["exits"]:
        print(f"\n🔔 EXITS:")
        for e in results["exits"]:
            emoji = "✅" if e["pnl"] > 0 else "❌"
            print(f"   {emoji} {e['symbol']}: ${e['pnl']:+.2f} ({e['pnl_pct']:+.1f}%) - {e['reason']}")
    
    print()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Day Trading Simulator")
    parser.add_argument("command", choices=["run", "status", "close", "reset"], nargs="?", default="run")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--global", "-g", dest="global_mode", action="store_true",
                        help="Include global markets (crypto/forex 24/7)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # If global mode, run scanner first to refresh watchlist with global assets
        if args.global_mode:
            import subprocess
            subprocess.run([
                str(Path(__file__).parent / "../day-trader/venv/bin/python"),
                str(Path(__file__).parent / "scanner.py"),
                "scan", "--global", "--json"
            ], capture_output=True)
        
        results = run_trading_cycle()
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_status(results)
    
    elif args.command == "status":
        summary = get_daily_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("\n📊 DAILY SUMMARY")
            print("="*40)
            print(f"Date: {summary['date']}")
            print(f"Trades: {summary['total_trades']} ({summary['win_rate']:.0f}% win rate)")
            print(f"Realized P&L: ${summary['realized_pnl']:+,.2f}")
            print(f"Open P&L: ${summary['open_pnl']:+,.2f}")
            print(f"Total P&L: ${summary['total_pnl']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)")
            if summary['open_positions']:
                print(f"Open: {', '.join(summary['open_positions'])}")
            print()
    
    elif args.command == "close":
        print("Closing all positions...")
        closed = close_all_positions()
        print(f"Closed {len(closed)} positions")
        for c in closed:
            print(f"  {c['symbol']}: ${c['pnl']:+.2f}")
    
    elif args.command == "reset":
        positions = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "capital": STARTING_CAPITAL,
            "cash": STARTING_CAPITAL,
            "positions": {},
            "closed_trades": [],
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "gross_pnl": 0
        }
        save_positions(positions)
        print("Day trading account reset to $100,000")

if __name__ == "__main__":
    main()
