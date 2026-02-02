#!/usr/bin/env python3
"""
Automation Controller - Manages autonomous trading with human override

Modes:
- FULL_AUTO: Scan, trade, manage positions automatically
- APPROVAL: Queue trades for human approval
- PAUSED: No new trades, but manage existing positions
- STOPPED: No activity
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread, Event
import sys

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "automation_config.json"
PENDING_PATH = BASE_DIR / "pending_trades.json"
LOG_PATH = BASE_DIR / "automation_log.jsonl"

# Import our trading modules
sys.path.insert(0, str(BASE_DIR))
from scanner import run_full_scan
from day_trader import run_trading_cycle, close_all_positions, execute_entry, execute_exit, load_positions, get_current_price, load_watchlist, get_intraday_data, check_entry_signal
from regime_detector import get_market_status

DEFAULT_CONFIG = {
    "mode": "PAUSED",  # FULL_AUTO, APPROVAL, PAUSED, STOPPED
    "scan_interval_minutes": 30,
    "trade_interval_minutes": 5,
    "auto_close_eod": True,
    "eod_close_minutes_before": 15,
    "max_daily_trades": 20,
    "max_daily_loss_pct": 5.0,
    "notifications_enabled": True,
    "last_scan": None,
    "last_trade_cycle": None,
    "trades_today": 0,
    "started_at": None,
}

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            return {**DEFAULT_CONFIG, **config}
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)

def load_pending():
    if PENDING_PATH.exists():
        with open(PENDING_PATH) as f:
            return json.load(f)
    return {"trades": []}

def save_pending(pending):
    with open(PENDING_PATH, "w") as f:
        json.dump(pending, f, indent=2, default=str)

def log_event(event_type, data):
    with open(LOG_PATH, "a") as f:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data
        }
        f.write(json.dumps(entry) + "\n")

def queue_trade(trade_data):
    """Queue a trade for human approval"""
    pending = load_pending()
    trade_data["queued_at"] = datetime.now(timezone.utc).isoformat()
    trade_data["id"] = f"trade_{int(time.time() * 1000)}"
    pending["trades"].append(trade_data)
    save_pending(pending)
    log_event("TRADE_QUEUED", trade_data)
    return trade_data

def approve_trade(trade_id):
    """Approve and execute a pending trade"""
    pending = load_pending()
    trade = None
    
    for i, t in enumerate(pending["trades"]):
        if t["id"] == trade_id:
            trade = pending["trades"].pop(i)
            break
    
    if not trade:
        return {"error": "Trade not found"}
    
    save_pending(pending)
    
    # Execute the trade
    positions = load_positions()
    price = get_current_price(trade["symbol"])
    
    if not price:
        log_event("TRADE_FAILED", {"trade_id": trade_id, "reason": "Could not get price"})
        return {"error": "Could not get current price"}
    
    if trade["action"] == "BUY" or trade["action"] == "LONG":
        result = execute_entry(
            positions,
            trade["symbol"],
            price,
            "LONG",
            {"type": "MANUAL_APPROVAL", "original": trade.get("signal")}
        )
    elif trade["action"] == "SHORT":
        result = execute_entry(
            positions,
            trade["symbol"],
            price,
            "SHORT",
            {"type": "MANUAL_APPROVAL", "original": trade.get("signal")}
        )
    elif trade["action"] == "SELL" or trade["action"] == "CLOSE":
        result = execute_exit(positions, trade["symbol"], price, "MANUAL_APPROVAL")
    else:
        return {"error": f"Unknown action: {trade['action']}"}
    
    log_event("TRADE_APPROVED", {"trade_id": trade_id, "result": result})
    return {"success": True, "result": result}

def reject_trade(trade_id):
    """Reject a pending trade"""
    pending = load_pending()
    
    for i, t in enumerate(pending["trades"]):
        if t["id"] == trade_id:
            trade = pending["trades"].pop(i)
            save_pending(pending)
            log_event("TRADE_REJECTED", {"trade_id": trade_id, "trade": trade})
            return {"success": True}
    
    return {"error": "Trade not found"}

def manual_trade(symbol, action, reason="MANUAL"):
    """Execute a manual trade immediately"""
    positions = load_positions()
    price = get_current_price(symbol)
    
    if not price:
        return {"error": "Could not get current price"}
    
    if action == "BUY":
        result = execute_entry(positions, symbol, price, "LONG", {"type": "MANUAL", "reason": reason})
    elif action == "SHORT":
        result = execute_entry(positions, symbol, price, "SHORT", {"type": "MANUAL", "reason": reason})
    elif action == "CLOSE":
        if symbol in positions["positions"]:
            result = execute_exit(positions, symbol, price, "MANUAL_CLOSE")
        else:
            return {"error": f"No position in {symbol}"}
    else:
        return {"error": f"Unknown action: {action}"}
    
    log_event("MANUAL_TRADE", {"symbol": symbol, "action": action, "result": result})
    return {"success": True, "result": result}

def run_scan_cycle(config):
    """Run the scanner"""
    log_event("SCAN_START", {})
    try:
        results = run_full_scan()
        config["last_scan"] = datetime.now(timezone.utc).isoformat()
        save_config(config)
        log_event("SCAN_COMPLETE", {"watchlist_size": len(results.get("watchlist", []))})
        return results
    except Exception as e:
        log_event("SCAN_ERROR", {"error": str(e)})
        return None

def run_trade_cycle(config):
    """Run trading cycle based on mode"""
    mode = config["mode"]
    
    if mode == "STOPPED":
        return None
    
    # Check daily limits
    positions = load_positions()
    if positions["total_trades"] >= config["max_daily_trades"]:
        log_event("LIMIT_REACHED", {"reason": "max_daily_trades"})
        return None
    
    # Check loss limit
    day_pnl_pct = (positions["gross_pnl"] / 100000) * 100  # Assuming 100k start
    if day_pnl_pct <= -config["max_daily_loss_pct"]:
        log_event("LIMIT_REACHED", {"reason": "max_daily_loss", "pnl_pct": day_pnl_pct})
        return None
    
    if mode == "FULL_AUTO":
        # Run full automatic trading
        results = run_trading_cycle()
        config["last_trade_cycle"] = datetime.now(timezone.utc).isoformat()
        config["trades_today"] = positions["total_trades"]
        save_config(config)
        log_event("TRADE_CYCLE", {"mode": "FULL_AUTO", "results": results})
        return results
    
    elif mode == "APPROVAL":
        # Check for signals but queue for approval
        watchlist_data = load_watchlist()
        watchlist = watchlist_data.get("watchlist", [])
        
        for item in watchlist[:5]:  # Top 5 candidates
            symbol = item["symbol"]
            
            # Skip if already in position
            if symbol in positions["positions"]:
                continue
            
            intraday = get_intraday_data(symbol)
            if not intraday:
                continue
            
            signal = check_entry_signal(symbol, item, intraday)
            
            if signal and signal["strength"] >= 0.6:
                # Queue for approval instead of executing
                queue_trade({
                    "symbol": symbol,
                    "action": signal["direction"],
                    "price": intraday["current"],
                    "signal": signal,
                    "reason": signal["reason"]
                })
        
        config["last_trade_cycle"] = datetime.now(timezone.utc).isoformat()
        save_config(config)
        log_event("TRADE_CYCLE", {"mode": "APPROVAL"})
        return {"mode": "APPROVAL", "queued": True}
    
    elif mode == "PAUSED":
        # Only manage existing positions (check stops/targets)
        results = {"mode": "PAUSED", "positions_checked": 0, "exits": []}
        
        for symbol in list(positions["positions"].keys()):
            price = get_current_price(symbol)
            if price:
                results["positions_checked"] += 1
                # Check exits handled by day_trader module
        
        log_event("TRADE_CYCLE", {"mode": "PAUSED", "results": results})
        return results
    
    return None

def check_eod_close(config):
    """Check if we should close all positions before market close"""
    if not config["auto_close_eod"]:
        return
    
    market_status = get_market_status()
    
    # Check if US market is about to close
    if not market_status.get("US", {}).get("open", False):
        return
    
    now = datetime.now(timezone.utc)
    # US market closes at 21:00 UTC
    minutes_to_close = (21 * 60) - (now.hour * 60 + now.minute)
    
    if 0 < minutes_to_close <= config["eod_close_minutes_before"]:
        log_event("EOD_CLOSE", {"minutes_to_close": minutes_to_close})
        close_all_positions("END_OF_DAY")

def get_status():
    """Get current automation status"""
    config = load_config()
    pending = load_pending()
    positions = load_positions()
    market_status = get_market_status()
    
    active_markets = [m for m, s in market_status.items() if s.get("open", False)]
    
    return {
        "mode": config["mode"],
        "active_markets": active_markets,
        "last_scan": config.get("last_scan"),
        "last_trade_cycle": config.get("last_trade_cycle"),
        "trades_today": positions["total_trades"],
        "pending_trades": len(pending["trades"]),
        "open_positions": len(positions["positions"]),
        "config": config
    }

def set_mode(new_mode):
    """Change automation mode"""
    if new_mode not in ["FULL_AUTO", "APPROVAL", "PAUSED", "STOPPED"]:
        return {"error": f"Invalid mode: {new_mode}"}
    
    config = load_config()
    old_mode = config["mode"]
    config["mode"] = new_mode
    
    if new_mode in ["FULL_AUTO", "APPROVAL"] and not config.get("started_at"):
        config["started_at"] = datetime.now(timezone.utc).isoformat()
    
    save_config(config)
    log_event("MODE_CHANGE", {"from": old_mode, "to": new_mode})
    
    return {"success": True, "mode": new_mode}

# CLI interface
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Automation Controller")
    parser.add_argument("command", choices=[
        "status", "start", "pause", "stop", "approval",
        "scan", "trade", "approve", "reject", "manual",
        "pending", "close-all"
    ])
    parser.add_argument("--id", help="Trade ID for approve/reject")
    parser.add_argument("--symbol", help="Symbol for manual trade")
    parser.add_argument("--action", choices=["BUY", "SHORT", "CLOSE"])
    parser.add_argument("--json", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "status":
        result = get_status()
    elif args.command == "start":
        result = set_mode("FULL_AUTO")
    elif args.command == "pause":
        result = set_mode("PAUSED")
    elif args.command == "stop":
        result = set_mode("STOPPED")
    elif args.command == "approval":
        result = set_mode("APPROVAL")
    elif args.command == "scan":
        config = load_config()
        result = run_scan_cycle(config)
    elif args.command == "trade":
        config = load_config()
        result = run_trade_cycle(config)
    elif args.command == "approve":
        if not args.id:
            result = {"error": "Need --id"}
        else:
            result = approve_trade(args.id)
    elif args.command == "reject":
        if not args.id:
            result = {"error": "Need --id"}
        else:
            result = reject_trade(args.id)
    elif args.command == "manual":
        if not args.symbol or not args.action:
            result = {"error": "Need --symbol and --action"}
        else:
            result = manual_trade(args.symbol, args.action)
    elif args.command == "pending":
        result = load_pending()
    elif args.command == "close-all":
        result = close_all_positions("MANUAL_CLOSE_ALL")
    else:
        result = {"error": "Unknown command"}
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if args.command == "status":
            print(f"\n🤖 AUTOMATION STATUS")
            print("=" * 40)
            print(f"Mode: {result['mode']}")
            print(f"Active Markets: {', '.join(result['active_markets']) or 'None'}")
            print(f"Trades Today: {result['trades_today']}")
            print(f"Pending Approval: {result['pending_trades']}")
            print(f"Open Positions: {result['open_positions']}")
            print(f"Last Scan: {result['last_scan'] or 'Never'}")
            print(f"Last Trade Cycle: {result['last_trade_cycle'] or 'Never'}")
        elif args.command == "pending":
            trades = result.get("trades", [])
            if trades:
                print(f"\n⏳ PENDING TRADES ({len(trades)})")
                print("=" * 40)
                for t in trades:
                    print(f"\nID: {t['id']}")
                    print(f"  {t['action']} {t['symbol']} @ ${t['price']:.2f}")
                    print(f"  Reason: {t.get('reason', 'N/A')}")
                    print(f"  Queued: {t['queued_at']}")
            else:
                print("\nNo pending trades")
        else:
            print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
