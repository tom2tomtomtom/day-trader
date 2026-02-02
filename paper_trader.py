#!/usr/bin/env python3
"""
Paper Trading System

Executes dummy trades based on regime signals
Tracks performance with fake $100,000 starting capital
"""

import json
from datetime import datetime
from pathlib import Path
import yfinance as yf

BASE_DIR = Path(__file__).parent
PORTFOLIO_PATH = BASE_DIR / "portfolio.json"
TRADE_LOG_PATH = BASE_DIR / "trade_log.jsonl"

STARTING_CAPITAL = 100000

def load_portfolio():
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    return {
        "created": datetime.now().isoformat(),
        "starting_capital": STARTING_CAPITAL,
        "cash": STARTING_CAPITAL,
        "positions": {},  # symbol -> {shares, avg_cost, entry_date}
        "total_trades": 0,
        "winning_trades": 0,
        "total_realized_pnl": 0
    }

def save_portfolio(portfolio):
    portfolio["last_updated"] = datetime.now().isoformat()
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2)

def log_trade(trade):
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(trade) + "\n")

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    if hist.empty:
        return None
    return float(hist['Close'].iloc[-1])

def get_portfolio_value(portfolio):
    """Calculate total portfolio value"""
    total = portfolio["cash"]
    
    for symbol, pos in portfolio["positions"].items():
        price = get_current_price(symbol)
        if price:
            total += price * pos["shares"]
    
    return total

def execute_trade(action, symbol, amount_or_pct, reason, regime_data=None):
    """
    Execute a paper trade
    
    action: BUY, SELL, SELL_ALL
    amount_or_pct: dollar amount or percentage of portfolio (0-1)
    """
    portfolio = load_portfolio()
    price = get_current_price(symbol)
    
    if not price:
        return {"error": f"Could not get price for {symbol}"}
    
    trade = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "price": price,
        "reason": reason,
        "regime": regime_data
    }
    
    if action == "BUY":
        # Determine amount to invest
        if amount_or_pct <= 1:
            # Percentage of portfolio
            portfolio_value = get_portfolio_value(portfolio)
            invest_amount = portfolio_value * amount_or_pct
        else:
            invest_amount = amount_or_pct
        
        # Don't exceed cash
        invest_amount = min(invest_amount, portfolio["cash"])
        
        if invest_amount < 100:
            return {"error": "Insufficient funds", "cash": portfolio["cash"]}
        
        shares = int(invest_amount / price)
        cost = shares * price
        
        # Update portfolio
        portfolio["cash"] -= cost
        
        if symbol in portfolio["positions"]:
            # Average in
            pos = portfolio["positions"][symbol]
            total_shares = pos["shares"] + shares
            total_cost = pos["avg_cost"] * pos["shares"] + cost
            pos["avg_cost"] = total_cost / total_shares
            pos["shares"] = total_shares
        else:
            portfolio["positions"][symbol] = {
                "shares": shares,
                "avg_cost": price,
                "entry_date": datetime.now().isoformat()
            }
        
        trade["shares"] = shares
        trade["cost"] = cost
        trade["new_cash"] = portfolio["cash"]
        
    elif action in ["SELL", "SELL_ALL"]:
        if symbol not in portfolio["positions"]:
            return {"error": f"No position in {symbol}"}
        
        pos = portfolio["positions"][symbol]
        
        if action == "SELL_ALL":
            shares_to_sell = pos["shares"]
        else:
            if amount_or_pct <= 1:
                shares_to_sell = int(pos["shares"] * amount_or_pct)
            else:
                shares_to_sell = min(int(amount_or_pct / price), pos["shares"])
        
        proceeds = shares_to_sell * price
        cost_basis = shares_to_sell * pos["avg_cost"]
        pnl = proceeds - cost_basis
        pnl_pct = (price / pos["avg_cost"] - 1) * 100
        
        # Update portfolio
        portfolio["cash"] += proceeds
        pos["shares"] -= shares_to_sell
        
        if pos["shares"] <= 0:
            del portfolio["positions"][symbol]
        
        # Track stats
        portfolio["total_trades"] += 1
        portfolio["total_realized_pnl"] += pnl
        if pnl > 0:
            portfolio["winning_trades"] += 1
        
        trade["shares"] = shares_to_sell
        trade["proceeds"] = proceeds
        trade["pnl"] = round(pnl, 2)
        trade["pnl_pct"] = round(pnl_pct, 2)
        trade["new_cash"] = portfolio["cash"]
    
    save_portfolio(portfolio)
    log_trade(trade)
    
    return trade

def get_status():
    """Get current portfolio status"""
    portfolio = load_portfolio()
    
    positions_detail = []
    total_unrealized = 0
    
    for symbol, pos in portfolio["positions"].items():
        price = get_current_price(symbol)
        if price:
            market_value = price * pos["shares"]
            cost_basis = pos["avg_cost"] * pos["shares"]
            unrealized_pnl = market_value - cost_basis
            unrealized_pct = (price / pos["avg_cost"] - 1) * 100
            total_unrealized += unrealized_pnl
            
            positions_detail.append({
                "symbol": symbol,
                "shares": pos["shares"],
                "avg_cost": round(pos["avg_cost"], 2),
                "current_price": round(price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pct": round(unrealized_pct, 2)
            })
    
    portfolio_value = get_portfolio_value(portfolio)
    total_return = portfolio_value - STARTING_CAPITAL
    total_return_pct = (portfolio_value / STARTING_CAPITAL - 1) * 100
    
    return {
        "timestamp": datetime.now().isoformat(),
        "starting_capital": STARTING_CAPITAL,
        "cash": round(portfolio["cash"], 2),
        "positions": positions_detail,
        "portfolio_value": round(portfolio_value, 2),
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "realized_pnl": round(portfolio["total_realized_pnl"], 2),
        "total_trades": portfolio["total_trades"],
        "winning_trades": portfolio["winning_trades"],
        "win_rate": round(portfolio["winning_trades"] / portfolio["total_trades"] * 100, 1) if portfolio["total_trades"] > 0 else 0
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paper Trading")
    parser.add_argument("command", choices=["status", "buy", "sell", "reset"])
    parser.add_argument("--symbol", "-s", default="SPY")
    parser.add_argument("--amount", "-a", type=float, default=0.1, help="Amount or percentage (0-1)")
    parser.add_argument("--reason", "-r", default="manual")
    parser.add_argument("--json", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "status":
        status = get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\n" + "="*50)
            print("PAPER TRADING PORTFOLIO")
            print("="*50)
            print(f"\n💰 Cash: ${status['cash']:,.2f}")
            print(f"📊 Portfolio Value: ${status['portfolio_value']:,.2f}")
            print(f"📈 Total Return: ${status['total_return']:+,.2f} ({status['total_return_pct']:+.2f}%)")
            
            if status["positions"]:
                print("\n📋 POSITIONS:")
                for p in status["positions"]:
                    emoji = "🟢" if p["unrealized_pct"] > 0 else "🔴"
                    print(f"   {emoji} {p['symbol']}: {p['shares']} shares @ ${p['avg_cost']:.2f}")
                    print(f"      Current: ${p['current_price']:.2f} | P&L: ${p['unrealized_pnl']:+.2f} ({p['unrealized_pct']:+.1f}%)")
            
            if status["total_trades"] > 0:
                print(f"\n📊 STATS:")
                print(f"   Trades: {status['total_trades']} | Win rate: {status['win_rate']:.1f}%")
                print(f"   Realized P&L: ${status['realized_pnl']:+,.2f}")
            print()
    
    elif args.command == "buy":
        result = execute_trade("BUY", args.symbol, args.amount, args.reason)
        print(json.dumps(result, indent=2))
    
    elif args.command == "sell":
        result = execute_trade("SELL_ALL", args.symbol, 1.0, args.reason)
        print(json.dumps(result, indent=2))
    
    elif args.command == "reset":
        if PORTFOLIO_PATH.exists():
            PORTFOLIO_PATH.unlink()
        if TRADE_LOG_PATH.exists():
            TRADE_LOG_PATH.unlink()
        print("Portfolio reset to $100,000")

if __name__ == "__main__":
    main()
