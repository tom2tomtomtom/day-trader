#!/usr/bin/env python3
"""
Backtesting Framework for Day Trading Signals

Tests our signals against historical data to measure:
- Win rate
- Average win/loss
- Profit factor
- Max drawdown
- Sharpe ratio (simplified)
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import yfinance as yf

BASE_DIR = Path(__file__).parent
BACKTEST_RESULTS = BASE_DIR / "backtest_results.json"

# Default test parameters
DEFAULT_STOP_LOSS = 0.02  # 2%
DEFAULT_TAKE_PROFIT = 0.03  # 3%
DEFAULT_HOLD_DAYS = 5  # Max hold period


def get_historical_data(symbol, period="6mo"):
    """Fetch historical data for backtesting"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")
        return hist
    except:
        return None


def calculate_signals(hist, signal_type="gap"):
    """
    Calculate buy/sell signals based on historical data
    Returns list of (date, signal, price) tuples
    """
    signals = []
    
    if len(hist) < 20:
        return signals
    
    for i in range(20, len(hist)):
        date = hist.index[i]
        current = hist.iloc[i]
        prev = hist.iloc[i-1]
        
        price = float(current['Close'])
        prev_close = float(prev['Close'])
        change_pct = (price - prev_close) / prev_close * 100
        
        # Volume analysis
        vol_20d = hist['Volume'].iloc[i-20:i].mean()
        current_vol = current['Volume']
        rel_vol = current_vol / vol_20d if vol_20d > 0 else 1
        
        # Range position
        high_20d = hist['High'].iloc[i-20:i].max()
        low_20d = hist['Low'].iloc[i-20:i].min()
        range_pos = (price - low_20d) / (high_20d - low_20d) if high_20d != low_20d else 0.5
        
        signal = None
        
        if signal_type == "gap":
            # Gap fade strategy: fade gaps > 3%
            if change_pct < -3:
                signal = ("BUY", "gap_down_fade")
            elif change_pct > 3:
                signal = ("SELL", "gap_up_fade")
                
        elif signal_type == "mean_reversion":
            # Mean reversion: buy oversold, sell overbought
            if range_pos < 0.1 and change_pct < -2:
                signal = ("BUY", "oversold")
            elif range_pos > 0.9 and change_pct > 2:
                signal = ("SELL", "overbought")
                
        elif signal_type == "momentum":
            # Momentum: follow strong moves with volume
            if change_pct > 2 and rel_vol > 1.5:
                signal = ("BUY", "momentum_long")
            elif change_pct < -2 and rel_vol > 1.5:
                signal = ("SELL", "momentum_short")
                
        elif signal_type == "volume_breakout":
            # Volume breakout: high volume at range extreme
            if range_pos > 0.95 and rel_vol > 2:
                signal = ("BUY", "volume_breakout")
            elif range_pos < 0.05 and rel_vol > 2:
                signal = ("SELL", "volume_breakdown")
        
        elif signal_type == "inverse_hype":
            # Inverse hype for memes: buy on capitulation
            vol_5d = hist['Volume'].iloc[i-5:i].mean()
            vol_trend = vol_5d / vol_20d if vol_20d > 0 else 1
            move_5d = (price - float(hist['Close'].iloc[i-5])) / float(hist['Close'].iloc[i-5]) * 100
            
            # Capitulation: volume spike + price crash + at lows
            if rel_vol > 2 and move_5d < -15 and range_pos < 0.3:
                signal = ("BUY", "capitulation")
            # Hype top: volume dying + extended + momentum fading
            elif vol_trend < 0.6 and range_pos > 0.8 and change_pct < 1:
                signal = ("SELL", "hype_exhaustion")
        
        if signal:
            signals.append({
                "date": date,
                "signal": signal[0],
                "reason": signal[1],
                "entry_price": price,
                "change_pct": change_pct,
                "rel_volume": rel_vol,
                "range_pos": range_pos
            })
    
    return signals


def simulate_trades(hist, signals, stop_loss=DEFAULT_STOP_LOSS, 
                   take_profit=DEFAULT_TAKE_PROFIT, max_hold=DEFAULT_HOLD_DAYS):
    """
    Simulate trades based on signals
    Returns list of completed trades with P&L
    """
    trades = []
    
    for sig in signals:
        entry_date = sig["date"]
        entry_price = sig["entry_price"]
        direction = sig["signal"]
        
        # Find exit
        entry_idx = hist.index.get_loc(entry_date)
        exit_price = None
        exit_reason = None
        exit_date = None
        
        for j in range(1, min(max_hold + 1, len(hist) - entry_idx)):
            future = hist.iloc[entry_idx + j]
            future_date = hist.index[entry_idx + j]
            
            high = float(future['High'])
            low = float(future['Low'])
            close = float(future['Close'])
            
            if direction == "BUY":
                # Check stop loss
                if (low - entry_price) / entry_price <= -stop_loss:
                    exit_price = entry_price * (1 - stop_loss)
                    exit_reason = "stop_loss"
                    exit_date = future_date
                    break
                # Check take profit
                elif (high - entry_price) / entry_price >= take_profit:
                    exit_price = entry_price * (1 + take_profit)
                    exit_reason = "take_profit"
                    exit_date = future_date
                    break
            else:  # SELL (short)
                # Check stop loss
                if (high - entry_price) / entry_price >= stop_loss:
                    exit_price = entry_price * (1 + stop_loss)
                    exit_reason = "stop_loss"
                    exit_date = future_date
                    break
                # Check take profit
                elif (entry_price - low) / entry_price >= take_profit:
                    exit_price = entry_price * (1 - take_profit)
                    exit_reason = "take_profit"
                    exit_date = future_date
                    break
        
        # If no exit yet, exit at max hold
        if exit_price is None and entry_idx + max_hold < len(hist):
            future = hist.iloc[entry_idx + max_hold]
            exit_price = float(future['Close'])
            exit_reason = "time_exit"
            exit_date = hist.index[entry_idx + max_hold]
        
        if exit_price:
            if direction == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                "entry_date": entry_date.isoformat(),
                "exit_date": exit_date.isoformat(),
                "direction": direction,
                "reason": sig["reason"],
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "exit_reason": exit_reason,
                "pnl_pct": round(pnl_pct, 2),
                "win": pnl_pct > 0
            })
    
    return trades


def calculate_metrics(trades):
    """Calculate performance metrics from trades"""
    if not trades:
        return {"error": "No trades"}
    
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    
    total_trades = len(trades)
    win_rate = len(wins) / total_trades * 100
    
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    total_pnl = sum(t["pnl_pct"] for t in trades)
    
    # Calculate drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t["pnl_pct"]
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    
    return {
        "total_trades": total_trades,
        "winners": len(wins),
        "losers": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "total_pnl_pct": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "expectancy": round((win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss), 2)
    }


def run_backtest(symbols, signal_type="gap", period="6mo", 
                stop_loss=DEFAULT_STOP_LOSS, take_profit=DEFAULT_TAKE_PROFIT):
    """Run full backtest on multiple symbols"""
    
    print(f"\n{'='*60}")
    print(f"📊 BACKTEST: {signal_type.upper()} STRATEGY")
    print(f"   Period: {period} | Stop: {stop_loss*100}% | Target: {take_profit*100}%")
    print(f"{'='*60}\n")
    
    all_trades = []
    symbol_results = {}
    
    for symbol in symbols:
        print(f"Testing {symbol}...", end=" ", flush=True)
        
        hist = get_historical_data(symbol, period)
        if hist is None or len(hist) < 30:
            print("SKIP (insufficient data)")
            continue
        
        signals = calculate_signals(hist, signal_type)
        trades = simulate_trades(hist, signals, stop_loss, take_profit)
        
        if trades:
            metrics = calculate_metrics(trades)
            symbol_results[symbol] = metrics
            all_trades.extend(trades)
            print(f"{len(trades)} trades | {metrics['win_rate']}% win | {metrics['total_pnl_pct']:+.1f}% P&L")
        else:
            print("No signals")
    
    # Overall metrics
    overall = calculate_metrics(all_trades) if all_trades else {}
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": signal_type,
        "period": period,
        "parameters": {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "max_hold_days": DEFAULT_HOLD_DAYS
        },
        "symbols_tested": len(symbols),
        "overall_metrics": overall,
        "by_symbol": symbol_results,
        "sample_trades": all_trades[:20]  # First 20 for review
    }
    
    # Save results
    with open(BACKTEST_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    if overall:
        print(f"\n{'='*60}")
        print("📈 OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades:    {overall['total_trades']}")
        print(f"Win Rate:        {overall['win_rate']}%")
        print(f"Avg Win:         +{overall['avg_win_pct']}%")
        print(f"Avg Loss:        {overall['avg_loss_pct']}%")
        print(f"Profit Factor:   {overall['profit_factor']}")
        print(f"Total P&L:       {overall['total_pnl_pct']:+.1f}%")
        print(f"Max Drawdown:    {overall['max_drawdown_pct']}%")
        print(f"Expectancy:      {overall['expectancy']}% per trade")
        print()
    
    return results


# Test universes
STOCKS_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "AMD", "META"]
CRYPTO_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"]
MEME_UNIVERSE = ["DOGE-USD", "SHIB-USD", "GME", "AMC", "MSTR"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Trading Signals")
    parser.add_argument("--strategy", "-s", default="gap", 
                       choices=["gap", "mean_reversion", "momentum", "volume_breakout", "inverse_hype"],
                       help="Strategy to test")
    parser.add_argument("--universe", "-u", default="stocks",
                       choices=["stocks", "crypto", "meme", "all"],
                       help="Universe to test")
    parser.add_argument("--period", "-p", default="6mo", help="Lookback period")
    parser.add_argument("--stop", type=float, default=0.02, help="Stop loss %")
    parser.add_argument("--target", type=float, default=0.03, help="Take profit %")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    # Select universe
    if args.universe == "stocks":
        symbols = STOCKS_UNIVERSE
    elif args.universe == "crypto":
        symbols = CRYPTO_UNIVERSE
    elif args.universe == "meme":
        symbols = MEME_UNIVERSE
    else:
        symbols = STOCKS_UNIVERSE + CRYPTO_UNIVERSE
    
    results = run_backtest(
        symbols=symbols,
        signal_type=args.strategy,
        period=args.period,
        stop_loss=args.stop,
        take_profit=args.target
    )
    
    if args.json:
        print(json.dumps(results, indent=2))
