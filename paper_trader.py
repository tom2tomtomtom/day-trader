#!/usr/bin/env python3
"""
Paper Trading System - AI-Driven Market Prediction
Uses ensemble approach with technical indicators for trading decisions.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Data
import yfinance as yf
import pandas as pd
import numpy as np

# Technical Analysis
try:
    import ta
except ImportError:
    ta = None

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"
TRADES_FILE = Path(__file__).parent / "trades.json"

def load_portfolio():
    """Load or initialize portfolio."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash": 100000.0,
        "positions": {},
        "initial_capital": 100000.0,
        "created_at": datetime.now().isoformat()
    }

def save_portfolio(portfolio):
    """Save portfolio to disk."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

def load_trades():
    """Load trade history."""
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            return json.load(f)
    return []

def save_trade(trade):
    """Append trade to history."""
    trades = load_trades()
    trades.append(trade)
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)

def get_price_data(symbol, period="60d"):
    """Fetch price data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_signals(df):
    """Calculate technical indicators and generate signals."""
    if df is None or len(df) < 20:
        return None, 0, {}
    
    df = df.copy()
    
    # Basic indicators
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean() if len(df) >= 50 else df['Close'].rolling(20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    # Volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    signals = {
        "price": float(latest['Close']),
        "rsi": float(latest['RSI']) if not pd.isna(latest['RSI']) else 50,
        "macd": float(latest['MACD']) if not pd.isna(latest['MACD']) else 0,
        "macd_signal": float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else 0,
        "sma_10": float(latest['SMA_10']) if not pd.isna(latest['SMA_10']) else float(latest['Close']),
        "sma_20": float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else float(latest['Close']),
        "bb_upper": float(latest['BB_Upper']) if not pd.isna(latest['BB_Upper']) else float(latest['Close']) * 1.02,
        "bb_lower": float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else float(latest['Close']) * 0.98,
        "volatility": float(latest['Volatility']) if not pd.isna(latest['Volatility']) else 0.2,
        "trend": "up" if latest['SMA_10'] > latest['SMA_20'] else "down",
        "momentum": float(latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] > 0 else 0
    }
    
    # Score calculation (-100 to +100)
    score = 0
    
    # RSI signal - extreme readings get bigger scores
    if signals['rsi'] < 20:
        score += 45  # Extremely oversold - very bullish
    elif signals['rsi'] < 30:
        score += 30  # Oversold - bullish
    elif signals['rsi'] > 80:
        score -= 45  # Extremely overbought - very bearish
    elif signals['rsi'] > 70:
        score -= 30  # Overbought - bearish
    elif signals['rsi'] < 45:
        score += 15
    elif signals['rsi'] > 55:
        score -= 15
    
    # MACD signal
    if signals['macd'] > signals['macd_signal']:
        score += 25  # Bullish crossover
    else:
        score -= 25
    
    # Trend
    if signals['trend'] == 'up':
        score += 20
    else:
        score -= 20
    
    # Bollinger position
    price = signals['price']
    if price < signals['bb_lower']:
        score += 25  # Near lower band - potential bounce
    elif price > signals['bb_upper']:
        score -= 25  # Near upper band - potential pullback
    
    return df, score, signals

def calculate_position_size(portfolio, price, abs_score, volatility):
    """Calculate position size based on Kelly criterion and risk management."""
    total_value = get_portfolio_value(portfolio)
    
    # Max 2% of portfolio per trade
    max_risk = total_value * 0.02
    
    # Adjust for signal strength (stronger signal = larger position)
    confidence = abs_score / 100  # 0 to 1
    
    # Adjust for volatility (extra conservative)
    vol_adj = max(0.5, min(1.5, 0.2 / max(volatility, 0.1)))
    
    # Calculate position value
    position_value = max_risk * confidence * vol_adj * 5  # Scale up
    
    # Cap at 10% of portfolio
    position_value = min(position_value, total_value * 0.10)
    
    # Calculate shares/units
    shares = position_value / price
    
    return shares, position_value

def get_portfolio_value(portfolio):
    """Calculate total portfolio value."""
    total = portfolio['cash']
    for symbol, position in portfolio['positions'].items():
        df = get_price_data(symbol, period="5d")
        if df is not None and not df.empty:
            current_price = df['Close'].iloc[-1]
            if position['side'] == 'long':
                total += position['quantity'] * current_price
            else:  # short
                # Short P&L = entry_price - current_price
                pnl = (position['entry_price'] - current_price) * position['quantity']
                total += position['notional'] + pnl
    return total

def execute_trade(portfolio, symbol, action, quantity, price, signals):
    """Execute a paper trade."""
    trade = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "value": quantity * price,
        "signals": signals
    }
    
    if action == "BUY":
        cost = quantity * price
        if cost > portfolio['cash']:
            print(f"Insufficient cash for {symbol} BUY")
            return False
        portfolio['cash'] -= cost
        if symbol in portfolio['positions']:
            pos = portfolio['positions'][symbol]
            if pos['side'] == 'long':
                # Add to position
                total_qty = pos['quantity'] + quantity
                avg_price = (pos['quantity'] * pos['entry_price'] + quantity * price) / total_qty
                pos['quantity'] = total_qty
                pos['entry_price'] = avg_price
            else:
                # Close short
                pnl = (pos['entry_price'] - price) * min(quantity, pos['quantity'])
                portfolio['cash'] += pos['notional'] + pnl
                pos['quantity'] -= quantity
                if pos['quantity'] <= 0:
                    del portfolio['positions'][symbol]
        else:
            portfolio['positions'][symbol] = {
                "side": "long",
                "quantity": quantity,
                "entry_price": price,
                "entry_time": datetime.now().isoformat()
            }
    
    elif action == "SELL":
        if symbol in portfolio['positions'] and portfolio['positions'][symbol]['side'] == 'long':
            pos = portfolio['positions'][symbol]
            sell_qty = min(quantity, pos['quantity'])
            portfolio['cash'] += sell_qty * price
            pos['quantity'] -= sell_qty
            if pos['quantity'] <= 0:
                del portfolio['positions'][symbol]
        else:
            print(f"No long position in {symbol} to sell")
            return False
    
    elif action == "SHORT":
        notional = quantity * price
        if notional > portfolio['cash'] * 0.5:  # Require 50% margin
            print(f"Insufficient margin for {symbol} SHORT")
            return False
        if symbol in portfolio['positions']:
            print(f"Already have position in {symbol}")
            return False
        portfolio['positions'][symbol] = {
            "side": "short",
            "quantity": quantity,
            "entry_price": price,
            "notional": notional,
            "entry_time": datetime.now().isoformat()
        }
        portfolio['cash'] -= notional * 0.5  # Hold margin
    
    elif action == "COVER":
        if symbol in portfolio['positions'] and portfolio['positions'][symbol]['side'] == 'short':
            pos = portfolio['positions'][symbol]
            pnl = (pos['entry_price'] - price) * pos['quantity']
            portfolio['cash'] += pos['notional'] * 0.5 + pnl  # Return margin + P&L
            del portfolio['positions'][symbol]
        else:
            print(f"No short position in {symbol} to cover")
            return False
    
    trade['portfolio_value'] = get_portfolio_value(portfolio)
    save_trade(trade)
    save_portfolio(portfolio)
    
    print(f"✅ {action} {quantity:.4f} {symbol} @ ${price:.2f} (${quantity*price:.2f})")
    return True

def analyze_and_trade(symbols=None):
    """Main trading loop - analyze markets and execute trades."""
    if symbols is None:
        symbols = [
    "SOL-USD", "BTC-USD", "ETH-USD", "SPY", "QQQ",  # Original assets
    "PEPE-USD", "DOGE-USD", "SHIB-USD",  # Top meme coins
    "BONK-USD", "WIF-USD",  # Emerging meme coins
    "USELESS-USD"  # Mentioned as high performer
]
    
    portfolio = load_portfolio()
    total_value = get_portfolio_value(portfolio)
    
    print(f"\n{'='*60}")
    print(f"📊 PAPER TRADING SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"💰 Portfolio Value: ${total_value:,.2f}")
    print(f"💵 Cash: ${portfolio['cash']:,.2f}")
    print(f"📈 P&L: ${total_value - portfolio['initial_capital']:+,.2f} ({(total_value/portfolio['initial_capital']-1)*100:+.2f}%)")
    
    if portfolio['positions']:
        print(f"\n📋 Current Positions:")
        for sym, pos in portfolio['positions'].items():
            df = get_price_data(sym, period="5d")
            if df is not None:
                current = df['Close'].iloc[-1]
                if pos['side'] == 'long':
                    pnl = (current - pos['entry_price']) * pos['quantity']
                    pnl_pct = (current / pos['entry_price'] - 1) * 100
                else:
                    pnl = (pos['entry_price'] - current) * pos['quantity']
                    pnl_pct = (pos['entry_price'] / current - 1) * 100
                print(f"  {sym}: {pos['side'].upper()} {pos['quantity']:.4f} @ ${pos['entry_price']:.2f} → ${current:.2f} ({pnl_pct:+.2f}% / ${pnl:+,.2f})")
    
    print(f"\n🔍 Analyzing Markets...")
    
    opportunities = []
    
    for symbol in symbols:
        df, score, signals = calculate_signals(get_price_data(symbol))
        if signals:
            print(f"\n  {symbol}:")
            print(f"    Price: ${signals['price']:.2f} | RSI: {signals['rsi']:.1f} | Trend: {signals['trend']}")
            print(f"    MACD: {signals['macd']:.4f} | Signal: {signals['macd_signal']:.4f}")
            print(f"    Score: {score:+d}/100 | Volatility: {signals['volatility']*100:.1f}%")
            
            opportunities.append({
                "symbol": symbol,
                "score": score,
                "signals": signals,
                "df": df
            })
    
    # Sort by absolute score (strongest signals first)
    opportunities.sort(key=lambda x: abs(x['score']), reverse=True)
    
    print(f"\n🎯 Trading Decisions:")
    
    trades_made = 0
    max_trades = 3  # Limit trades per session
    
    for opp in opportunities:
        if trades_made >= max_trades:
            break
            
        symbol = opp['symbol']
        score = opp['score']
        signals = opp['signals']
        
        # Check existing position
        has_position = symbol in portfolio['positions']
        position_side = portfolio['positions'][symbol]['side'] if has_position else None
        
        # Decision thresholds (lowered for extreme conditions)
        if score >= 25 and not has_position:
            # Strong bullish - open long
            qty, val = calculate_position_size(portfolio, signals['price'], abs(score), signals['volatility'])
            if val >= 100 and portfolio['cash'] >= val:
                print(f"\n  📈 LONG SIGNAL: {symbol} (score: {score:+d})")
                if execute_trade(portfolio, symbol, "BUY", qty, signals['price'], signals):
                    trades_made += 1
                    
        elif score <= -25 and not has_position:
            # Strong bearish - open short
            qty, val = calculate_position_size(portfolio, signals['price'], abs(score), signals['volatility'])
            if val >= 100 and portfolio['cash'] >= val * 0.5:
                print(f"\n  📉 SHORT SIGNAL: {symbol} (score: {score:+d})")
                if execute_trade(portfolio, symbol, "SHORT", qty, signals['price'], signals):
                    trades_made += 1
                    
        elif has_position:
            pos = portfolio['positions'][symbol]
            current_price = signals['price']
            entry_price = pos['entry_price']
            
            if position_side == 'long':
                pnl_pct = (current_price / entry_price - 1) * 100
                # Stop loss at -5% or take profit at +10% or signal reversal
                if pnl_pct <= -5 or pnl_pct >= 10 or score <= -20:
                    print(f"\n  🔄 CLOSE LONG: {symbol} (P&L: {pnl_pct:+.2f}%, score: {score:+d})")
                    execute_trade(portfolio, symbol, "SELL", pos['quantity'], current_price, signals)
                    trades_made += 1
                    
            elif position_side == 'short':
                pnl_pct = (entry_price / current_price - 1) * 100
                # Stop loss at -5% or take profit at +10% or signal reversal
                if pnl_pct <= -5 or pnl_pct >= 10 or score >= 20:
                    print(f"\n  🔄 COVER SHORT: {symbol} (P&L: {pnl_pct:+.2f}%, score: {score:+d})")
                    execute_trade(portfolio, symbol, "COVER", pos['quantity'], current_price, signals)
                    trades_made += 1
    
    if trades_made == 0:
        print("\n  No strong signals - holding current positions")
    
    # Final summary
    portfolio = load_portfolio()
    final_value = get_portfolio_value(portfolio)
    print(f"\n{'='*60}")
    print(f"💼 Final Portfolio Value: ${final_value:,.2f}")
    print(f"{'='*60}\n")
    
    return portfolio

# Ensure the function is defined and can be called
if __name__ == "__main__":
    analyze_and_trade()