#!/usr/bin/env python3
"""
EDGE HUNTER
Aggressive Trading System for Volatile Assets
"""

import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta

class EdgeHunter:
    def __init__(self, initial_capital=250000):
        # ULTRA-AGGRESSIVE CONFIGURATION
        self.capital = initial_capital
        self.max_leverage = 10
        self.risk_per_trade = 0.05  # 5% risk per trade
        
        # MARKETS TO DOMINATE (Most liquid markets)
        self.markets = {
            'top_crypto': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT'],
            'alt_crypto': ['DOGE/USDT', 'LINK/USDT', 'DOT/USDT', 'AVAX/USDT']
        }
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # Futures trading
            }
        })
    
    def _fetch_market_data(self, symbol):
        """Fetch market data via Binance"""
        try:
            # Fetch 1-hour candles for the last 30 periods
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=30)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Advanced volatility metrics
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['volume_spike'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            return df.dropna()
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
            return None
    
    def _calculate_edge_score(self, data):
        """Calculate trading edge potential"""
        if data is None or len(data) < 10:
            return 0
        
        # Volatility indicators
        recent_returns = data['returns'].tail(10)
        volatility = data['volatility'].iloc[-1]
        volume_spike = data['volume_spike'].iloc[-1]
        
        # Edge scoring mechanism
        edge_score = 0
        
        # Volatility boost
        if volatility > 1.5:  # Extremely volatile
            edge_score += 2
        elif volatility > 1:  # High volatility
            edge_score += 1
        
        # Volume spike
        if volume_spike > 3:  # 3x average volume
            edge_score += 1
        
        # Momentum
        if recent_returns.mean() > 0.05:  # Strong positive momentum
            edge_score += 1
        elif recent_returns.mean() < -0.05:  # Strong negative momentum
            edge_score += 1
        
        return edge_score
    
    def _generate_trading_signal(self, data):
        """Generate aggressive trading signals"""
        if data is None:
            return 0
        
        edge_score = self._calculate_edge_score(data)
        recent_returns = data['returns'].tail(10)
        
        # Signal generation
        if edge_score >= 3 and recent_returns.mean() > 0.05:
            return 2  # STRONG BUY
        elif edge_score >= 3 and recent_returns.mean() < -0.05:
            return -2  # STRONG SELL
        elif edge_score >= 2 and recent_returns.mean() > 0:
            return 1  # BUY
        elif edge_score >= 2 and recent_returns.mean() < 0:
            return -1  # SELL
        
        return 0  # NEUTRAL
    
    def _calculate_position_size(self, current_price, signal_strength):
        """Dynamic position sizing for edge trades"""
        risk_amount = self.capital * self.risk_per_trade
        position_multiplier = abs(signal_strength)
        max_position = risk_amount * position_multiplier * self.max_leverage
        shares = max_position / current_price
        return int(shares)
    
    def hunt_edges(self):
        """Aggressive edge hunting strategy"""
        print("🐺 EDGE HUNTER ACTIVATED 🐺")
        print("=" * 50)
        
        executed_trades = []
        total_potential_profit = 0
        
        # Scan all markets
        for market_type, symbols in self.markets.items():
            for symbol in symbols:
                # Fetch market data
                data = self._fetch_market_data(symbol)
                
                # Skip if no data
                if data is None or data.empty:
                    continue
                
                # Generate trading signal
                signal = self._generate_trading_signal(data)
                
                # Skip neutral signals
                if signal == 0:
                    continue
                
                # Current market price
                current_price = data['close'].iloc[-1]
                
                # Calculate position
                shares = self._calculate_position_size(current_price, signal)
                
                # Trade value and potential profit
                trade_value = shares * current_price
                potential_profit = trade_value * 0.05 * signal  # 5% expected move
                
                # Trade details
                trade = {
                    'symbol': symbol,
                    'market_type': market_type,
                    'signal': signal,
                    'shares': shares,
                    'entry_price': current_price,
                    'potential_profit': potential_profit,
                    'timestamp': datetime.now()
                }
                
                executed_trades.append(trade)
                total_potential_profit += potential_profit
        
        # Performance summary
        print("\n📊 EDGE HUNT RESULTS:")
        print(f"Total Trades: {len(executed_trades)}")
        print(f"Potential Profit: ${total_potential_profit:,.2f}")
        
        for trade in executed_trades:
            print(f"🎯 {trade['symbol']} ({trade['market_type']}) → {trade['signal']} | "
                  f"Shares: {trade['shares']:.2f} | "
                  f"Potential Profit: ${trade['potential_profit']:,.2f}")
        
        return {
            'total_trades': len(executed_trades),
            'total_potential_profit': total_potential_profit,
            'trades': executed_trades
        }

def main():
    hunter = EdgeHunter(initial_capital=500000)
    hunter.hunt_edges()

if __name__ == "__main__":
    main()