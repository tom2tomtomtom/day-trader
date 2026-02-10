#!/usr/bin/env python3
"""
Advanced Day Trading Intelligence System

Comprehensive intraday trading strategy with:
- Real-time market data analysis
- Machine learning-powered signal generation
- Dynamic risk management
- Multi-asset support
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

from market_data_integrator import MarketDataIntegrator
from trading_intelligence_core import TradingIntelligenceCore

class DayTrader:
    def __init__(self, base_dir=None):
        """
        Initialize Day Trading System
        
        Args:
            base_dir (str, optional): Base directory for configuration
        """
        # Directory setup
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.data_dir = os.path.join(self.base_dir, 'data')
        
        # Create necessary directories
        for dir_path in [self.config_dir, self.log_dir, self.data_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(self.log_dir, 'day_trader.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Market data and intelligence modules
        self.market_data_integrator = MarketDataIntegrator(self.base_dir)
        self.intelligence_core = TradingIntelligenceCore(self.base_dir)
        
        # Trading configuration
        self.load_trading_config()
    
    def load_trading_config(self):
        """
        Load or create default trading configuration
        """
        config_path = os.path.join(self.config_dir, 'day_trading_config.json')
        
        default_config = {
            'trading_hours': {
                'start': '09:30',  # Market open (Eastern Time)
                'end': '16:00'     # Market close
            },
            'assets': {
                'stocks': [
                    {'symbol': 'SPY', 'max_allocation': 0.25},
                    {'symbol': 'QQQ', 'max_allocation': 0.25},
                    {'symbol': 'DIA', 'max_allocation': 0.25}
                ],
                'crypto': [
                    {'symbol': 'BTC/USDT', 'max_allocation': 0.15},
                    {'symbol': 'ETH/USDT', 'max_allocation': 0.15}
                ]
            },
            'risk_management': {
                'max_daily_loss': 0.02,   # 2% max daily loss
                'max_single_trade_risk': 0.005,  # 0.5% risk per trade
                'stop_loss_percentage': 0.02,  # 2% stop loss
                'take_profit_percentage': 0.03  # 3% take profit
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = default_config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            self.logger.info("Created default day trading configuration")
    
    def fetch_intraday_data(self, symbols, timeframe='1m', lookback_period=60):
        """
        Fetch high-frequency intraday market data
        
        Args:
            symbols (list): Trading symbols
            timeframe (str): Data resolution (1m, 5m, 15m)
            lookback_period (int): Minutes of historical data
        
        Returns:
            dict: Intraday market data for each symbol
        """
        # Separate stocks and crypto
        stock_symbols = [s for s in symbols if not s.endswith('/USDT')]
        crypto_symbols = [s for s in symbols if s.endswith('/USDT')]
        
        market_data = {}
        
        # Fetch stock data
        if stock_symbols:
            stock_data = self.market_data_integrator.fetch_market_data(
                symbols=stock_symbols,
                asset_type='stocks',
                timeframe=timeframe,
                lookback_period=lookback_period/1440  # Convert minutes to days
            )
            market_data.update(stock_data)
        
        # Fetch crypto data
        if crypto_symbols:
            crypto_data = self.market_data_integrator.fetch_market_data(
                symbols=crypto_symbols,
                asset_type='crypto',
                timeframe=timeframe,
                lookback_period=lookback_period/1440  # Convert minutes to days
            )
            market_data.update(crypto_data)
        
        return market_data
    
    def generate_trading_signals(self, market_data):
        """
        Generate trading signals using multiple strategies
        
        Args:
            market_data (dict): Market data for multiple symbols
        
        Returns:
            dict: Trading signals and confidence levels
        """
        trading_signals = {}
        
        for symbol, data in market_data.items():
            # Use core intelligence for signal generation
            signals = self.intelligence_core.multi_strategy_ensemble(data)
            
            # Calculate signal confidence
            confidence = self.intelligence_core.calculate_signal_confidence(signals)
            
            trading_signals[symbol] = {
                'signals': signals,
                'confidence': confidence
            }
        
        return trading_signals
    
    def risk_management(self, trading_signals, portfolio_value):
        """
        Apply advanced risk management to trading signals
        
        Args:
            trading_signals (dict): Trading signals for multiple symbols
            portfolio_value (float): Current portfolio value
        
        Returns:
            dict: Risk-adjusted trading decisions
        """
        trading_decisions = {}
        
        for symbol, signal_data in trading_signals.items():
            signals = signal_data['signals']
            confidence = signal_data['confidence']
            
            # Get configuration for the symbol
            symbol_config = next(
                (cfg for cfg in 
                 self.config['assets']['stocks'] + self.config['assets']['crypto'] 
                 if cfg['symbol'] == symbol), 
                None
            )
            
            if not symbol_config:
                continue
            
            # Calculate maximum trade size based on risk parameters
            max_trade_risk = portfolio_value * self.config['risk_management']['max_single_trade_risk']
            max_allocation = symbol_config['max_allocation']
            
            # Fetch current market price (simplified)
            market_data = self.fetch_intraday_data([symbol], timeframe='1m', lookback_period=1)
            current_price = market_data[symbol]['close'].iloc[-1]
            
            # Dynamic position sizing
            position_size = max_trade_risk / (current_price * self.config['risk_management']['stop_loss_percentage'])
            position_size *= confidence.get(signals.iloc[-1], 0.5)  # Adjust by signal confidence
            
            # Trading decision
            trading_decisions[symbol] = {
                'signal': signals.iloc[-1],
                'position_size': position_size,
                'confidence': confidence.get(signals.iloc[-1], 0.5)
            }
        
        return trading_decisions
    
    def execute_trades(self, trading_decisions, portfolio_value):
        """
        Execute trading decisions
        
        Args:
            trading_decisions (dict): Risk-adjusted trading decisions
            portfolio_value (float): Current portfolio value
        """
        for symbol, decision in trading_decisions.items():
            signal = decision['signal']
            position_size = decision['position_size']
            confidence = decision['confidence']
            
            # Simplified trade execution logic
            if signal > 0 and confidence > 0.5:
                # Buy signal
                self.logger.info(f"Buying {position_size} units of {symbol}")
            elif signal < 0 and confidence > 0.5:
                # Sell signal
                self.logger.info(f"Selling {position_size} units of {symbol}")
    
    def run_day_trading_cycle(self):
        """
        Execute a complete day trading cycle
        """
        try:
            # Fetch market data for configured symbols
            all_symbols = [
                asset['symbol'] for asset_group in self.config['assets'].values()
                for asset in asset_group
            ]
            
            # Fetch intraday market data
            market_data = self.fetch_intraday_data(
                symbols=all_symbols, 
                timeframe='1m', 
                lookback_period=60
            )
            
            # Generate trading signals
            trading_signals = self.generate_trading_signals(market_data)
            
            # Simulate portfolio value (in a real system, this would be from a broker)
            portfolio_value = 100000  # $100k starting capital
            
            # Apply risk management
            trading_decisions = self.risk_management(
                trading_signals, 
                portfolio_value
            )
            
            # Execute trades
            self.execute_trades(trading_decisions, portfolio_value)
            
            self.logger.info("Day trading cycle completed successfully")
        
        except Exception as e:
            self.logger.error(f"Day trading cycle failed: {e}")
            raise

def main():
    # Initialize and run day trader
    day_trader = DayTrader()
    day_trader.run_day_trading_cycle()

if __name__ == '__main__':
    main()