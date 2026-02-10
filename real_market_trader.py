#!/usr/bin/env python3
"""
Real Market Trading Intelligence System

Comprehensive trading platform designed for:
- Live market data integration
- Advanced risk management
- Automated trading execution
- Performance tracking and optimization
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ccxt  # Cryptocurrency exchange library
import alpaca_trade_api as tradeapi  # Stock trading
import yfinance as yf  # Backup market data

from trading_intelligence_core import TradingIntelligenceCore, APIKeyManager

class RealMarketTrader:
    def __init__(self, base_dir=None):
        """
        Initialize Real Market Trading System
        
        Args:
            base_dir (str, optional): Base directory for configuration
        """
        # Core intelligence setup
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        
        # Create necessary directories
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(self.log_dir, 'real_market_trader.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize core trading intelligence
        self.intelligence_core = TradingIntelligenceCore()
        self.api_key_manager = APIKeyManager()
        
        # Trading platforms configuration
        self.platforms = {
            'stocks': self._setup_stock_trading(),
            'crypto': self._setup_crypto_trading(),
        }
        
        # Risk management configuration
        self.risk_config = {
            'max_portfolio_risk': 0.05,  # 5% max portfolio risk
            'max_single_trade_risk': 0.02,  # 2% max risk per trade
            'stop_loss_percentage': 0.03,  # 3% stop loss
            'take_profit_percentage': 0.05,  # 5% take profit
        }
        
        # Watchlist configuration
        self.watchlist = self._load_watchlist()
    
    def _setup_stock_trading(self):
        """
        Set up stock trading platform (Alpaca)
        
        Returns:
            dict: Configured stock trading platform
        """
        # Retrieve Alpaca API keys
        alpaca_key = self.api_key_manager.get_key('alpaca')
        if not alpaca_key:
            self.logger.warning("No Alpaca API keys found. Stock trading disabled.")
            return None
        
        try:
            # Parse API keys (assuming key is in format "API_KEY,SECRET_KEY")
            api_key, secret_key = alpaca_key.split(',')
            
            # Initialize Alpaca trading API
            return {
                'api': tradeapi.REST(
                    key_id=api_key, 
                    secret_key=secret_key, 
                    base_url='https://paper.iex.cloud'  # Paper trading URL
                ),
                'key_id': api_key
            }
        except Exception as e:
            self.logger.error(f"Failed to set up Alpaca trading: {e}")
            return None
    
    def _setup_crypto_trading(self):
        """
        Set up cryptocurrency trading platforms
        
        Returns:
            dict: Configured crypto trading platforms
        """
        crypto_platforms = {}
        
        # Supported crypto exchanges
        exchanges = [
            'binance', 'coinbase', 'kraken', 
            'bybit', 'okx', 'kucoin'
        ]
        
        for exchange_name in exchanges:
            # Retrieve API keys for each exchange
            key = self.api_key_manager.get_key(exchange_name)
            if not key:
                continue
            
            try:
                # Parse API keys (assuming format "API_KEY,SECRET_KEY")
                api_key, secret_key = key.split(',')
                
                # Initialize exchange
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future'  # Futures trading
                    }
                })
                
                crypto_platforms[exchange_name] = {
                    'api': exchange,
                    'key_id': api_key
                }
            except Exception as e:
                self.logger.error(f"Failed to set up {exchange_name} trading: {e}")
        
        return crypto_platforms
    
    def _load_watchlist(self):
        """
        Load trading watchlist from configuration
        
        Returns:
            dict: Configured watchlist with trading parameters
        """
        watchlist_path = os.path.join(self.config_dir, 'watchlist.json')
        
        # Default watchlist if no config exists
        default_watchlist = {
            'stocks': [
                {'symbol': 'AAPL', 'allocation': 0.2},
                {'symbol': 'MSFT', 'allocation': 0.2},
                {'symbol': 'GOOGL', 'allocation': 0.2}
            ],
            'crypto': [
                {'symbol': 'BTC/USDT', 'allocation': 0.3},
                {'symbol': 'ETH/USDT', 'allocation': 0.2}
            ]
        }
        
        try:
            with open(watchlist_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Save default watchlist
            with open(watchlist_path, 'w') as f:
                json.dump(default_watchlist, f, indent=4)
            
            self.logger.info("Created default watchlist configuration")
            return default_watchlist
    
    def _fetch_real_market_data(self, symbol, asset_type='stock'):
        """
        Fetch real-time market data for a given symbol
        
        Args:
            symbol (str): Trading symbol
            asset_type (str): Type of asset (stock or crypto)
        
        Returns:
            pd.DataFrame: Market data
        """
        try:
            if asset_type == 'stock':
                # Fetch stock data using yfinance
                stock = yf.Ticker(symbol)
                data = stock.history(period='1mo', interval='1d')
            elif asset_type == 'crypto':
                # Fetch crypto data from exchanges
                exchanges = list(self.platforms['crypto'].values())
                if not exchanges:
                    raise ValueError("No crypto exchanges configured")
                
                # Use first available exchange
                exchange = exchanges[0]['api']
                ohlcv = exchange.fetch_ohlcv(symbol, '1d')
                data = pd.DataFrame(
                    ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def generate_trading_signals(self, market_data):
        """
        Generate trading signals using core intelligence
        
        Args:
            market_data (pd.DataFrame): Market data
        
        Returns:
            dict: Trading signals and confidence
        """
        # Use the core intelligence system to generate signals
        ensemble_signals = self.intelligence_core.multi_strategy_ensemble(market_data)
        
        # Calculate signal confidence
        confidence = self.intelligence_core.calculate_signal_confidence(ensemble_signals)
        
        return {
            'signals': ensemble_signals,
            'confidence': confidence
        }
    
    def execute_trades(self, trading_signals):
        """
        Execute trades based on generated signals
        
        Args:
            trading_signals (dict): Trading signals and confidence
        """
        # Placeholder for trade execution logic
        signals = trading_signals['signals']
        confidence = trading_signals['confidence']
        
        # Execute stock trades
        if self.platforms['stocks']:
            self._execute_stock_trades(signals, confidence)
        
        # Execute crypto trades
        if self.platforms['crypto']:
            self._execute_crypto_trades(signals, confidence)
    
    def _execute_stock_trades(self, signals, confidence):
        """
        Execute stock trades via Alpaca
        
        Args:
            signals (pd.Series): Trading signals
            confidence (dict): Signal confidence levels
        """
        alpaca_api = self.platforms['stocks']['api']
        
        for stock in self.watchlist['stocks']:
            symbol = stock['symbol']
            allocation = stock['allocation']
            
            # Get signal and confidence
            signal = signals.get(symbol, 0)
            conf = confidence.get(signal, 0.5)
            
            try:
                # Get current account info
                account = alpaca_api.get_account()
                portfolio_value = float(account.equity)
                
                # Calculate trade size
                trade_size = portfolio_value * allocation * conf
                
                # Place order based on signal
                if signal > 0:  # Buy signal
                    alpaca_api.submit_order(
                        symbol=symbol,
                        qty=trade_size / alpaca_api.get_last_quote(symbol).askprice,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                elif signal < 0:  # Sell signal
                    alpaca_api.submit_order(
                        symbol=symbol,
                        qty=trade_size / alpaca_api.get_last_quote(symbol).bidprice,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
            except Exception as e:
                self.logger.error(f"Stock trade error for {symbol}: {e}")
    
    def _execute_crypto_trades(self, signals, confidence):
        """
        Execute cryptocurrency trades
        
        Args:
            signals (pd.Series): Trading signals
            confidence (dict): Signal confidence levels
        """
        for exchange_name, platform in self.platforms['crypto'].items():
            exchange = platform['api']
            
            for crypto in self.watchlist['crypto']:
                symbol = crypto['symbol']
                allocation = crypto['allocation']
                
                # Get signal and confidence
                signal = signals.get(symbol, 0)
                conf = confidence.get(signal, 0.5)
                
                try:
                    # Get account balance
                    balance = exchange.fetch_balance()
                    portfolio_value = balance['total']['USDT']
                    
                    # Calculate trade size
                    trade_size = portfolio_value * allocation * conf
                    
                    # Fetch current market price
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Place order based on signal
                    if signal > 0:  # Buy signal
                        exchange.create_market_buy_order(symbol, trade_size / current_price)
                    elif signal < 0:  # Sell signal
                        exchange.create_market_sell_order(symbol, trade_size / current_price)
                    
                except Exception as e:
                    self.logger.error(f"Crypto trade error for {symbol} on {exchange_name}: {e}")
    
    def run_trading_cycle(self):
        """
        Execute a complete trading cycle
        """
        try:
            # Comprehensive trading workflow
            self.logger.info("Starting trading cycle")
            
            # Fetch and process market data for each asset
            all_market_data = {}
            
            # Process stocks
            for stock in self.watchlist['stocks']:
                market_data = self._fetch_real_market_data(stock['symbol'], 'stock')
                if market_data is not None:
                    all_market_data[stock['symbol']] = market_data
            
            # Process crypto
            for crypto in self.watchlist['crypto']:
                market_data = self._fetch_real_market_data(crypto['symbol'], 'crypto')
                if market_data is not None:
                    all_market_data[crypto['symbol']] = market_data
            
            # Generate trading signals
            trading_signals = {}
            for symbol, market_data in all_market_data.items():
                signals = self.generate_trading_signals(market_data)
                trading_signals[symbol] = signals
            
            # Execute trades
            self.execute_trades(trading_signals)
            
            self.logger.info("Trading cycle completed successfully")
        
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
    
    def continuous_trading_loop(self, interval_minutes=15):
        """
        Continuously run trading cycles
        
        Args:
            interval_minutes (int): Time between trading cycles
        """
        while True:
            try:
                self.run_trading_cycle()
                
                # Wait before next cycle
                self.logger.info(f"Waiting {interval_minutes} minutes before next cycle")
                time.sleep(interval_minutes * 60)
            
            except KeyboardInterrupt:
                self.logger.info("Trading loop terminated by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                # Wait before retrying
                time.sleep(interval_minutes * 60)

def main():
    trader = RealMarketTrader()
    trader.continuous_trading_loop()

if __name__ == '__main__':
    main()