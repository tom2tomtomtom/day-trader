#!/usr/bin/env python3
"""
Advanced Market Data Integrator

Comprehensive solution for:
- Multi-source real-time market data collection
- Data normalization and preprocessing
- Advanced feature engineering
- Cross-market correlation analysis
"""

import os
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import ccxt
import requests

class MarketDataIntegrator:
    def __init__(self, base_dir=None):
        """
        Initialize Market Data Integrator
        
        Args:
            base_dir (str, optional): Base directory for data storage
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'market_data')
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(self.data_dir, 'market_data_integrator.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Data sources configuration
        self.data_sources = {
            'stocks': {
                'yfinance': self._fetch_yfinance_data,
                'alpha_vantage': self._fetch_alpha_vantage_data,
                'polygon': self._fetch_polygon_data
            },
            'crypto': {
                'binance': self._fetch_binance_data,
                'coinbase': self._fetch_coinbase_data,
                'kraken': self._fetch_kraken_data
            },
            'forex': {
                'exchangerate_api': self._fetch_forex_data
            }
        }
    
    def fetch_market_data(
        self, 
        symbols: List[str], 
        asset_type: str = 'stocks', 
        timeframe: str = '1d',
        lookback_period: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data from multiple sources
        
        Args:
            symbols (List[str]): List of trading symbols
            asset_type (str): Type of assets (stocks, crypto, forex)
            timeframe (str): Data timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            lookback_period (int): Number of days of historical data
        
        Returns:
            Dict[str, pd.DataFrame]: Market data for each symbol
        """
        market_data = {}
        
        for symbol in symbols:
            # Attempt to fetch from multiple sources
            symbol_data = self._fetch_from_multiple_sources(
                symbol, 
                asset_type, 
                timeframe, 
                lookback_period
            )
            
            if symbol_data is not None:
                market_data[symbol] = symbol_data
            else:
                self.logger.warning(f"Could not fetch data for {symbol}")
        
        return market_data
    
    def _fetch_from_multiple_sources(
        self, 
        symbol: str, 
        asset_type: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from multiple sources with fallback mechanism
        
        Args:
            symbol (str): Trading symbol
            asset_type (str): Type of asset
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Fetched market data
        """
        sources = self.data_sources.get(asset_type, {})
        
        for source_name, fetch_func in sources.items():
            try:
                data = fetch_func(symbol, timeframe, lookback_period)
                if data is not None and not data.empty:
                    # Enhance data with additional features
                    enhanced_data = self._feature_engineering(data)
                    self.logger.info(f"Successfully fetched data for {symbol} from {source_name}")
                    return enhanced_data
            except Exception as e:
                self.logger.warning(f"Error fetching {symbol} from {source_name}: {e}")
        
        return None
    
    def _fetch_yfinance_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data using yfinance
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Stock market data
        """
        try:
            stock = yf.Ticker(symbol)
            
            # Map timeframe to yfinance interval
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', 
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            interval = interval_map.get(timeframe, '1d')
            
            # Fetch historical data
            data = stock.history(
                period=f'{lookback_period}d', 
                interval=interval
            )
            
            return data
        except Exception as e:
            self.logger.error(f"yfinance data fetch error for {symbol}: {e}")
            return None
    
    def _fetch_alpha_vantage_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Market data
        """
        # Placeholder for Alpha Vantage API implementation
        # Requires API key and specific implementation
        return None
    
    def _fetch_polygon_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Polygon.io
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Market data
        """
        # Placeholder for Polygon.io API implementation
        # Requires API key and specific implementation
        return None
    
    def _fetch_binance_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data from Binance
        
        Args:
            symbol (str): Crypto trading pair
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Crypto market data
        """
        try:
            exchange = ccxt.binance()
            
            # Map timeframe to ccxt format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', 
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            mapped_timeframe = timeframe_map.get(timeframe, '1d')
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, mapped_timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Limit to lookback period
            df = df.last(f'{lookback_period}D')
            
            return df
        except Exception as e:
            self.logger.error(f"Binance data fetch error for {symbol}: {e}")
            return None
    
    def _fetch_coinbase_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data from Coinbase
        
        Args:
            symbol (str): Crypto trading pair
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Crypto market data
        """
        try:
            exchange = ccxt.coinbase()
            
            # Map timeframe to ccxt format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', 
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            mapped_timeframe = timeframe_map.get(timeframe, '1d')
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, mapped_timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Limit to lookback period
            df = df.last(f'{lookback_period}D')
            
            return df
        except Exception as e:
            self.logger.error(f"Coinbase data fetch error for {symbol}: {e}")
            return None
    
    def _fetch_kraken_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data from Kraken
        
        Args:
            symbol (str): Crypto trading pair
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Crypto market data
        """
        try:
            exchange = ccxt.kraken()
            
            # Map timeframe to ccxt format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', 
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            mapped_timeframe = timeframe_map.get(timeframe, '1d')
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, mapped_timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Limit to lookback period
            df = df.last(f'{lookback_period}D')
            
            return df
        except Exception as e:
            self.logger.error(f"Kraken data fetch error for {symbol}: {e}")
            return None
    
    def _fetch_forex_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_period: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch forex data from Exchange Rate API
        
        Args:
            symbol (str): Currency pair
            timeframe (str): Data timeframe
            lookback_period (int): Number of days of historical data
        
        Returns:
            Optional[pd.DataFrame]: Forex market data
        """
        # Placeholder for forex data implementation
        return None
    
    def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform advanced feature engineering on market data
        
        Args:
            data (pd.DataFrame): Raw market data
        
        Returns:
            pd.DataFrame: Enhanced market data
        """
        # Calculate technical indicators
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['SMA_200'] = data['close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        relative_strength = avg_gain / avg_loss
        data['RSI'] = 100.0 - (100.0 / (1.0 + relative_strength))
        
        # Bollinger Bands
        data['BB_Middle'] = data['close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['close'].rolling(window=20).std()
        
        # Percentage Price Oscillator (PPO)
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        data['PPO'] = ((ema_12 - ema_26) / ema_26) * 100
        
        # Volatility indicators
        data['ATR'] = self._average_true_range(data)
        
        return data
    
    def _average_true_range(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            data (pd.DataFrame): Market data
            period (int): Calculation period
        
        Returns:
            pd.Series: Average True Range values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        true_range = pd.Series(
            np.maximum(
                high - low,
                np.abs(high - close.shift()),
                np.abs(low - close.shift())
            )
        )
        
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def correlate_market_data(
        self, 
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate cross-market correlations
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for multiple symbols
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Extract closing prices
        close_prices = {
            symbol: data['close'] for symbol, data in market_data.items()
        }
        
        # Align time series
        price_df = pd.DataFrame(close_prices)
        
        # Calculate correlation matrix
        correlation_matrix = price_df.corr()
        
        return correlation_matrix
    
    def save_market_data(
        self, 
        market_data: Dict[str, pd.DataFrame], 
        output_format: str = 'csv'
    ):
        """
        Save market data to files
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Market data for multiple symbols
            output_format (str): Output file format (csv or parquet)
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.data_dir, 
            f'market_data_{timestamp}'
        )
        os.makedirs(output_path, exist_ok=True)
        
        # Save each symbol's data
        for symbol, data in market_data.items():
            # Sanitize filename
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            
            if output_format == 'csv':
                file_path = os.path.join(output_path, f'{safe_symbol}_data.csv')
                data.to_csv(file_path)
            elif output_format == 'parquet':
                file_path = os.path.join(output_path, f'{safe_symbol}_data.parquet')
                data.to_parquet(file_path)
            
            self.logger.info(f"Saved {symbol} data to {file_path}")

def main():
    # Example usage
    integrator = MarketDataIntegrator()
    
    # Example symbols
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL']
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    # Fetch market data
    market_data = {}
    
    # Fetch stocks
    stocks = integrator.fetch_market_data(
        symbols=stock_symbols, 
        asset_type='stocks', 
        timeframe='1d', 
        lookback_period=365
    )
    market_data.update(stocks)
    
    # Fetch crypto
    cryptos = integrator.fetch_market_data(
        symbols=crypto_symbols, 
        asset_type='crypto', 
        timeframe='1d', 
        lookback_period=365
    )
    market_data.update(cryptos)
    
    # Calculate correlations
    correlations = integrator.correlate_market_data(market_data)
    print("Market Correlations:")
    print(correlations)
    
    # Save market data
    integrator.save_market_data(market_data, output_format='csv')

if __name__ == '__main__':
    main()