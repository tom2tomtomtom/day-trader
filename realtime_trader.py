#!/usr/bin/env python3
"""
REAL-TIME MARKET PREDATOR
Adaptive Machine Learning Trading System
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime, timedelta
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[
                        logging.FileHandler("/root/clawd/day-trader/trader.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

class RealTimePredator:
    def __init__(self, 
                 initial_capital=250000, 
                 risk_tolerance=0.05,
                 database_path='/root/clawd/day-trader/trading_data.db'):
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_tolerance = risk_tolerance
        
        # Core trading universe
        self.trading_universe = [
            'SPY',    # S&P 500 ETF
            'QQQ',    # Nasdaq 100 ETF
            'IWM',    # Russell 2000 ETF
            'XLK',    # Technology Sector
            'XLF',    # Financial Sector
            'XLE',    # Energy Sector
            'GLD',    # Gold
            'USO'     # Oil
        ]
        
        # Machine Learning Models
        self.price_predictor = None
        self.signal_classifier = None
        
        # Database for tracking
        self.database_path = database_path
        self._initialize_database()
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
    def _initialize_database(self):
        """Create SQLite database for tracking trades and performance"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT,
            entry_time DATETIME,
            exit_time DATETIME,
            entry_price REAL,
            exit_price REAL,
            shares REAL,
            profit_loss REAL,
            strategy_score REAL
        )
        ''')
        
        # Market data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp DATETIME,
            symbol TEXT,
            price REAL,
            volume REAL,
            volatility REAL,
            PRIMARY KEY (timestamp, symbol)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _fetch_historical_data(self, symbol, period='5y', interval='1d'):
        """Fetch comprehensive historical data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Advanced feature engineering
            df['returns'] = df['Close'].pct_change()
            df['volatility'] = df['Close'].rolling(window=20).std()
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['ma_50'] = df['Close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            return df.dropna()
        except Exception as e:
            logging.error(f"Data fetch error for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self):
        """Prepare comprehensive training dataset"""
        logging.info("Preparing training data...")
        X_total = []
        y_price_total = []
        y_signal_total = []
        
        for symbol in self.trading_universe:
            df = self._fetch_historical_data(symbol)
            if df is None or len(df) < 100:
                continue
            
            # Features
            features = df[['returns', 'volatility', 'ma_20', 'ma_50', 'rsi', 'Volume']]
            
            # Remove NaNs
            features = features.dropna()
            
            # Price prediction target
            future_price = df['Close'].shift(-5)  # 5-day forward price
            
            # Align future price with features
            future_price = future_price.loc[features.index]
            
            # Signal classification
            signals = np.zeros(len(features))
            returns = features['returns']
            signals[returns > returns.quantile(0.7)] = 2  # Strong Buy
            signals[returns < returns.quantile(0.3)] = -2  # Strong Sell
            signals[(returns > 0) & (returns <= returns.quantile(0.7))] = 1  # Buy
            signals[(returns < 0) & (returns >= returns.quantile(0.3))] = -1  # Sell
            
            X_total.append(features.values)
            y_price_total.append(future_price.values)
            y_signal_total.append(signals)
        
        # Combine data from all symbols
        X = np.vstack(X_total)
        y_price = np.concatenate(y_price_total)
        y_signal = np.concatenate(y_signal_total)
        
        # Impute missing values
        X = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Remove NaN from targets
        valid_indices = ~np.isnan(y_price)
        X_scaled = X_scaled[valid_indices]
        y_price = y_price[valid_indices]
        y_signal = y_signal[valid_indices]
        
        # Split data
        X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
            X_scaled, y_price, test_size=0.2, random_state=42
        )
        
        X_train_signal, X_test_signal, y_train_signal, y_test_signal = train_test_split(
            X_scaled, y_signal, test_size=0.2, random_state=42
        )
        
        # Train price predictor
        self.price_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.price_predictor.fit(X_train_price, y_train_price)
        
        # Train signal classifier
        self.signal_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.signal_classifier.fit(X_train_signal, y_train_signal)
        
        # Evaluation
        price_score = self.price_predictor.score(X_test_price, y_test_price)
        signal_score = self.signal_classifier.score(X_test_signal, y_test_signal)
        
        logging.info(f"Price Prediction Score: {price_score:.2%}")
        logging.info(f"Signal Classification Score: {signal_score:.2%}")
        
        # Save models
        joblib.dump(self.price_predictor, '/root/clawd/day-trader/price_predictor.joblib')
        joblib.dump(self.signal_classifier, '/root/clawd/day-trader/signal_classifier.joblib')
    
    def predict_trade_opportunities(self):
        """Generate real-time trade opportunities"""
        if self.price_predictor is None or self.signal_classifier is None:
            logging.warning("Models not trained. Call prepare_training_data() first.")
            return []
        
        trade_opportunities = []
        
        for symbol in self.trading_universe:
            # Fetch latest data
            df = self._fetch_historical_data(symbol, period='3mo', interval='1d')
            if df is None:
                continue
            
            # Latest features
            latest_features = df.iloc[-1][['returns', 'volatility', 'ma_20', 'ma_50', 'rsi', 'Volume']]
            latest_features_scaled = self.scaler.transform(
                self.imputer.transform([latest_features])
            )
            
            # Predictions
            predicted_price = self.price_predictor.predict(latest_features_scaled)[0]
            predicted_signal = self.signal_classifier.predict(latest_features_scaled)[0]
            
            # Current market price
            current_price = df['Close'].iloc[-1]
            
            # Trade opportunity assessment
            opportunity = {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'signal': predicted_signal,
                'price_difference_pct': ((predicted_price - current_price) / current_price) * 100
            }
            
            trade_opportunities.append(opportunity)
        
        # Sort opportunities by potential
        trade_opportunities.sort(key=lambda x: abs(x['price_difference_pct']), reverse=True)
        
        return trade_opportunities
    
    def execute_trades(self, opportunities):
        """Execute trades based on opportunities"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        for opportunity in opportunities[:3]:  # Top 3 opportunities
            if opportunity['signal'] in [2, 1]:  # Buy signals
                shares = self._calculate_position_size(opportunity['current_price'])
                trade_value = shares * opportunity['current_price']
                
                # Record trade
                cursor.execute('''
                    INSERT INTO trades 
                    (symbol, entry_time, entry_price, shares, strategy_score) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    opportunity['symbol'], 
                    datetime.now(), 
                    opportunity['current_price'], 
                    shares,
                    abs(opportunity['price_difference_pct'])
                ))
            
            elif opportunity['signal'] in [-2, -1]:  # Sell signals
                # Sell existing positions logic would go here
                pass
        
        conn.commit()
        conn.close()
    
    def _calculate_position_size(self, current_price):
        """Dynamically calculate position size"""
        max_risk_per_trade = self.current_capital * self.risk_tolerance
        shares = max_risk_per_trade / current_price
        return int(shares)
    
    def run_trading_cycle(self):
        """Complete trading cycle"""
        logging.info("🚀 TRADING CYCLE INITIATED")
        
        # Ensure models are trained
        if not os.path.exists('/root/clawd/day-trader/price_predictor.joblib'):
            self.prepare_training_data()
        else:
            # Load existing models
            self.price_predictor = joblib.load('/root/clawd/day-trader/price_predictor.joblib')
            self.signal_classifier = joblib.load('/root/clawd/day-trader/signal_classifier.joblib')
        
        # Find trade opportunities
        opportunities = self.predict_trade_opportunities()
        
        # Execute trades
        self.execute_trades(opportunities)
        
        logging.info("Trading Cycle Completed")
        return opportunities

def main():
    predator = RealTimePredator()
    opportunities = predator.run_trading_cycle()
    
    print("\n🐺 TRADING OPPORTUNITIES 🐺")
    for opp in opportunities:
        print(f"{opp['symbol']}: Signal {opp['signal']} | "
              f"Current: ${opp['current_price']:.2f} | "
              f"Predicted: ${opp['predicted_price']:.2f} | "
              f"Potential: {opp['price_difference_pct']:+.2f}%")

if __name__ == "__main__":
    main()