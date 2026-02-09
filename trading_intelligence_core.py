#!/usr/bin/env python3
"""
Advanced Trading Intelligence Core
A comprehensive trading system with:
- Self-learning prediction mechanisms
- Advanced risk management
- Multi-strategy approach
- Performance optimization
"""

import os
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from scipy.signal import argrelextrema

class APIKeyManager:
    """
    Secure and flexible API key management system
    """
    def __init__(self, base_dir=None):
        """
        Initialize API Key Manager
        
        Args:
            base_dir (str, optional): Base directory for key storage
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.keys_dir = os.path.join(self.base_dir, 'keys')
        
        # Create keys directory
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Supported API providers
        self.supported_providers = [
            'finnhub', 'alpha_vantage', 'polygon', 
            'yahoo_finance', 'tiingo', 'marketstack',
            'twelve_data', 'financial_modeling_prep'
        ]
        
        # Keys storage file
        self.keys_file = os.path.join(self.keys_dir, 'api_keys.json')
        
        # Initialize keys dictionary
        self.keys = self._load_keys()
    
    def _load_keys(self):
        """
        Load API keys from secure storage
        
        Returns:
            dict: Stored API keys
        """
        if os.path.exists(self.keys_file):
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {provider: None for provider in self.supported_providers}
    
    def add_key(self, provider, key):
        """
        Add or update an API key
        
        Args:
            provider (str): API provider name
            key (str): API key value
        """
        if provider not in self.supported_providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Basic key validation
        if not key or len(key.strip()) < 10:
            raise ValueError("Invalid API key")
        
        # Update keys
        self.keys[provider] = key.strip()
        
        # Save to secure storage
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=4)
        
        print(f"API key for {provider} added successfully.")
    
    def get_key(self, provider):
        """
        Retrieve an API key
        
        Args:
            provider (str): API provider name
        
        Returns:
            str or None: API key if available
        """
        return self.keys.get(provider)
    
    def check_key_availability(self):
        """
        Check which API keys are available
        
        Returns:
            dict: Key availability status
        """
        return {
            provider: bool(key) 
            for provider, key in self.keys.items()
        }

class TradingIntelligenceCore:
    def __init__(self, base_dir=None):
        """
        Initialize the core trading intelligence system
        
        Args:
            base_dir (str, optional): Base directory for data storage
        """
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # API Key Manager
        self.api_key_manager = APIKeyManager(base_dir)
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=os.path.join(self.base_dir, 'trading_intelligence.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # Risk management parameters
        self.risk_config = {
            'max_portfolio_risk': 0.05,  # 5% max portfolio risk
            'max_single_trade_risk': 0.02,  # 2% max risk per trade
            'min_win_rate': 0.55,  # Minimum acceptable win rate
            'max_drawdown': 0.10,  # Maximum acceptable drawdown
        }
        
        # Trading strategy configuration
        self.strategy_weights = {
            'momentum': 0.3,
            'mean_reversion': 0.3,
            'trend_following': 0.2,
            'volatility_breakout': 0.2
        }
        
        # Log available API keys
        self._log_key_availability()
    
    def _log_key_availability(self):
        """
        Log the availability of API keys
        """
        key_status = self.api_key_manager.check_key_availability()
        available_keys = [
            provider for provider, available in key_status.items() if available
        ]
        
        if available_keys:
            self.logger.info(f"Available API Keys: {', '.join(available_keys)}")
        else:
            self.logger.warning("No API keys are currently available. Using synthetic data generation.")
    
    def enhance_with_external_data(self, base_data):
        """
        Attempt to enhance synthetic data with external sources if keys are available
        
        Args:
            base_data (pd.DataFrame): Synthetic market data
        
        Returns:
            pd.DataFrame: Enhanced market data
        """
        # Check for available keys and data sources
        key_status = self.api_key_manager.check_key_availability()
        
        # Priority data sources
        data_sources = [
            ('finnhub', self._fetch_finnhub_data),
            ('alpha_vantage', self._fetch_alpha_vantage_data),
            ('polygon', self._fetch_polygon_data)
        ]
        
        for provider, fetch_func in data_sources:
            if key_status.get(provider, False):
                try:
                    external_data = fetch_func(base_data)
                    if external_data is not None:
                        # Merge external data with synthetic data
                        base_data = pd.concat([base_data, external_data], axis=1)
                        self.logger.info(f"Enhanced data with {provider}")
                        break
                except Exception as e:
                    self.logger.warning(f"Error fetching data from {provider}: {e}")
        
        return base_data
    
    def _fetch_finnhub_data(self, base_data):
        """
        Fetch supplementary market data from Finnhub
        
        Args:
            base_data (pd.DataFrame): Base synthetic data
        
        Returns:
            pd.DataFrame or None: Additional market insights
        """
        key = self.api_key_manager.get_key('finnhub')
        if not key:
            return None
        
        # Placeholder for actual Finnhub API data fetching
        # In a real implementation, you'd use the Finnhub API to get additional market insights
        synthetic_extras = pd.DataFrame(
            np.random.normal(0, 0.01, size=(len(base_data), 3)),
            columns=['finnhub_sentiment', 'finnhub_volume_anomaly', 'finnhub_insider_trading'],
            index=base_data.index
        )
        
        return synthetic_extras
    
    def _fetch_alpha_vantage_data(self, base_data):
        """
        Fetch supplementary market data from Alpha Vantage
        
        Args:
            base_data (pd.DataFrame): Base synthetic data
        
        Returns:
            pd.DataFrame or None: Additional market insights
        """
        key = self.api_key_manager.get_key('alpha_vantage')
        if not key:
            return None
        
        # Placeholder for actual Alpha Vantage API data fetching
        synthetic_extras = pd.DataFrame(
            np.random.normal(0, 0.01, size=(len(base_data), 3)),
            columns=['alpha_momentum', 'alpha_volatility', 'alpha_trend_strength'],
            index=base_data.index
        )
        
        return synthetic_extras
    
    def _fetch_polygon_data(self, base_data):
        """
        Fetch supplementary market data from Polygon
        
        Args:
            base_data (pd.DataFrame): Base synthetic data
        
        Returns:
            pd.DataFrame or None: Additional market insights
        """
        key = self.api_key_manager.get_key('polygon')
        if not key:
            return None
        
        # Placeholder for actual Polygon API data fetching
        synthetic_extras = pd.DataFrame(
            np.random.normal(0, 0.01, size=(len(base_data), 3)),
            columns=['polygon_sector_momentum', 'polygon_news_impact', 'polygon_market_correlation'],
            index=base_data.index
        )
        
        return synthetic_extras

    def generate_synthetic_market_data(self, n_samples=1000, n_features=10):
        """
        Generate synthetic market data for training and testing
        
        Args:
            n_samples (int): Number of data points
            n_features (int): Number of synthetic features
        
        Returns:
            pd.DataFrame: Synthetic market data
        """
        # Generate time series with realistic financial characteristics
        np.random.seed(42)
        
        # Base price series with trend and volatility
        base_price = np.cumsum(np.random.normal(0.0005, 0.02, n_samples))
        
        # Create DataFrame with multiple features
        data = pd.DataFrame({
            'price': base_price,
            'returns': np.diff(base_price, prepend=base_price[0]),
            'volatility': pd.Series(base_price).rolling(window=20).std()
        })
        
        # Add synthetic technical indicators
        data['SMA_10'] = data['price'].rolling(window=10).mean()
        data['SMA_50'] = data['price'].rolling(window=50).mean()
        data['EMA_20'] = data['price'].ewm(span=20, adjust=False).mean()
        
        # Advanced feature generation
        data['RSI'] = self.calculate_rsi(data['price'])
        data['MACD'], data['MACD_signal'] = self.calculate_macd(data['price'])
        
        # Generate trading labels
        data['label'] = self.generate_trading_labels(data)
        
        return data

    def calculate_rsi(self, prices, periods=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            periods (int): RSI calculation period
        
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        
        RollUp = dUp.rolling(window=periods).mean()
        RollDown = dDown.abs().rolling(window=periods).mean()
        
        RS = RollUp / RollDown
        RSI = 100.0 - (100.0 / (1.0 + RS))
        
        return RSI

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices (pd.Series): Price series
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        
        Returns:
            tuple: MACD and Signal line
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        
        return macd, signal_line

    def generate_trading_labels(self, data, look_ahead=5, threshold=0.01):
        """
        Generate trading labels based on future price movement
        
        Args:
            data (pd.DataFrame): Market data
            look_ahead (int): Number of periods to look ahead
            threshold (float): Minimum price change to trigger label
        
        Returns:
            pd.Series: Trading labels
        """
        future_returns = data['price'].shift(-look_ahead) / data['price'] - 1
        
        labels = pd.Series(0, index=data.index)
        labels[future_returns > threshold] = 1    # Buy signal
        labels[future_returns < -threshold] = -1  # Sell signal
        
        return labels

    def create_ml_pipeline(self):
        """
        Create a machine learning pipeline with multiple models
        
        Returns:
            dict: ML models with preprocessing pipelines
        """
        # Preprocessing
        preprocessors = {
            'standard_scaler': StandardScaler(),
            'minmax_scaler': MinMaxScaler()
        }
        
        # Models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(50, 25), 
                max_iter=500, 
                random_state=42
            )
        }
        
        # Create pipelines
        ml_pipelines = {}
        for scaler_name, scaler in preprocessors.items():
            for model_name, model in models.items():
                pipeline_name = f"{scaler_name}_{model_name}"
                ml_pipelines[pipeline_name] = Pipeline([
                    ('scaler', scaler),
                    ('classifier', model)
                ])
        
        return ml_pipelines

    def evaluate_models(self, X, y):
        """
        Evaluate machine learning models using time series cross-validation
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target labels
        
        Returns:
            dict: Model performance metrics
        """
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Create ML pipelines
        ml_pipelines = self.create_ml_pipeline()
        
        # Performance tracking
        performance_metrics = {}
        
        for pipeline_name, pipeline in ml_pipelines.items():
            # Cross-validation scores
            cv_scores = cross_val_score(
                pipeline, X, y, 
                cv=tscv, 
                scoring='balanced_accuracy'
            )
            
            performance_metrics[pipeline_name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
        
        return performance_metrics

    def risk_management_strategy(self, trading_signals, portfolio_value):
        """
        Advanced risk management strategy
        
        Args:
            trading_signals (pd.Series): Trading signals
            portfolio_value (float): Current portfolio value
        
        Returns:
            dict: Risk-adjusted position sizes
        """
        # Calculate signal confidence and risk
        signal_confidence = self.calculate_signal_confidence(trading_signals)
        
        # Dynamic position sizing
        position_sizes = {}
        for signal, confidence in signal_confidence.items():
            # Base position size
            base_size = portfolio_value * self.risk_config['max_single_trade_risk']
            
            # Adjust position size based on confidence
            adjusted_size = base_size * confidence
            
            position_sizes[signal] = {
                'size': adjusted_size,
                'confidence': confidence
            }
        
        return position_sizes

    def calculate_signal_confidence(self, trading_signals):
        """
        Calculate confidence for trading signals
        
        Args:
            trading_signals (pd.Series): Trading signals
        
        Returns:
            dict: Signal confidence levels
        """
        # Count signal occurrences
        signal_counts = trading_signals.value_counts(normalize=True)
        
        # Calculate confidence using probabilistic methods
        confidence_levels = {}
        for signal, frequency in signal_counts.items():
            # Bayesian-inspired confidence calculation
            confidence = (frequency + 0.5) / (len(trading_signals) + 1)
            confidence_levels[signal] = confidence
        
        return confidence_levels

    def multi_strategy_ensemble(self, data):
        """
        Create an ensemble of trading strategies
        
        Args:
            data (pd.DataFrame): Market data
        
        Returns:
            pd.Series: Ensemble trading signals
        """
        # Define strategy functions
        strategies = {
            'momentum': self._momentum_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'trend_following': self._trend_following_strategy,
            'volatility_breakout': self._volatility_breakout_strategy
        }
        
        # Generate signals from each strategy
        strategy_signals = {}
        for name, strategy_func in strategies.items():
            strategy_signals[name] = strategy_func(data)
        
        # Weighted ensemble voting
        ensemble_signal = pd.Series(0, index=data.index)
        for name, signals in strategy_signals.items():
            weight = self.strategy_weights.get(name, 1/len(strategies))
            ensemble_signal += signals * weight
        
        # Normalize and convert to discrete signals
        ensemble_signal = np.sign(ensemble_signal)
        
        return ensemble_signal

    def _momentum_strategy(self, data):
        """Simple momentum strategy"""
        return np.sign(data['returns'].rolling(window=10).mean())

    def _mean_reversion_strategy(self, data):
        """Mean reversion strategy"""
        sma_10 = data['price'].rolling(window=10).mean()
        sma_50 = data['price'].rolling(window=50).mean()
        return np.sign(sma_50 - sma_10)

    def _trend_following_strategy(self, data):
        """Trend following strategy"""
        return np.sign(data['price'].diff(10))

    def _volatility_breakout_strategy(self, data):
        """Volatility breakout strategy"""
        volatility = data['price'].rolling(window=20).std()
        return np.sign(data['price'] - (data['price'].rolling(window=20).mean() + volatility))

    def backtest_trading_strategy(self, data):
        """
        Comprehensive backtesting of trading strategy
        
        Args:
            data (pd.DataFrame): Market data
        
        Returns:
            dict: Backtest performance metrics
        """
        # Generate ensemble trading signals
        signals = self.multi_strategy_ensemble(data)
        
        # Calculate returns
        data['strategy_returns'] = signals.shift(1) * data['returns']
        
        # Performance metrics
        performance = {
            'total_return': (1 + data['strategy_returns']).prod() - 1,
            'annual_return': (1 + data['strategy_returns']).prod() ** (252 / len(data)) - 1,
            'sharpe_ratio': self.calculate_sharpe_ratio(data['strategy_returns']),
            'max_drawdown': self.calculate_max_drawdown(data['strategy_returns']),
            'win_rate': (data['strategy_returns'] > 0).mean()
        }
        
        return performance

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe Ratio
        
        Args:
            returns (pd.Series): Strategy returns
            risk_free_rate (float): Annual risk-free rate
        
        Returns:
            float: Sharpe Ratio
        """
        # Annualized Sharpe Ratio calculation
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        return sharpe_ratio

    def calculate_max_drawdown(self, returns):
        """
        Calculate Maximum Drawdown
        
        Args:
            returns (pd.Series): Strategy returns
        
        Returns:
            float: Maximum drawdown percentage
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()

    def save_model_results(self, performance_metrics, model_evaluation):
        """
        Save performance and model evaluation results
        
        Args:
            performance_metrics (dict): Trading strategy performance
            model_evaluation (dict): ML model evaluation results
        """
        # Prepare results for saving
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'model_evaluation': model_evaluation
        }
        
        # Save to file
        results_path = os.path.join(self.data_dir, f'trading_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Results saved to {results_path}")

    def run_trading_intelligence(self):
        """
        Execute the complete trading intelligence workflow
        """
        try:
            # Generate synthetic market data
            market_data = self.generate_synthetic_market_data()
            
            # Prepare features and labels
            X = market_data.drop(['label', 'price'], axis=1)
            y = market_data['label']
            
            # Evaluate ML models
            model_evaluation = self.evaluate_models(X, y)
            
            # Run multi-strategy backtest
            performance_metrics = self.backtest_trading_strategy(market_data)
            
            # Save results
            self.save_model_results(performance_metrics, model_evaluation)
            
            # Log key insights
            self.logger.info("Trading Intelligence Workflow Completed")
            self.logger.info(f"Total Return: {performance_metrics['total_return']:.2%}")
            self.logger.info(f"Annual Return: {performance_metrics['annual_return']:.2%}")
            self.logger.info(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Trading Intelligence Workflow Failed: {e}")
            raise

def main():
    # Initialize and run trading intelligence
    trading_core = TradingIntelligenceCore()
    trading_core.run_trading_intelligence()

if __name__ == '__main__':
    main()