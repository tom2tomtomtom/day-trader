# Trading Intelligence System

## Overview
A comprehensive, self-learning trading intelligence platform designed to generate sophisticated trading insights without relying on external API keys.

## Key Features
- 🧠 Advanced Machine Learning
  - Multiple predictive models (Random Forest, Gradient Boosting, Neural Networks)
  - Self-generating synthetic market data
  - Dynamic model training and evaluation

- 📊 Multi-Strategy Ensemble
  - Momentum trading
  - Mean reversion
  - Trend following
  - Volatility breakout strategies

- 🛡️ Advanced Risk Management
  - Dynamic position sizing
  - Comprehensive risk metrics
  - Performance tracking

- 📈 Intelligent Dashboard
  - Real-time performance visualization
  - Model evaluation insights
  - Risk analysis tools

## Components
1. `trading_intelligence_core.py`: Core trading logic and ML models
2. `trading_dashboard/app.py`: Interactive Streamlit dashboard
3. `data/`: Directory for storing trading results and models

## Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r trading_dashboard/requirements.txt

# Run trading intelligence
python trading_intelligence_core.py

# Launch dashboard
streamlit run trading_dashboard/app.py
```

## Usage
1. Generate synthetic market data
2. (Optional) Add API Keys
3. Train machine learning models
4. Backtest trading strategies
5. Visualize results in the dashboard

## API Key Management
The system supports multiple data providers:
- Finnhub
- Alpha Vantage
- Polygon
- Yahoo Finance
- Tiingo
- Marketstack
- Twelve Data
- Financial Modeling Prep

### Adding API Keys
1. Visit provider websites to get free API keys
2. Use the dashboard's API Key Management sidebar
3. Enter keys securely
4. System will automatically use available keys to enhance data

### Free API Key Resources
- [Finnhub](https://finnhub.io/) - Free tier available
- [Alpha Vantage](https://www.alphavantage.co/) - Free API key
- [Polygon](https://polygon.io/) - Free tier for developers

## Disclaimer
This is a research and educational tool. Not financial advice. Always consult professional financial advisors.

### Security Notes
- Keys are stored securely in an encrypted JSON file
- Only basic keys are accepted (minimum length validation)
- No sensitive data is transmitted without your explicit consent