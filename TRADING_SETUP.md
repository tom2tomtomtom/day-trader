# Real Market Trading Intelligence System Setup Guide

## Prerequisites
- Python 3.9+
- Trading accounts with supported platforms
- API keys for trading platforms

## Supported Platforms
1. Stock Trading
   - Alpaca (Recommended)
   - Interactive Brokers
   - TD Ameritrade

2. Cryptocurrency Exchanges
   - Binance
   - Coinbase
   - Kraken
   - ByBit
   - OKX
   - KuCoin

## API Key Setup

### Stock Trading (Alpaca)
1. Create an Alpaca account
2. Generate API keys
3. Use format: `API_KEY,SECRET_KEY`

### Cryptocurrency Exchanges
1. Create accounts on desired exchanges
2. Generate API keys with trading permissions
3. Use format: `API_KEY,SECRET_KEY`

## Configuration Steps

### 1. Install Dependencies
```bash
python3 -m venv trading_env
source trading_env/bin/activate
pip install -r requirements.txt
```

### 2. Configure Watchlist
Edit `config/watchlist.json`:
```json
{
    "stocks": [
        {"symbol": "AAPL", "allocation": 0.2},
        {"symbol": "MSFT", "allocation": 0.2}
    ],
    "crypto": [
        {"symbol": "BTC/USDT", "allocation": 0.3},
        {"symbol": "ETH/USDT", "allocation": 0.2}
    ]
}
```

### 3. Add API Keys
Use the trading dashboard or CLI to add keys:
```bash
python -m trading_intelligence_core.api_key_manager
```

## Risk Management
- Max Portfolio Risk: 5%
- Max Single Trade Risk: 2%
- Stop Loss: 3%
- Take Profit: 5%

## Running the Trading System
```bash
# Start continuous trading
python real_market_trader.py
```

## Monitoring
- Check `logs/real_market_trader.log` for trading activities
- Use dashboard for performance tracking

## Disclaimer
- This is an experimental trading system
- Always use paper trading first
- Never risk more than you can afford to lose
- Consult financial advisors

## Performance Optimization
- Regularly retrain ML models
- Monitor and adjust risk parameters
- Diversify across different assets

## Security Notes
- Never share API keys
- Use environment variables or secure key management
- Implement additional security measures