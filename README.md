# Day Trader 📈

An AI-powered day trading simulator with global market support (crypto, forex, stocks).

## Features

- **24/7 Global Markets**: Crypto (BTC, ETH, SOL, etc.), Forex (EUR/USD, GBP/USD, etc.), and US stocks/ETFs
- **Smart Scanner**: Finds gaps, volume surges, breakouts, mean reversion, and volatility plays
- **Paper Trading**: Simulated $100k account with stop-losses and take-profits
- **Learning System**: Adjusts indicator weights based on trade outcomes
- **Regime Detection**: Identifies market conditions (trending, volatile, ranging)
- **Next.js Dashboard**: Real-time web UI for monitoring

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install yfinance numpy pandas

# Check market status
python scanner.py markets

# Run a global scan (crypto + forex + stocks)
python scanner.py scan --global

# Run trading cycle
python day_trader.py run --global

# Check status
python day_trader.py status
```

## Commands

### Scanner
```bash
python scanner.py markets          # Show which markets are open
python scanner.py scan             # Scan tradeable assets only
python scanner.py scan --global    # Scan all assets (24/7)
python scanner.py watchlist        # Show current watchlist
```

### Day Trader
```bash
python day_trader.py run           # Run one trading cycle
python day_trader.py run --global  # Include global markets
python day_trader.py status        # Daily P&L summary
python day_trader.py close         # Close all positions
python day_trader.py reset         # Reset to $100k
```

### Regime Detector
```bash
python regime_detector.py          # Analyze current market regime
python hourly_runner.py            # Full hourly check with regime + trades
```

## Dashboard

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

## Market Hours

| Market | Hours (UTC) | Days |
|--------|-------------|------|
| Crypto | 24/7 | Every day |
| Forex | Sun 22:00 - Fri 22:00 | 24/5 |
| US Stocks | 14:30 - 21:00 | Mon-Fri |
| US Pre-market | 09:00 - 14:30 | Mon-Fri |

## Architecture

```
├── scanner.py          # Finds trading opportunities
├── day_trader.py       # Executes paper trades
├── regime_detector.py  # Market regime analysis
├── hourly_runner.py    # Automated hourly checks
├── learner.py          # Pattern learning from outcomes
├── paper_trader.py     # Alternative paper trading logic
├── sentiment.py        # News/sentiment analysis
├── dashboard/          # Next.js web UI
│   ├── src/app/        # Pages (positions, watchlist, markets)
│   └── src/components/ # React components
└── *.json              # State files (positions, watchlist, etc.)
```

## Cron Integration

Works with Clawdbot cron for automated trading:
- Every 15 min: Trading cycle with global markets
- Every 4 hours: Full scanner refresh
- Daily: Trade intel analysis
- EOD: Position close and P&L report

## License

MIT
