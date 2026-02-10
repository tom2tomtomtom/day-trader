# APEX TRADER — Mission Document

## The Mission

Build an autonomous ML-driven trading system with one objective: **maximize profit**.

The system trades stocks and crypto — penny stocks, meme coins, volatile momentum plays — wherever the biggest opportunities live. It starts aggressive, takes calculated risks, and uses machine learning to learn which risks pay off. Over time, the ML compresses risk while maintaining high returns.

This is not a conservative portfolio manager. This is a profit-hunting machine that gets smarter every day.

## Philosophy

1. **Profit is the only metric that matters.** Everything else (Sharpe, win rate, drawdown) is diagnostic — useful for learning, not for constraining.

2. **Start aggressive, let ML tighten.** We don't impose conservative limits upfront. The system discovers which risks are worth taking by taking them and measuring outcomes.

3. **Every trade is a data point.** Win or lose, every trade feeds the ML pipeline. Market conditions, entry timing, position size, hold duration, exit reason — all captured, all learned from.

4. **The system earns trust through performance.** Paper trading phase proves the system works. Real money follows demonstrated results.

## Phases

### Phase 1: Paper Trading Experiment (Current)
- $100,000 virtual portfolio
- Fully autonomous — finds opportunities, sizes positions, executes trades
- Aggressive positioning — big swings on high-conviction signals
- ML learning from every trade outcome
- Dashboard for monitoring, alerts for high-conviction plays
- **Success criteria**: Demonstrate the system can find and exploit profitable patterns

### Phase 2: Live Trading
- Graduate to real money after paper trading proves the model
- Start with a defined bankroll
- Same autonomous operation with real execution via Alpaca (stocks) + exchange APIs (crypto)
- ML continues learning from real market feedback

## What We Trade

- **Stocks**: Penny stocks, momentum plays, gap trades, squeeze candidates
- **Crypto**: Meme coins (DOGE, SHIB, PEPE, etc.), volatile altcoins, momentum tokens
- **Strategy**: Go where the volatility is. The system hunts for outsized moves and learns to ride them.

## How It Learns

The ML pipeline tracks every trade with full context:

- **Entry conditions**: Technical signals, sentiment, regime, news catalysts
- **Position management**: Size, leverage, stop/target placement
- **Outcome**: P&L, hold duration, max adverse excursion, exit reason
- **Market context**: Regime at entry, VIX, sector momentum, correlation state

Over time, the system builds a model of: *given these conditions, what position size and strategy maximizes expected profit?*

Features the ML evaluates:
- Technical indicators (RSI, MACD, BB, volume profile)
- Sentiment signals (Fear/Greed, social media hype, news sentiment)
- Insider/institutional flow (congressional trades, insider buying clusters)
- Market regime (trending/ranging/volatile/crisis)
- Cross-asset correlations
- Time-of-day and day-of-week patterns
- Catalyst proximity (earnings, FOMC, token unlocks)

## Architecture

### Core Engine (Python)
- **Orchestrator**: Coordinates all modules, runs on schedule
- **Trading Model**: Signal generation with ML-enhanced scoring
- **Risk Engine**: ML-driven position sizing (not static rules)
- **Regime Engine**: Market state detection
- **Paper Trader → Live Trader**: Execution layer
- **ML Pipeline**: Feature engineering, model training, outcome tracking
- **Intelligence Modules**: Phantom Council (AI debate), sentiment, macro analysis

### Data Layer (Supabase/PostgreSQL)
- Trade log with full context (every trade, every feature, every outcome)
- Market data cache
- ML model state and feature store
- Portfolio state and positions
- Watchlist and opportunity scores

### Dashboard (Next.js on Railway)
- Live P&L and positions
- Trade history with ML confidence scores
- Opportunity scanner
- Regime and sentiment gauges
- ML model performance over time
- Alert configuration

### Automation (Railway)
- Scheduled scans (market hours for stocks, 24/7 for crypto)
- Autonomous trade execution
- ML model retraining on new data
- Alert dispatch

## APIs & Data Sources

| Service | Purpose | Status |
|---------|---------|--------|
| Yahoo Finance | Price data, historical | Free, working |
| Finnhub | Insider trades, news sentiment, quotes | API key available |
| Claude (Anthropic) | AI analysis, Phantom Council, narratives | API key available |
| Perplexity | Real-time market research, news | API key available |
| Supabase | Database, real-time subscriptions | Set up |
| Alpaca | Stock paper/live trading | Needs setup (Phase 2) |
| CCXT/Exchanges | Crypto trading | Needs setup (Phase 2) |

## Tech Stack

- **Backend**: Python 3.12+ (trading engine, ML pipeline)
- **Database**: Supabase (PostgreSQL + real-time)
- **Dashboard**: Next.js + React + TailwindCSS
- **Deployment**: Railway (backend services + dashboard)
- **ML**: scikit-learn, potentially PyTorch for deeper models
- **APIs**: yfinance, finnhub, anthropic, ccxt

## Non-Goals

- This is NOT a conservative long-term portfolio manager
- This is NOT a buy-and-hold system
- We do NOT impose arbitrary risk limits that prevent the system from learning
- We do NOT need forex (stocks + crypto only)
- We do NOT need the Streamlit dashboard (Next.js only)
