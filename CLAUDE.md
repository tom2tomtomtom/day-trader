# CLAUDE.md — Apex Trader

## Project Overview

Autonomous ML-driven trading system. Trades stocks and crypto (penny stocks, meme coins, volatile plays) with one goal: maximize profit. Paper trading phase first, then real money.

Read MISSION.md for full context.

## Architecture

```
day-trader/
├── core/                    # Core trading engine (Python)
│   ├── orchestrator.py      # Main brain — coordinates everything
│   ├── trading_model.py     # Technical signal scoring (-100 to +100)
│   ├── backtester.py        # Historical strategy testing
│   ├── paper_trader.py      # Virtual portfolio execution
│   ├── risk_engine.py       # ML-driven position sizing
│   ├── regime_engine.py     # Market state detection
│   ├── signal_ensemble.py   # Multi-signal fusion
│   ├── data_layer.py        # Data aggregation (Yahoo, Finnhub, etc.)
│   ├── intelligence_pipeline.py  # Coordinates intelligence modules
│   ├── phantom_council.py   # AI investor personas debate trades
│   ├── congressional_intel.py    # Congressional trade tracking
│   ├── macro_intel.py       # Macro trigger detection
│   ├── opportunity_scorer.py     # Multi-factor conviction scoring
│   └── trade_narrator.py    # AI-powered trade narratives
├── dashboard/               # Next.js web dashboard
├── trading_dashboard/       # Streamlit dashboard (legacy, keep for reference)
├── MISSION.md              # Mission document — read this first
└── requirements.txt        # Python dependencies
```

## Key Principles

1. **core/ is canonical.** All trading logic lives in the core/ package. Root-level .py scripts are legacy standalone experiments — reference them for ideas but don't build on them.

2. **Profit is the mission.** The system is aggressive by design. Don't impose conservative limits that prevent learning. Let the ML discover optimal risk levels.

3. **Every trade is training data.** Log everything: entry conditions, features, position details, outcome, market context. The ML pipeline needs rich data.

4. **Supabase is the database.** No more JSON files for state. All trades, positions, signals, and ML features go to PostgreSQL via Supabase.

5. **Dashboard is Next.js.** The dashboard/ directory is the web UI. Ignore trading_dashboard/ (Streamlit legacy).

## Database (Supabase)

- **Project URL**: https://kvvmrbftegiclspxxcay.supabase.co
- Tables: trades, positions, signals, market_data, ml_features, ml_models, watchlist, alerts
- Use Supabase client libraries (Python for backend, JS for dashboard)

## Environment Variables

Required in `.env` (never commit):
```
SUPABASE_URL=https://kvvmrbftegiclspxxcay.supabase.co
SUPABASE_ANON_KEY=<anon key>
SUPABASE_SERVICE_ROLE_KEY=<service role key>
ANTHROPIC_API_KEY=<claude api key>
FINNHUB_API_KEY=<finnhub key>
PERPLEXITY_API_KEY=<perplexity key>
```

## Deployment

- **Platform**: Railway
- **Services**: Python backend (trading engine) + Next.js dashboard
- **Database**: Supabase (external PostgreSQL)
- Deploy with Railway CLI or git push

## Commands

```bash
# Python backend
python -m core.orchestrator              # Run full analysis
python -m core.orchestrator --paper      # Run paper trader
python -m core.orchestrator --backtest   # Run backtester

# Dashboard
cd dashboard && npm run dev              # Local dev
cd dashboard && npm run build            # Build check

# Tests
pytest                                   # Run test suite
```

## Build & Verify

- Always `npm run build` in dashboard/ before committing frontend changes
- Always run `python -m py_compile core/<file>.py` to check Python syntax
- Run the orchestrator after core/ changes to verify integration

## Code Style

- Python: Type hints, docstrings on public methods, Black formatter
- TypeScript: Strict mode, no any types in new code
- Commits: Conventional commits (feat:, fix:, chore:, etc.)

## Root-Level Scripts (Legacy Reference)

These are standalone experiments from before the core/ architecture. They contain useful patterns but should NOT be extended:

- `day_trader.py`, `paper_trader.py`, `enhanced_day_trader.py` — paper trading variants
- `apex_dominator.py`, `market_dominator.py`, `profit_hunter.py` — aggressive strategies
- `hyper_aggressive_model.py`, `profit_maximizer.py` — ML model experiments
- `market_data_integrator.py`, `trading_intelligence_core.py` — data integration
- `scanner.py`, `edge_scanner.py`, `combined_signals.py` — signal scanning
- `automation.py`, `hourly_runner.py` — scheduling experiments

Extract patterns from these into core/ modules. Do not add new features to them.

## What NOT to Do

- Don't store state in JSON files — use Supabase
- Don't create new root-level trading scripts — extend core/
- Don't impose conservative risk limits — let ML learn optimal risk
- Don't edit database.ts manually if we generate Supabase types
- Don't commit .env, .joblib, .db, cache/, or .venv/
