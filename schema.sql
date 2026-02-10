-- Apex Trader — Supabase Schema
-- Run this in the Supabase SQL Editor (https://kvvmrbftegiclspxxcay.supabase.co)

-- Portfolio state snapshots
CREATE TABLE IF NOT EXISTS portfolio_state (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cash NUMERIC NOT NULL DEFAULT 0,
    portfolio_value NUMERIC NOT NULL DEFAULT 0,
    total_return_pct NUMERIC DEFAULT 0,
    max_drawdown_pct NUMERIC DEFAULT 0,
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    win_rate NUMERIC DEFAULT 0,
    profit_factor NUMERIC DEFAULT 0,
    portfolio_heat NUMERIC DEFAULT 0,
    open_positions INT DEFAULT 0,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Open / closed positions
CREATE TABLE IF NOT EXISTS positions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'long',
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC NOT NULL,
    shares INT NOT NULL DEFAULT 0,
    stop_loss NUMERIC DEFAULT 0,
    take_profit NUMERIC DEFAULT 0,
    entry_date TIMESTAMPTZ,
    entry_score INT DEFAULT 0,
    entry_features JSONB DEFAULT '{}',
    unrealized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    max_favorable_excursion NUMERIC DEFAULT 0,
    max_adverse_excursion NUMERIC DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, status) -- only one open position per symbol
);

-- Closed trades (ML training gold)
CREATE TABLE IF NOT EXISTS trades (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'long',
    entry_date TIMESTAMPTZ,
    entry_price NUMERIC NOT NULL,
    exit_date TIMESTAMPTZ,
    exit_price NUMERIC DEFAULT 0,
    shares INT DEFAULT 0,
    pnl_dollars NUMERIC DEFAULT 0,
    pnl_pct NUMERIC DEFAULT 0,
    exit_reason TEXT DEFAULT '',
    entry_score INT DEFAULT 0,
    entry_features JSONB DEFAULT '{}',
    max_favorable_excursion NUMERIC DEFAULT 0,
    max_adverse_excursion NUMERIC DEFAULT 0,
    regime_at_entry TEXT DEFAULT '',
    is_backtest BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Generated signals
CREATE TABLE IF NOT EXISTS signals (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT 'HOLD',
    score NUMERIC DEFAULT 0,
    confidence NUMERIC DEFAULT 0,
    reasons JSONB DEFAULT '[]',
    regime TEXT DEFAULT '',
    ml_quality_score NUMERIC,
    ml_size_multiplier NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Market regime snapshots
CREATE TABLE IF NOT EXISTS market_snapshots (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    regime TEXT NOT NULL DEFAULT '',
    fear_greed INT DEFAULT 50,
    vix NUMERIC DEFAULT 20,
    spy_change_pct NUMERIC,
    portfolio_value NUMERIC,
    extra JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ML model registry
CREATE TABLE IF NOT EXISTS ml_models (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT DEFAULT 'gradient_boosting',
    version INT DEFAULT 1,
    accuracy NUMERIC,
    precision_score NUMERIC,
    recall NUMERIC,
    f1 NUMERIC,
    feature_importance JSONB DEFAULT '{}',
    training_samples INT DEFAULT 0,
    is_active BOOLEAN DEFAULT FALSE,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Equity curve
CREATE TABLE IF NOT EXISTS equity_curve (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    portfolio_value NUMERIC NOT NULL,
    cash NUMERIC NOT NULL,
    positions_value NUMERIC NOT NULL DEFAULT 0,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    active BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    title TEXT NOT NULL,
    message TEXT DEFAULT '',
    symbol TEXT,
    data JSONB DEFAULT '{}',
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_exit_date ON trades (exit_date DESC);
CREATE INDEX IF NOT EXISTS idx_trades_is_backtest ON trades (is_backtest);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_created ON market_snapshots (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models (model_name, is_active);
CREATE INDEX IF NOT EXISTS idx_equity_curve_recorded ON equity_curve (recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts (acknowledged, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_active ON watchlist (active);

-- Enable Row Level Security (disable for service role access)
ALTER TABLE portfolio_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE equity_curve ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (backend uses service_role_key)
CREATE POLICY "Service role full access" ON portfolio_state FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON positions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON trades FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON signals FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON market_snapshots FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON ml_models FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON equity_curve FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON watchlist FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON alerts FOR ALL USING (true) WITH CHECK (true);

-- Anon key read access (for dashboard)
CREATE POLICY "Anon read access" ON portfolio_state FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON positions FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON trades FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON signals FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON market_snapshots FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON ml_models FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON equity_curve FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON watchlist FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON alerts FOR SELECT USING (true);
