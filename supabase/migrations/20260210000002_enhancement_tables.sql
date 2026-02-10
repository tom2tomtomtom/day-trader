-- Apex Trader — Enhancement tables (Phases 2-4)
-- Run this AFTER 20260210000000_initial_schema.sql

-- ML model artifacts (base64-serialized models for Railway persistence)
CREATE TABLE IF NOT EXISTS ml_model_artifacts (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    model_type TEXT DEFAULT 'lightgbm',
    version INT DEFAULT 1,
    artifacts JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Signal evaluations (nightly accuracy tracking)
CREATE TABLE IF NOT EXISTS signal_evaluations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    symbol TEXT NOT NULL,
    regime TEXT DEFAULT '',
    action TEXT NOT NULL,
    total_signals INT DEFAULT 0,
    correct_signals INT DEFAULT 0,
    accuracy NUMERIC DEFAULT 0,
    avg_pnl_pct NUMERIC DEFAULT 0,
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Strategy configurations (declarative strategy params)
CREATE TABLE IF NOT EXISTS strategy_configs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    enabled BOOLEAN DEFAULT TRUE,
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Walk-forward optimization results
CREATE TABLE IF NOT EXISTS optimization_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    best_params JSONB DEFAULT '{}',
    best_sharpe NUMERIC DEFAULT 0,
    best_return NUMERIC DEFAULT 0,
    n_folds INT DEFAULT 5,
    total_combos INT DEFAULT 0,
    all_results JSONB DEFAULT '[]',
    period TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_signal_evals_symbol ON signal_evaluations (symbol);
CREATE INDEX IF NOT EXISTS idx_signal_evals_created ON signal_evaluations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_strategy ON optimization_runs (strategy, symbol);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_created ON optimization_runs (created_at DESC);

-- RLS
ALTER TABLE ml_model_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_runs ENABLE ROW LEVEL SECURITY;

-- Service role full access
CREATE POLICY "Service role full access" ON ml_model_artifacts FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON signal_evaluations FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON strategy_configs FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON optimization_runs FOR ALL USING (true) WITH CHECK (true);

-- Anon read access (dashboard)
CREATE POLICY "Anon read access" ON ml_model_artifacts FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON signal_evaluations FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON strategy_configs FOR SELECT USING (true);
CREATE POLICY "Anon read access" ON optimization_runs FOR SELECT USING (true);

-- Anon write for strategy_configs (dashboard saves configs)
CREATE POLICY "Anon write strategy configs" ON strategy_configs FOR INSERT WITH CHECK (true);
CREATE POLICY "Anon update strategy configs" ON strategy_configs FOR UPDATE USING (true) WITH CHECK (true);
