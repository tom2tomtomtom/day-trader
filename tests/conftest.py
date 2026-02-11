"""Shared test fixtures for learning framework tests."""

import json
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


def make_trade(
    symbol="AAPL",
    pnl_pct=2.5,
    regime="trending_up",
    exit_reason="take_profit",
    day_of_week=1,
    hour_of_day=10,
    rsi_14=45.0,
    return_1d=0.5,
    **extra_features,
):
    """Create a realistic trade dict for testing."""
    features = {
        "rsi_14": rsi_14,
        "macd_line": 0.5,
        "macd_signal": 0.3,
        "macd_histogram": 0.2,
        "bb_position": 0.5,
        "bb_width": 3.0,
        "sma_10_dist": 0.5,
        "sma_20_dist": 1.0,
        "sma_50_dist": 2.0,
        "ma_alignment": 15,
        "stochastic_k": 60.0,
        "stochastic_d": 55.0,
        "adx": 25.0,
        "obv_slope": 100.0,
        "vwap_distance_pct": 0.1,
        "volatility_20d": 0.25,
        "atr_14": 2.0,
        "atr_pct": 1.5,
        "relative_volume": 1.2,
        "volume_trend": 50.0,
        "return_1d": return_1d,
        "return_5d": 1.0,
        "return_20d": 5.0,
        "price_vs_high_52w": -5.0,
        "price_vs_low_52w": 20.0,
        "regime": regime,
        "regime_confidence": 0.7,
        "fear_greed": 45,
        "fear_greed_trend": "stable",
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "is_market_open": True,
        "composite_score": 50,
        "ensemble_confidence": 0.6,
        **extra_features,
    }
    return {
        "symbol": symbol,
        "direction": "long",
        "entry_date": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
        "entry_price": 150.0,
        "exit_date": datetime.now(timezone.utc).isoformat(),
        "exit_price": 150.0 * (1 + pnl_pct / 100),
        "shares": 10,
        "pnl_dollars": pnl_pct * 15,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "entry_score": 60,
        "entry_features": features,
        "max_favorable_excursion": max(pnl_pct, 0) * 1.2,
        "max_adverse_excursion": min(pnl_pct, 0) * 1.2,
        "regime_at_entry": regime,
        "is_backtest": False,
    }


@pytest.fixture
def sample_trades():
    """Generate 40 diverse trades for testing."""
    np.random.seed(42)
    trades = []
    regimes = ["trending_up", "trending_down", "ranging", "high_vol"]
    exits = ["take_profit", "stop_loss", "trailing_stop"]

    for i in range(40):
        regime = regimes[i % len(regimes)]
        exit_r = exits[i % len(exits)]
        pnl = np.random.normal(1.0 if regime == "trending_up" else -0.5, 3.0)
        day = i % 5
        hour = 9 + (i % 8)

        trades.append(make_trade(
            symbol=["AAPL", "MSFT", "GOOGL", "TSLA"][i % 4],
            pnl_pct=round(pnl, 2),
            regime=regime,
            exit_reason=exit_r,
            day_of_week=day,
            hour_of_day=hour,
            rsi_14=30 + np.random.rand() * 40,
            return_1d=np.random.normal(0, 1),
        ))

    return trades


@pytest.fixture
def mock_db():
    """Mock TradingDB that returns empty but doesn't crash."""
    db = MagicMock()
    db.connected = True
    db.get_trades_with_features.return_value = []
    db.get_active_model.return_value = None
    db.get_hypotheses.return_value = []
    db.get_experiments.return_value = []
    db.get_ensemble_weight_overrides.return_value = []
    db.get_temporal_adjustments.return_value = []
    db.save_hypothesis.return_value = True
    db.save_experiment.return_value = True
    db.save_learning_action.return_value = True
    db.save_feature_drift.return_value = True
    return db
