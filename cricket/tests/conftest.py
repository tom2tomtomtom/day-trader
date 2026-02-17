"""Shared test fixtures for cricket trading engine tests."""

from __future__ import annotations

import pytest

from cricket.config import EngineConfig, ModelConfig, RiskConfig, SignalConfig
from cricket.data.ball_event import MatchInfo


@pytest.fixture
def engine_config() -> EngineConfig:
    """Standard test configuration."""
    return EngineConfig(
        risk=RiskConfig(
            max_stake_pct=0.02,
            max_exposure_pct=0.08,
            daily_loss_limit_pct=0.05,
            kelly_fraction=0.25,
        ),
        model=ModelConfig(),
        signal=SignalConfig(),
        bankroll=10_000.0,
        paper_trading=True,
    )


@pytest.fixture
def t20_match_info() -> MatchInfo:
    """Standard T20 match for testing."""
    return MatchInfo(
        match_id="test_t20_001",
        format="t20",
        team_a="Thunder",
        team_b="Strikers",
        venue="Test Ground",
        city="Test City",
        venue_avg_first_innings_score=165.0,
        team_a_elo=1520,
        team_b_elo=1480,
    )
