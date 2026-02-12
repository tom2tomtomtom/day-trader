"""
Configuration management for the Cricket Trading Engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class MatchFormat(Enum):
    T20 = "t20"
    ODI = "odi"
    TEST = "test"


class InningsPhase(Enum):
    POWERPLAY = "powerplay"
    MIDDLE = "middle"
    DEATH = "death"


class MarketType(Enum):
    MATCH_ODDS = "match_odds"
    INNINGS_RUNS = "innings_runs"
    SESSION = "session"


@dataclass(frozen=True)
class ExchangeConfig:
    """Betfair exchange configuration."""
    username: str = ""
    password: str = ""
    app_key: str = ""
    cert_path: str = ""
    event_type_id: str = "4"  # Cricket
    commission_rate: float = 0.02  # 2% standard
    min_liquidity: float = 10_000.0  # GBP matched
    streaming_enabled: bool = True


@dataclass(frozen=True)
class CricketDataConfig:
    """Cricket data feed configuration."""
    sportradar_api_key: str = ""
    cricsheet_data_dir: str = "data/cricsheet"
    roanuz_api_key: str = ""
    primary_provider: str = "sportradar"
    failover_provider: str = "roanuz"
    max_feed_latency_ms: int = 5000


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    max_stake_pct: float = 0.02  # 2% of bankroll per trade
    max_exposure_pct: float = 0.08  # 8% per match
    max_concurrent_matches: int = 3
    stop_loss_pct: float = 0.50  # 50% of potential profit
    daily_loss_limit_pct: float = 0.05  # 5% of bankroll
    min_edge_ticks: int = 2  # Minimum ticks of edge to trade
    kelly_fraction: float = 0.25  # Quarter-Kelly for safety


@dataclass(frozen=True)
class ModelConfig:
    """Model ensemble configuration."""
    statistical_weight: float = 0.30
    xgboost_weight: float = 0.45
    lstm_weight: float = 0.25
    confidence_high_threshold: float = 0.02  # 2% agreement
    confidence_medium_threshold: float = 0.05  # 5% agreement
    retrain_interval_days: int = 30
    min_training_matches: int = 500


@dataclass(frozen=True)
class SignalConfig:
    """Signal generation thresholds."""
    overreaction_min_edge_ticks: int = 3
    divergence_min_probability_gap: float = 0.05  # 5%
    pattern_min_edge_ticks: int = 2
    pattern_min_touches: int = 3  # Support/resistance touches
    powerplay_wicket_probability: float = 0.78  # Historical T20
    mean_reversion_window_overs: int = 3
    mean_reversion_probability: float = 0.72


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    cricket_data: CricketDataConfig = field(default_factory=CricketDataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)

    bankroll: float = 10_000.0  # Starting bankroll in GBP
    paper_trading: bool = True  # Start in paper mode
    log_level: str = "INFO"
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Load configuration from environment variables."""
        return cls(
            exchange=ExchangeConfig(
                username=os.getenv("BETFAIR_USERNAME", ""),
                password=os.getenv("BETFAIR_PASSWORD", ""),
                app_key=os.getenv("BETFAIR_APP_KEY", ""),
                cert_path=os.getenv("BETFAIR_CERT_PATH", ""),
            ),
            cricket_data=CricketDataConfig(
                sportradar_api_key=os.getenv("SPORTRADAR_API_KEY", ""),
                roanuz_api_key=os.getenv("ROANUZ_API_KEY", ""),
            ),
            bankroll=float(os.getenv("CRICKET_BANKROLL", "10000")),
            paper_trading=os.getenv("CRICKET_LIVE_TRADING", "").lower() != "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Format-specific constants
FORMAT_OVERS: dict[MatchFormat, Optional[int]] = {
    MatchFormat.T20: 20,
    MatchFormat.ODI: 50,
    MatchFormat.TEST: None,  # Unlimited
}

T20_PHASES: dict[InningsPhase, tuple[int, int]] = {
    InningsPhase.POWERPLAY: (1, 6),
    InningsPhase.MIDDLE: (7, 15),
    InningsPhase.DEATH: (16, 20),
}

ODI_PHASES: dict[InningsPhase, tuple[int, int]] = {
    InningsPhase.POWERPLAY: (1, 10),
    InningsPhase.MIDDLE: (11, 40),
    InningsPhase.DEATH: (41, 50),
}
