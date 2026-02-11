#!/usr/bin/env python3
"""
CENTRALIZED CONFIG — Single source of truth for all configuration.

Loads from .env file and provides typed access to all settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    anon_key: str
    service_role_key: str


@dataclass(frozen=True)
class APIKeys:
    anthropic: str
    finnhub: str
    perplexity: str
    unusual_whales: str
    glassnode: str


@dataclass(frozen=True)
class TradingConfig:
    initial_capital: float = 100_000
    # Risk limits (aggressive — Phase 3)
    max_portfolio_heat: float = 0.50
    max_single_position: float = 0.25
    max_correlated_exposure: float = 0.35
    max_drawdown_halt: float = 0.30
    kelly_fraction: float = 0.50
    # Scan intervals (seconds)
    stock_scan_interval: int = 300       # 5 min
    crypto_scan_interval: int = 900      # 15 min
    ml_retrain_interval: int = 86400     # 24 hours
    # ML
    min_trades_for_ml: int = 30


@dataclass(frozen=True)
class FeatureFlags:
    ml_gate_enabled: bool = False
    pre_trade_risk_enabled: bool = False
    rl_agent_enabled: bool = False
    telegram_enabled: bool = False
    advanced_orders_enabled: bool = False
    learning_loop_enabled: bool = False


class Config:
    """Centralized configuration with typed access."""

    def __init__(self):
        self.supabase = SupabaseConfig(
            url=os.getenv("SUPABASE_URL", ""),
            anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
            service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        )
        self.api_keys = APIKeys(
            anthropic=os.getenv("ANTHROPIC_API_KEY", ""),
            finnhub=os.getenv("FINNHUB_API_KEY", ""),
            perplexity=os.getenv("PERPLEXITY_API_KEY", ""),
            unusual_whales=os.getenv("UNUSUAL_WHALES_KEY", ""),
            glassnode=os.getenv("GLASSNODE_KEY", ""),
        )
        self.trading = TradingConfig()
        self.base_dir = BASE_DIR
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase.url and self.supabase.service_role_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.api_keys.anthropic)

    @property
    def has_finnhub(self) -> bool:
        return bool(self.api_keys.finnhub)

    @property
    def has_perplexity(self) -> bool:
        return bool(self.api_keys.perplexity)


# Singletons
_config: Optional[Config] = None
_feature_flags: Optional[FeatureFlags] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_feature_flags() -> FeatureFlags:
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags(
            ml_gate_enabled=os.getenv("FEATURE_ML_GATE", "false").lower() == "true",
            pre_trade_risk_enabled=os.getenv("FEATURE_PRE_TRADE_RISK", "false").lower() == "true",
            rl_agent_enabled=os.getenv("FEATURE_RL_AGENT", "false").lower() == "true",
            telegram_enabled=os.getenv("FEATURE_TELEGRAM", "false").lower() == "true",
            advanced_orders_enabled=os.getenv("FEATURE_ADVANCED_ORDERS", "false").lower() == "true",
            learning_loop_enabled=os.getenv("FEATURE_LEARNING_LOOP", "false").lower() == "true",
        )
    return _feature_flags
