"""Tests for the Execution Engine (Layer 5)."""

from __future__ import annotations

import pytest

from cricket.config import RiskConfig
from cricket.execution.engine import (
    ExecutionEngine,
    OrderStatus,
    PositionStatus,
    RiskManager,
)
from cricket.signals.signals import SignalDirection, SignalType, TradeSignal

from datetime import datetime


def make_signal(
    edge: float = 0.05,
    direction: SignalDirection = SignalDirection.BACK,
    market_odds: float = 2.0,
    match_id: str = "exec_test",
    confidence: str = "HIGH",
) -> TradeSignal:
    model_prob = 0.55 if direction == SignalDirection.BACK else 0.45
    return TradeSignal(
        signal_id=f"TEST-{id(edge)}",
        signal_type=SignalType.DIVERGENCE,
        direction=direction,
        selection_name="Thunder",
        model_probability=model_prob,
        market_probability=model_prob - edge,
        model_fair_odds=1.0 / model_prob,
        market_odds=market_odds,
        edge_probability=edge,
        edge_ticks=int(edge / 0.01),
        match_id=match_id,
        innings=1,
        over=5.0,
        score=40,
        wickets=1,
        confidence=confidence,
    )


class TestRiskManager:
    def test_approves_valid_trade(self):
        risk = RiskManager(RiskConfig(), bankroll=10000.0)
        signal = make_signal()
        approved, reason, stake = risk.check_trade(signal, 0, 0)
        assert approved
        assert stake > 0

    def test_respects_daily_loss_limit(self):
        risk = RiskManager(RiskConfig(), bankroll=10000.0)
        # Simulate big daily loss
        risk._daily_pnl = -600.0  # > 5% of 10000
        signal = make_signal()
        approved, reason, stake = risk.check_trade(signal, 0, 0)
        assert not approved
        assert "Daily loss limit" in reason

    def test_respects_match_exposure_limit(self):
        risk = RiskManager(RiskConfig(), bankroll=10000.0)
        risk.record_exposure("match_1", 800.0)  # 8% of 10000 = max

        signal = make_signal(match_id="match_1")
        approved, reason, stake = risk.check_trade(signal, 800.0, 1)
        assert not approved

    def test_kelly_sizing(self):
        risk = RiskManager(RiskConfig(kelly_fraction=0.25), bankroll=10000.0)
        signal = make_signal(edge=0.10, market_odds=2.0)
        approved, reason, stake = risk.check_trade(signal, 0, 0)
        assert approved
        assert 0 < stake <= 10000 * 0.02  # Max 2% per trade


class TestExecutionEngine:
    def test_execute_signal(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )
        signal = make_signal()
        position = engine.execute_signal(signal)

        assert position is not None
        assert position.status == PositionStatus.OPEN
        assert position.entry_stake > 0
        assert engine.portfolio.open_positions == 1

    def test_close_position_profit(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )
        signal = make_signal(direction=SignalDirection.BACK, market_odds=2.0)
        position = engine.execute_signal(signal)

        # Price dropped (good for back) → profit
        pnl = engine.close_position(position.position_id, 1.5, reason="take_profit")
        assert pnl is not None
        assert pnl > 0
        assert position.status == PositionStatus.CLOSED

    def test_close_position_loss(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )
        signal = make_signal(direction=SignalDirection.BACK, market_odds=2.0)
        position = engine.execute_signal(signal)

        # Price went up (bad for back) → loss
        pnl = engine.close_position(position.position_id, 3.0, reason="stop_loss")
        assert pnl is not None
        assert pnl < 0

    def test_stop_loss_trigger(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )
        signal = make_signal(direction=SignalDirection.BACK, market_odds=2.0)
        position = engine.execute_signal(signal)

        # Stop loss for back position is entry * 1.5 = 3.0
        closed = engine.check_stop_losses(
            "exec_test", {"Thunder": 3.5}  # Beyond stop loss
        )
        assert len(closed) == 1

    def test_green_book(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )
        signal = make_signal(direction=SignalDirection.BACK, market_odds=2.5)
        position = engine.execute_signal(signal)

        # Price moved in our favor
        profit = engine.green_book(position.position_id, 1.8)
        assert profit is not None
        assert profit > 0
        assert position.status == PositionStatus.GREEN_BOOKED

    def test_portfolio_tracking(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )

        assert engine.portfolio.bankroll == 10000.0
        assert engine.portfolio.open_positions == 0

        signal = make_signal()
        engine.execute_signal(signal)

        assert engine.portfolio.open_positions == 1
        assert engine.portfolio.trades_today == 1

    def test_performance_summary(self):
        engine = ExecutionEngine(
            risk_config=RiskConfig(),
            bankroll=10000.0,
            paper_mode=True,
        )
        perf = engine.get_performance_summary()
        assert perf["total_trades"] == 0
        assert perf["bankroll"] == 10000.0
