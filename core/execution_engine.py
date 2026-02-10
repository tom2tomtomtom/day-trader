#!/usr/bin/env python3
"""
UNIFIED EXECUTION ENGINE — Single code path for position management.

Both backtester and paper_trader delegate here so that backtest results
accurately reflect live trading behaviour (trailing stops, partial exits,
ATR-based stops, signal reversal).

Resolves the #1 structural priority: backtester previously ran completely
different exit logic than the paper trader, making backtest results unreliable.
"""

import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    TRAILING_STOP = "trailing_stop"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PARTIAL_TARGET = "partial_target"
    SIGNAL_REVERSAL = "signal_reversal"
    END_OF_DATA = "end_of_data"


@dataclass
class ManagedPosition:
    """A position managed by the execution engine."""
    symbol: str
    direction: str          # "long" or "short"
    entry_price: float
    current_price: float
    shares: int
    stop_loss: float
    take_profit: float
    entry_date: str
    entry_score: int
    entry_features: Dict = field(default_factory=dict)
    entry_idx: int = 0      # Bar index at entry (for backtester hold_days)

    # ATR trailing stop
    atr_at_entry: float = 0.0
    trailing_stop: float = 0.0

    # Partial exit tracking
    partial_exited: bool = False
    original_shares: int = 0

    # MFE / MAE tracking
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Unrealized P&L (updated each bar)
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def _trail_multiplier(self) -> float:
        """ATR multiplier: tighter after partial exit."""
        return 1.5 if self.partial_exited else 2.5

    def update_price(self, current_price: float):
        """Update position with new price, track MFE/MAE and trailing stop."""
        self.current_price = current_price

        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.shares
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.shares
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

        # Track MFE/MAE
        if self.unrealized_pnl_pct > self.max_favorable_excursion:
            self.max_favorable_excursion = self.unrealized_pnl_pct
        if self.unrealized_pnl_pct < -self.max_adverse_excursion:
            self.max_adverse_excursion = abs(self.unrealized_pnl_pct)

        # Ratchet ATR trailing stop
        if self.atr_at_entry > 0 and self.trailing_stop > 0:
            if self.direction == "long":
                new_trail = current_price - self.atr_at_entry * self._trail_multiplier()
                if new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail
            else:
                new_trail = current_price + self.atr_at_entry * self._trail_multiplier()
                if new_trail < self.trailing_stop:
                    self.trailing_stop = new_trail


@dataclass
class ExitSignal:
    """Result from check_exit — describes what happened."""
    reason: ExitReason
    exit_price: float
    exit_shares: int
    is_partial: bool = False


class ExecutionEngine:
    """
    Unified position management shared by backtester and paper trader.

    Handles: trailing stops, partial exits, ATR stops, signal reversal,
    slippage, MFE/MAE tracking.
    """

    LONG_ENTRY_THRESHOLD = 25
    SHORT_ENTRY_THRESHOLD = -25
    ATR_TRAIL_MULTIPLIER = 2.5

    def __init__(self, slippage_pct: float = 0.001, commission: float = 0.0):
        self.slippage_pct = slippage_pct
        self.commission = commission

    def create_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        take_profit: float,
        entry_date: str,
        entry_score: int,
        atr: float = 0.0,
        entry_features: Optional[Dict] = None,
        entry_idx: int = 0,
        apply_slippage: bool = True,
    ) -> ManagedPosition:
        """Create a new managed position with ATR trailing stop."""
        if apply_slippage:
            if direction == "long":
                entry_price = entry_price * (1 + self.slippage_pct)
            else:
                entry_price = entry_price * (1 - self.slippage_pct)
            entry_price = round(entry_price, 2)

        # ATR trailing stop
        if atr > 0:
            if direction == "long":
                trailing = entry_price - atr * self.ATR_TRAIL_MULTIPLIER
            else:
                trailing = entry_price + atr * self.ATR_TRAIL_MULTIPLIER
        else:
            trailing = 0.0

        return ManagedPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_date=entry_date,
            entry_score=entry_score,
            entry_features=entry_features or {},
            entry_idx=entry_idx,
            atr_at_entry=atr,
            trailing_stop=trailing,
            original_shares=shares,
        )

    def check_exit(
        self,
        position: ManagedPosition,
        current_price: float,
        signal_score: int = 0,
    ) -> Optional[ExitSignal]:
        """
        Check if position should be exited (fully or partially).

        Priority order:
        1. ATR trailing stop
        2. Fixed stop loss
        3. Take profit (with partial exit on first hit)
        4. Signal reversal

        Returns ExitSignal or None.
        """
        direction = position.direction

        # 1. ATR trailing stop
        if position.trailing_stop > 0:
            if direction == "long" and current_price <= position.trailing_stop:
                exit_price = self._apply_exit_slippage(current_price, direction)
                return ExitSignal(
                    reason=ExitReason.TRAILING_STOP,
                    exit_price=exit_price,
                    exit_shares=position.shares,
                )
            elif direction == "short" and current_price >= position.trailing_stop:
                exit_price = self._apply_exit_slippage(current_price, direction)
                return ExitSignal(
                    reason=ExitReason.TRAILING_STOP,
                    exit_price=exit_price,
                    exit_shares=position.shares,
                )

        # 2. Fixed stop loss
        if direction == "long" and current_price <= position.stop_loss:
            exit_price = self._apply_exit_slippage(current_price, direction)
            return ExitSignal(
                reason=ExitReason.STOP_LOSS,
                exit_price=exit_price,
                exit_shares=position.shares,
            )
        elif direction == "short" and current_price >= position.stop_loss:
            exit_price = self._apply_exit_slippage(current_price, direction)
            return ExitSignal(
                reason=ExitReason.STOP_LOSS,
                exit_price=exit_price,
                exit_shares=position.shares,
            )

        # 3. Take profit — partial exit on first hit
        if direction == "long" and current_price >= position.take_profit:
            if not position.partial_exited and position.shares > 1:
                exit_shares = position.shares // 2
                exit_price = self._apply_exit_slippage(current_price, direction)
                return ExitSignal(
                    reason=ExitReason.PARTIAL_TARGET,
                    exit_price=exit_price,
                    exit_shares=exit_shares,
                    is_partial=True,
                )
            exit_price = self._apply_exit_slippage(current_price, direction)
            return ExitSignal(
                reason=ExitReason.TAKE_PROFIT,
                exit_price=exit_price,
                exit_shares=position.shares,
            )
        elif direction == "short" and current_price <= position.take_profit:
            if not position.partial_exited and position.shares > 1:
                exit_shares = position.shares // 2
                exit_price = self._apply_exit_slippage(current_price, direction)
                return ExitSignal(
                    reason=ExitReason.PARTIAL_TARGET,
                    exit_price=exit_price,
                    exit_shares=exit_shares,
                    is_partial=True,
                )
            exit_price = self._apply_exit_slippage(current_price, direction)
            return ExitSignal(
                reason=ExitReason.TAKE_PROFIT,
                exit_price=exit_price,
                exit_shares=position.shares,
            )

        # 4. Signal reversal
        if direction == "long" and signal_score <= self.SHORT_ENTRY_THRESHOLD:
            exit_price = self._apply_exit_slippage(current_price, direction)
            return ExitSignal(
                reason=ExitReason.SIGNAL_REVERSAL,
                exit_price=exit_price,
                exit_shares=position.shares,
            )
        elif direction == "short" and signal_score >= self.LONG_ENTRY_THRESHOLD:
            exit_price = self._apply_exit_slippage(current_price, direction)
            return ExitSignal(
                reason=ExitReason.SIGNAL_REVERSAL,
                exit_price=exit_price,
                exit_shares=position.shares,
            )

        return None

    def apply_partial_exit(self, position: ManagedPosition, exit_shares: int):
        """
        Apply partial exit to position: reduce shares, move stop to breakeven,
        tighten trailing stop.
        """
        position.shares -= exit_shares
        position.stop_loss = position.entry_price  # Breakeven stop
        position.partial_exited = True
        # Trailing stop tightens via _trail_multiplier() (2.5 -> 1.5)

    def calculate_pnl(
        self,
        position: ManagedPosition,
        exit_price: float,
        exit_shares: int,
    ) -> Tuple[float, float]:
        """Calculate P&L dollars and percentage for an exit."""
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * exit_shares
            pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - exit_price) * exit_shares
            pnl_pct = (position.entry_price - exit_price) / position.entry_price * 100

        pnl -= self.commission
        return round(pnl, 2), round(pnl_pct, 4)

    def close_at_end(self, position: ManagedPosition, final_price: float) -> ExitSignal:
        """Force-close position at end of data (backtesting)."""
        exit_price = self._apply_exit_slippage(final_price, position.direction)
        return ExitSignal(
            reason=ExitReason.END_OF_DATA,
            exit_price=exit_price,
            exit_shares=position.shares,
        )

    def _apply_exit_slippage(self, price: float, direction: str) -> float:
        """Apply slippage on exit."""
        if direction == "long":
            return round(price * (1 - self.slippage_pct), 2)
        else:
            return round(price * (1 + self.slippage_pct), 2)
