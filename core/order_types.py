#!/usr/bin/env python3
"""
ORDER TYPES — Structured order representations for the execution engine.

Supports market, limit, stop-limit, and bracket orders. In backtesting mode,
orders simulate fills against OHLC data. In paper trading mode, orders check
against each scan's current price.

Feature-gated via FeatureFlags.advanced_orders_enabled. When disabled,
all orders behave as market orders (existing behaviour).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"


@dataclass
class MarketOrder:
    """Execute immediately at current price."""
    symbol: str
    side: OrderSide
    shares: int
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING

    def check_fill(self, current_price: float, high: float = None,
                   low: float = None) -> Optional[float]:
        """Market orders always fill at current price."""
        return current_price


@dataclass
class LimitOrder:
    """Execute only at limit_price or better."""
    symbol: str
    side: OrderSide
    shares: int
    limit_price: float
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.PENDING
    bars_alive: int = 0
    max_bars: int = 20  # Cancel after 20 bars if not filled

    def check_fill(self, current_price: float, high: float = None,
                   low: float = None) -> Optional[float]:
        """
        Check if limit order can fill against this bar's price range.

        Buy limit: fills if low <= limit_price (we got our price or better)
        Sell limit: fills if high >= limit_price
        """
        self.bars_alive += 1

        if self.bars_alive > self.max_bars:
            self.status = OrderStatus.EXPIRED
            return None

        if self.side == OrderSide.BUY:
            check_low = low if low is not None else current_price
            if check_low <= self.limit_price:
                return min(self.limit_price, current_price)
        else:
            check_high = high if high is not None else current_price
            if check_high >= self.limit_price:
                return max(self.limit_price, current_price)

        return None


@dataclass
class StopLimitOrder:
    """Trigger at stop_price, then execute as limit at limit_price."""
    symbol: str
    side: OrderSide
    shares: int
    stop_price: float
    limit_price: float
    order_type: OrderType = OrderType.STOP_LIMIT
    status: OrderStatus = OrderStatus.PENDING
    triggered: bool = False
    bars_alive: int = 0
    max_bars: int = 20

    def check_fill(self, current_price: float, high: float = None,
                   low: float = None) -> Optional[float]:
        """
        Two-phase check:
        1. Has stop been triggered? (price crosses stop_price)
        2. If triggered, can limit fill?
        """
        self.bars_alive += 1

        if self.bars_alive > self.max_bars:
            self.status = OrderStatus.EXPIRED
            return None

        # Phase 1: Check trigger
        if not self.triggered:
            if self.side == OrderSide.BUY:
                check_high = high if high is not None else current_price
                if check_high >= self.stop_price:
                    self.triggered = True
            else:
                check_low = low if low is not None else current_price
                if check_low <= self.stop_price:
                    self.triggered = True

        # Phase 2: Check limit fill
        if self.triggered:
            if self.side == OrderSide.BUY:
                check_low = low if low is not None else current_price
                if check_low <= self.limit_price:
                    return min(self.limit_price, current_price)
            else:
                check_high = high if high is not None else current_price
                if check_high >= self.limit_price:
                    return max(self.limit_price, current_price)

        return None


@dataclass
class BracketOrder:
    """
    Entry order + stop loss + take profit as a single unit.
    When entry fills, stop and take-profit become active.
    """
    symbol: str
    entry: MarketOrder  # or LimitOrder
    stop_loss_price: float
    take_profit_price: float
    direction: str = "long"  # "long" or "short"
    order_type: OrderType = OrderType.BRACKET
    status: OrderStatus = OrderStatus.PENDING
    entry_filled: bool = False
    entry_fill_price: float = 0.0

    def check_entry_fill(self, current_price: float, high: float = None,
                         low: float = None) -> Optional[float]:
        """Check if entry leg fills."""
        if self.entry_filled:
            return None
        fill = self.entry.check_fill(current_price, high, low)
        if fill is not None:
            self.entry_filled = True
            self.entry_fill_price = fill
            self.entry.status = OrderStatus.FILLED
        return fill

    def check_exit(self, current_price: float, high: float = None,
                   low: float = None) -> Optional[tuple]:
        """
        Check if stop loss or take profit is hit.
        Returns (exit_price, exit_reason) or None.
        """
        if not self.entry_filled:
            return None

        check_high = high if high is not None else current_price
        check_low = low if low is not None else current_price

        if self.direction == "long":
            if check_low <= self.stop_loss_price:
                return (self.stop_loss_price, "stop_loss")
            if check_high >= self.take_profit_price:
                return (self.take_profit_price, "take_profit")
        else:
            if check_high >= self.stop_loss_price:
                return (self.stop_loss_price, "stop_loss")
            if check_low <= self.take_profit_price:
                return (self.take_profit_price, "take_profit")

        return None


class OrderManager:
    """
    Manages pending orders and checks for fills each bar.

    In backtest mode, uses OHLC data for realistic fill simulation.
    In paper mode, uses current price only.
    """

    def __init__(self):
        self.pending_orders: List = []
        self.filled_orders: List = []

    def submit(self, order) -> None:
        """Submit an order to the manager."""
        self.pending_orders.append(order)
        logger.debug(f"Order submitted: {order.order_type.value} {order.symbol}")

    def check_fills(self, current_price: float, high: float = None,
                    low: float = None) -> List[tuple]:
        """
        Check all pending orders for fills.
        Returns list of (order, fill_price) tuples for filled orders.
        """
        fills = []
        still_pending = []

        for order in self.pending_orders:
            fill_price = order.check_fill(current_price, high, low)
            if fill_price is not None:
                order.status = OrderStatus.FILLED
                fills.append((order, fill_price))
                self.filled_orders.append(order)
            elif order.status == OrderStatus.EXPIRED:
                logger.debug(f"Order expired: {order.order_type.value} {order.symbol}")
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        return fills

    def cancel_all(self, symbol: str = None) -> int:
        """Cancel pending orders, optionally filtered by symbol."""
        cancelled = 0
        still_pending = []
        for order in self.pending_orders:
            if symbol is None or order.symbol == symbol:
                order.status = OrderStatus.CANCELLED
                cancelled += 1
            else:
                still_pending.append(order)
        self.pending_orders = still_pending
        return cancelled

    @property
    def has_pending(self) -> bool:
        return len(self.pending_orders) > 0
