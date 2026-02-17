"""
Execution Engine - Layer 5.

Handles order placement, position management, risk controls,
and green-book automation. Operates in both paper and live modes.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from cricket.config import RiskConfig
from cricket.signals.signals import SignalDirection, TradeSignal

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    PARTIALLY_MATCHED = "partially_matched"
    MATCHED = "matched"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PositionStatus(Enum):
    OPEN = "open"
    GREEN_BOOKED = "green_booked"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"


@dataclass
class Order:
    """A single order placed on the exchange."""

    order_id: str
    signal_id: str
    match_id: str
    selection_name: str
    direction: SignalDirection  # BACK or LAY
    price: float
    stake: float  # GBP
    status: OrderStatus = OrderStatus.PENDING
    matched_stake: float = 0.0
    matched_price: float = 0.0
    placed_at: datetime = field(default_factory=datetime.utcnow)
    matched_at: Optional[datetime] = None

    @property
    def liability(self) -> float:
        """Maximum loss on this order."""
        if self.direction == SignalDirection.BACK:
            return self.stake
        else:
            return self.stake * (self.price - 1)

    @property
    def potential_profit(self) -> float:
        """Maximum profit on this order."""
        if self.direction == SignalDirection.BACK:
            return self.stake * (self.price - 1)
        else:
            return self.stake


@dataclass
class Position:
    """A trading position (may consist of multiple orders)."""

    position_id: str
    signal_id: str
    signal_type: str
    match_id: str
    selection_name: str
    status: PositionStatus = PositionStatus.OPEN

    entry_orders: list[Order] = field(default_factory=list)
    exit_orders: list[Order] = field(default_factory=list)

    entry_price: float = 0.0
    entry_stake: float = 0.0
    entry_direction: SignalDirection = SignalDirection.BACK

    exit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None

    green_book_profit: Optional[float] = None

    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None

    # Context at entry
    entry_over: float = 0.0
    entry_score: int = 0
    entry_wickets: int = 0
    entry_model_prob: float = 0.0
    entry_market_prob: float = 0.0

    @property
    def pnl(self) -> float:
        """Calculate current P&L."""
        if self.green_book_profit is not None:
            return self.green_book_profit

        if not self.exit_orders:
            return 0.0

        entry_cost = sum(o.matched_stake for o in self.entry_orders)
        exit_value = sum(
            o.matched_stake * (o.matched_price - 1)
            for o in self.exit_orders
            if o.status == OrderStatus.MATCHED
        )

        if self.entry_direction == SignalDirection.BACK:
            return exit_value - entry_cost
        else:
            return entry_cost - exit_value

    @property
    def hold_time_overs(self) -> Optional[float]:
        """How long the position was held in overs."""
        if self.closed_at and self.opened_at:
            return None  # Use match overs instead
        return None


@dataclass
class PortfolioState:
    """Current state of the trading portfolio."""

    bankroll: float
    cash_available: float
    total_exposure: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    active_matches: set = field(default_factory=set)


class RiskManager:
    """Pre-trade risk checks and position sizing.

    Implements Kelly Criterion-based position sizing with
    safety limits from the risk configuration.
    """

    def __init__(self, config: RiskConfig, bankroll: float):
        self._config = config
        self._bankroll = bankroll
        self._daily_pnl = 0.0
        self._match_exposure: dict[str, float] = {}

    def check_trade(
        self,
        signal: TradeSignal,
        current_exposure: float,
        active_matches: int,
    ) -> tuple[bool, str, float]:
        """Check if a trade is allowed and calculate stake size.

        Returns:
            (approved, reason, stake_gbp)
        """
        # Daily loss limit
        if self._daily_pnl <= -self._bankroll * self._config.daily_loss_limit_pct:
            return False, "Daily loss limit reached", 0.0

        # Max concurrent matches
        if active_matches >= self._config.max_concurrent_matches:
            match_id = signal.match_id
            if match_id not in self._match_exposure:
                return False, "Max concurrent matches reached", 0.0

        # Match exposure limit
        match_exp = self._match_exposure.get(signal.match_id, 0.0)
        max_match_exp = self._bankroll * self._config.max_exposure_pct
        if match_exp >= max_match_exp:
            return False, "Match exposure limit reached", 0.0

        # Calculate optimal stake using Kelly Criterion
        stake = self._calculate_kelly_stake(signal)

        # Cap at max stake per trade
        max_stake = self._bankroll * self._config.max_stake_pct
        stake = min(stake, max_stake)

        # Cap at remaining match exposure
        stake = min(stake, max_match_exp - match_exp)

        # Minimum viable stake
        if stake < 2.0:
            return False, "Stake too small", 0.0

        return True, "Approved", round(stake, 2)

    def _calculate_kelly_stake(self, signal: TradeSignal) -> float:
        """Calculate Kelly Criterion stake.

        Kelly fraction = (bp - q) / b
        where b = net odds received (market_odds - 1), p = model win probability, q = 1-p
        """
        if signal.direction == SignalDirection.BACK:
            b = signal.market_odds - 1.0
            p = signal.model_probability
        else:
            b = 1.0 / (signal.market_odds - 1.0) if signal.market_odds > 1.0 else 0
            p = 1.0 - signal.model_probability

        q = 1.0 - p

        if b <= 0 or p <= 0:
            return 0.0

        kelly = (b * p - q) / b
        kelly = max(0, kelly)

        # Apply fractional Kelly for safety
        fractional_kelly = kelly * self._config.kelly_fraction

        return self._bankroll * fractional_kelly

    def record_result(self, pnl: float, match_id: str) -> None:
        """Record a trade result for daily tracking."""
        self._daily_pnl += pnl

    def record_exposure(self, match_id: str, amount: float) -> None:
        """Record exposure for a match."""
        self._match_exposure[match_id] = (
            self._match_exposure.get(match_id, 0.0) + amount
        )

    def release_exposure(self, match_id: str, amount: float) -> None:
        """Release exposure when a position is closed."""
        if match_id in self._match_exposure:
            self._match_exposure[match_id] = max(
                0, self._match_exposure[match_id] - amount
            )

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self._daily_pnl = 0.0

    def update_bankroll(self, new_bankroll: float) -> None:
        self._bankroll = new_bankroll


class ExecutionEngine:
    """Main execution engine for placing and managing trades.

    In paper trading mode, orders are filled at the requested price.
    In live mode, orders are sent to the Betfair API via betfairlightweight.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        bankroll: float,
        paper_mode: bool = True,
        commission_rate: float = 0.02,
    ):
        self._risk = RiskManager(risk_config, bankroll)
        self._paper_mode = paper_mode
        self._commission_rate = commission_rate
        self._bankroll = bankroll

        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._trade_log: list[dict] = []

        self._portfolio = PortfolioState(
            bankroll=bankroll,
            cash_available=bankroll,
        )

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    def execute_signal(self, signal: TradeSignal) -> Optional[Position]:
        """Execute a trading signal.

        Performs risk checks, calculates stake, places orders,
        and creates a position.

        Returns:
            Position if trade was executed, None if rejected
        """
        # Risk check
        approved, reason, stake = self._risk.check_trade(
            signal,
            self._portfolio.total_exposure,
            len(self._portfolio.active_matches),
        )

        if not approved:
            logger.info(
                "Signal %s rejected: %s", signal.signal_id, reason
            )
            return None

        # Create order
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            order_id=order_id,
            signal_id=signal.signal_id,
            match_id=signal.match_id,
            selection_name=signal.selection_name,
            direction=signal.direction,
            price=signal.market_odds,
            stake=stake,
        )

        # In paper mode, orders fill immediately
        if self._paper_mode:
            order.status = OrderStatus.MATCHED
            order.matched_stake = stake
            order.matched_price = signal.market_odds
            order.matched_at = datetime.utcnow()

        self._orders[order_id] = order

        # Create position
        position_id = str(uuid.uuid4())[:8]
        position = Position(
            position_id=position_id,
            signal_id=signal.signal_id,
            signal_type=signal.signal_type.value,
            match_id=signal.match_id,
            selection_name=signal.selection_name,
            entry_orders=[order],
            entry_price=signal.market_odds,
            entry_stake=stake,
            entry_direction=signal.direction,
            entry_over=signal.over,
            entry_score=signal.score,
            entry_wickets=signal.wickets,
            entry_model_prob=signal.model_probability,
            entry_market_prob=signal.market_probability,
        )

        # Calculate stop loss
        if signal.direction == SignalDirection.BACK:
            # Stop if odds go higher (probability dropped)
            position.stop_loss_price = signal.market_odds * 1.5
        else:
            # Stop if odds go lower (probability increased)
            position.stop_loss_price = signal.market_odds * 0.67

        self._positions[position_id] = position

        # Update portfolio
        self._portfolio.total_exposure += order.liability
        self._portfolio.cash_available -= order.liability
        self._portfolio.open_positions += 1
        self._portfolio.active_matches.add(signal.match_id)
        self._portfolio.trades_today += 1
        self._risk.record_exposure(signal.match_id, order.liability)

        logger.info(
            "Executed %s: %s %s @ %.2f, stake=£%.2f (signal: %s)",
            position_id,
            signal.direction.value.upper(),
            signal.selection_name,
            signal.market_odds,
            stake,
            signal.signal_id,
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[float]:
        """Close an open position.

        Args:
            position_id: Position to close
            exit_price: Current market odds to exit at
            reason: Why the position is being closed

        Returns:
            P&L in GBP, or None if position not found
        """
        position = self._positions.get(position_id)
        if not position or position.status != PositionStatus.OPEN:
            return None

        # Create exit order (opposite direction)
        exit_direction = (
            SignalDirection.LAY
            if position.entry_direction == SignalDirection.BACK
            else SignalDirection.BACK
        )

        order_id = str(uuid.uuid4())[:8]
        exit_order = Order(
            order_id=order_id,
            signal_id=position.signal_id,
            match_id=position.match_id,
            selection_name=position.selection_name,
            direction=exit_direction,
            price=exit_price,
            stake=position.entry_stake,
        )

        if self._paper_mode:
            exit_order.status = OrderStatus.MATCHED
            exit_order.matched_stake = position.entry_stake
            exit_order.matched_price = exit_price
            exit_order.matched_at = datetime.utcnow()

        position.exit_orders.append(exit_order)
        position.exit_price = exit_price
        position.closed_at = datetime.utcnow()

        # Calculate P&L
        pnl = self._calculate_pnl(position)

        # Apply commission on net winnings
        if pnl > 0:
            pnl *= (1 - self._commission_rate)

        position.status = PositionStatus.CLOSED

        # Update portfolio
        entry_liability = sum(o.liability for o in position.entry_orders)
        self._portfolio.total_exposure -= entry_liability
        self._portfolio.cash_available += entry_liability + pnl
        self._bankroll += pnl
        self._portfolio.bankroll = self._bankroll
        self._portfolio.open_positions -= 1
        self._portfolio.daily_pnl += pnl
        self._portfolio.total_pnl += pnl

        if pnl > 0:
            self._portfolio.wins_today += 1
        else:
            self._portfolio.losses_today += 1

        self._risk.record_result(pnl, position.match_id)
        self._risk.release_exposure(position.match_id, entry_liability)
        self._risk.update_bankroll(self._bankroll)

        # Log trade
        self._trade_log.append({
            "position_id": position_id,
            "signal_type": position.signal_type,
            "selection": position.selection_name,
            "direction": position.entry_direction.value,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "stake": position.entry_stake,
            "pnl": round(pnl, 2),
            "reason": reason,
            "opened_at": position.opened_at.isoformat(),
            "closed_at": position.closed_at.isoformat(),
        })

        logger.info(
            "Closed %s: P&L=£%.2f (%s), entry=%.2f exit=%.2f",
            position_id, pnl, reason,
            position.entry_price, exit_price,
        )

        return pnl

    def check_stop_losses(self, match_id: str, current_odds: dict[str, float]) -> list[str]:
        """Check and trigger stop losses for open positions.

        Args:
            match_id: Match to check
            current_odds: Dict of {selection_name: current_odds}

        Returns:
            List of closed position IDs
        """
        closed = []
        for pid, pos in list(self._positions.items()):
            if pos.match_id != match_id or pos.status != PositionStatus.OPEN:
                continue

            current = current_odds.get(pos.selection_name)
            if current is None or pos.stop_loss_price is None:
                continue

            triggered = False
            if pos.entry_direction == SignalDirection.BACK:
                if current >= pos.stop_loss_price:
                    triggered = True
            else:
                if current <= pos.stop_loss_price:
                    triggered = True

            if triggered:
                self.close_position(pid, current, reason="stop_loss")
                closed.append(pid)

        return closed

    def green_book(
        self, position_id: str, current_odds: float
    ) -> Optional[float]:
        """Attempt to green-book (lock in profit) a position.

        Places an offsetting bet to guarantee profit regardless
        of outcome.

        Returns:
            Guaranteed profit per outcome, or None if not profitable
        """
        position = self._positions.get(position_id)
        if not position or position.status != PositionStatus.OPEN:
            return None

        entry = position.entry_price
        stake = position.entry_stake

        if position.entry_direction == SignalDirection.BACK:
            # Backed at entry, need to lay at lower odds for profit
            if current_odds >= entry:
                return None
            # Lay stake to green: (entry_stake * entry_odds) / current_odds
            lay_stake = (stake * entry) / current_odds
            profit = stake * (entry - 1) - lay_stake * (current_odds - 1)
        else:
            # Laid at entry, need to back at higher odds for profit
            if current_odds <= entry:
                return None
            back_stake = (stake * entry) / current_odds
            profit = stake - back_stake

        if profit <= 0:
            return None

        # Apply commission
        profit *= (1 - self._commission_rate)

        # Execute the green book
        position.green_book_profit = round(profit, 2)
        position.status = PositionStatus.GREEN_BOOKED
        position.closed_at = datetime.utcnow()
        position.exit_price = current_odds

        # Update portfolio
        entry_liability = sum(o.liability for o in position.entry_orders)
        self._portfolio.total_exposure -= entry_liability
        self._portfolio.cash_available += entry_liability + profit
        self._bankroll += profit
        self._portfolio.bankroll = self._bankroll
        self._portfolio.open_positions -= 1
        self._portfolio.daily_pnl += profit
        self._portfolio.total_pnl += profit
        self._portfolio.wins_today += 1

        self._risk.record_result(profit, position.match_id)
        self._risk.release_exposure(position.match_id, entry_liability)
        self._risk.update_bankroll(self._bankroll)

        logger.info(
            "Green-booked %s: guaranteed profit=£%.2f",
            position_id, profit,
        )

        return profit

    def _calculate_pnl(self, position: Position) -> float:
        """Calculate raw P&L for a closed position."""
        entry = position.entry_price
        exit_price = position.exit_price or entry
        stake = position.entry_stake

        if position.entry_direction == SignalDirection.BACK:
            # Backed: win if outcome happens (price drops)
            # P&L from closing = entry_price movement * stake / entry
            pnl = stake * (entry - exit_price) / entry
        else:
            # Laid: win if outcome doesn't happen (price rises)
            pnl = stake * (exit_price - entry) / entry

        return pnl

    def get_trade_log(self) -> list[dict]:
        return list(self._trade_log)

    def get_match_positions(self, match_id: str) -> list[Position]:
        return [
            p for p in self._positions.values() if p.match_id == match_id
        ]

    def get_open_positions(self) -> list[Position]:
        return [
            p
            for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.GREEN_BOOKED)
        ]

    def get_performance_summary(self) -> dict:
        """Get overall performance metrics."""
        closed = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.CLOSED, PositionStatus.GREEN_BOOKED, PositionStatus.STOPPED_OUT)
        ]
        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "bankroll": self._bankroll,
            }

        pnls = [p.pnl for p in closed]
        wins = [p for p in pnls if p > 0]

        return {
            "total_trades": len(closed),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
            "bankroll": round(self._bankroll, 2),
            "roi_pct": round(self._portfolio.total_pnl / (self._bankroll - self._portfolio.total_pnl) * 100, 2) if self._bankroll != self._portfolio.total_pnl else 0.0,
        }
