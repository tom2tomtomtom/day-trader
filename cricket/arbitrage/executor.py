"""
Arbitrage Execution Engine.

Handles the execution of detected arbitrage opportunities.
The key challenge: all legs must be filled simultaneously.
If one leg fills and the other doesn't, you're left with
unhedged directional exposure — the opposite of what you want.

Execution strategies:
1. SIMULTANEOUS: Place all legs at once, cancel unfilled after timeout
2. SEQUENTIAL: Fill the harder leg first, then hedge with the easier leg
3. CONDITIONAL: Only place leg 2 after leg 1 confirms fill

For cross-exchange arbs, SEQUENTIAL is usually best because you
fill the less liquid exchange first and hedge on the more liquid one.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from cricket.arbitrage.detector import ArbLeg, ArbOpportunity, ArbStatus, ArbType

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"


@dataclass
class ArbExecution:
    """Tracks execution state of an arbitrage opportunity."""

    arb: ArbOpportunity
    strategy: ExecutionStrategy
    target_stake_multiplier: float = 1.0  # Scale relative to detected stakes

    legs_filled: int = 0
    legs_failed: int = 0
    actual_profit: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.legs_filled + self.legs_failed == len(self.arb.legs)

    @property
    def all_filled(self) -> bool:
        return self.legs_filled == len(self.arb.legs)

    @property
    def has_unhedged_exposure(self) -> bool:
        """True if some legs filled but not all — dangerous."""
        return 0 < self.legs_filled < len(self.arb.legs)


class ArbitrageExecutor:
    """Executes arbitrage opportunities in paper or live mode.

    In paper mode, simulates fills with configurable fill rates.
    In live mode, would send orders to the relevant exchanges.
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        max_arb_stake_pct: float = 0.05,  # Max 5% of bankroll per arb
        paper_mode: bool = True,
        paper_fill_rate: float = 0.8,  # 80% fill rate in paper mode
    ):
        self._bankroll = bankroll
        self._max_stake_pct = max_arb_stake_pct
        self._paper_mode = paper_mode
        self._paper_fill_rate = paper_fill_rate

        self._executions: list[ArbExecution] = []
        self._total_profit: float = 0.0
        self._total_attempted: int = 0
        self._total_filled: int = 0

    def execute(
        self,
        arb: ArbOpportunity,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    ) -> ArbExecution:
        """Execute an arbitrage opportunity.

        Args:
            arb: The detected opportunity
            strategy: How to execute the legs

        Returns:
            ArbExecution with fill results
        """
        self._total_attempted += 1

        # Scale stakes to bankroll limit
        max_total_stake = self._bankroll * self._max_stake_pct
        if arb.total_stake > 0:
            scale = min(1.0, max_total_stake / arb.total_stake)
        else:
            scale = 1.0

        execution = ArbExecution(
            arb=arb,
            strategy=strategy,
            target_stake_multiplier=scale,
            started_at=datetime.utcnow(),
        )

        # Scale all leg stakes
        for leg in arb.legs:
            leg.stake = round(leg.stake * scale, 2)

        arb.total_stake = sum(l.stake for l in arb.legs)

        if self._paper_mode:
            self._execute_paper(execution)
        else:
            self._execute_live(execution)

        execution.completed_at = datetime.utcnow()
        if execution.started_at:
            execution.execution_time_ms = (
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )

        self._executions.append(execution)

        if execution.all_filled:
            self._total_filled += 1
            self._total_profit += execution.actual_profit
            self._bankroll += execution.actual_profit
            logger.info(
                "Arb %s executed: profit=£%.2f (%s)",
                arb.arb_id, execution.actual_profit, arb.arb_type.value,
            )
        elif execution.has_unhedged_exposure:
            logger.warning(
                "Arb %s PARTIAL FILL — unhedged exposure! %d/%d legs filled",
                arb.arb_id, execution.legs_filled, len(arb.legs),
            )
        else:
            logger.info("Arb %s failed to fill any legs", arb.arb_id)

        return execution

    def _execute_paper(self, execution: ArbExecution) -> None:
        """Simulate execution with configurable fill rates."""
        import random

        arb = execution.arb
        all_filled = True

        for leg in arb.legs:
            if random.random() < self._paper_fill_rate:
                leg.filled = True
                leg.fill_price = leg.odds
                leg.fill_stake = leg.stake
                execution.legs_filled += 1
            else:
                leg.filled = False
                execution.legs_failed += 1
                all_filled = False

        if all_filled:
            execution.actual_profit = self._calculate_actual_profit(arb)
            arb.status = ArbStatus.FILLED
        elif execution.legs_filled > 0:
            # Partial fill — estimate exposure
            execution.actual_profit = 0.0  # Unknown until settled
            arb.status = ArbStatus.FAILED
        else:
            arb.status = ArbStatus.EXPIRED

    def _execute_live(self, execution: ArbExecution) -> None:
        """Live execution placeholder."""
        logger.warning("Live arb execution not yet implemented")
        execution.arb.status = ArbStatus.FAILED

    def _calculate_actual_profit(self, arb: ArbOpportunity) -> float:
        """Calculate actual profit based on fill prices."""
        if arb.arb_type == ArbType.CROSS_EXCHANGE:
            return self._calc_cross_exchange_profit(arb)
        elif arb.arb_type == ArbType.INTRA_MARKET:
            return arb.guaranteed_profit * (arb.total_stake / 100.0)  # Scale from normalized
        else:
            return arb.guaranteed_profit

    def _calc_cross_exchange_profit(self, arb: ArbOpportunity) -> float:
        """Calculate cross-exchange arb profit from actual fills."""
        back_leg = next((l for l in arb.legs if l.side == "back"), None)
        lay_leg = next((l for l in arb.legs if l.side == "lay"), None)

        if not back_leg or not lay_leg or not back_leg.filled or not lay_leg.filled:
            return 0.0

        back_price = back_leg.fill_price or back_leg.odds
        lay_price = lay_leg.fill_price or lay_leg.odds
        back_stake = back_leg.fill_stake or back_leg.stake
        lay_stake = lay_leg.fill_stake or lay_leg.stake

        # Profit if selection wins
        if_wins = back_stake * (back_price - 1) - lay_stake * (lay_price - 1)
        # Profit if selection loses
        if_loses = lay_stake - back_stake

        return min(if_wins, if_loses)

    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "total_attempted": self._total_attempted,
            "total_filled": self._total_filled,
            "fill_rate": (
                self._total_filled / self._total_attempted
                if self._total_attempted > 0 else 0.0
            ),
            "total_profit": round(self._total_profit, 2),
            "bankroll": round(self._bankroll, 2),
            "partial_fills": sum(
                1 for e in self._executions if e.has_unhedged_exposure
            ),
            "by_type": self._profit_by_type(),
        }

    def _profit_by_type(self) -> dict[str, dict]:
        """Break down profit by arb type."""
        result: dict[str, dict] = {}
        for ex in self._executions:
            atype = ex.arb.arb_type.value
            if atype not in result:
                result[atype] = {"count": 0, "filled": 0, "profit": 0.0}
            result[atype]["count"] += 1
            if ex.all_filled:
                result[atype]["filled"] += 1
                result[atype]["profit"] += ex.actual_profit
        return result
