"""
Odds and probability utility functions.

Handles conversion between different odds formats and
Betfair-specific price ladder operations.
"""

from __future__ import annotations

import math
from typing import Optional

# Betfair price increments
# Prices 1.01-2.00: increment 0.01
# Prices 2.00-3.00: increment 0.02
# Prices 3.00-4.00: increment 0.05
# Prices 4.00-6.00: increment 0.10
# Prices 6.00-10.0: increment 0.20
# Prices 10.0-20.0: increment 0.50
# Prices 20.0-30.0: increment 1.00
# Prices 30.0-50.0: increment 2.00
# Prices 50.0-100:  increment 5.00
# Prices 100-1000:  increment 10.0

BETFAIR_INCREMENTS = [
    (2.00, 0.01),
    (3.00, 0.02),
    (4.00, 0.05),
    (6.00, 0.10),
    (10.0, 0.20),
    (20.0, 0.50),
    (30.0, 1.00),
    (50.0, 2.00),
    (100.0, 5.00),
    (1000.0, 10.0),
]


def probability_to_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0:
        return 1000.0
    if prob >= 1:
        return 1.01
    return round(1.0 / prob, 2)


def odds_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 0:
        return 1.0
    return 1.0 / odds


def snap_to_betfair_price(price: float, round_up: bool = False) -> float:
    """Snap a price to the nearest valid Betfair price ladder tick.

    Args:
        price: Raw decimal odds
        round_up: If True, round to next higher valid price

    Returns:
        Nearest valid Betfair price
    """
    price = max(1.01, min(1000.0, price))

    increment = 0.01
    for threshold, inc in BETFAIR_INCREMENTS:
        if price < threshold:
            increment = inc
            break

    if round_up:
        return round(math.ceil(price / increment) * increment, 2)
    else:
        return round(round(price / increment) * increment, 2)


def ticks_between(price_a: float, price_b: float) -> int:
    """Count the number of Betfair ticks between two prices."""
    low = min(price_a, price_b)
    high = max(price_a, price_b)

    ticks = 0
    current = low
    while current < high - 1e-6:
        increment = 0.01
        for threshold, inc in BETFAIR_INCREMENTS:
            if current < threshold:
                increment = inc
                break
        current += increment
        ticks += 1

    return ticks


def move_price(price: float, ticks: int) -> float:
    """Move a price by N ticks on the Betfair ladder.

    Positive ticks = higher price (lower probability).
    Negative ticks = lower price (higher probability).
    """
    current = price
    direction = 1 if ticks > 0 else -1

    for _ in range(abs(ticks)):
        increment = 0.01
        for threshold, inc in BETFAIR_INCREMENTS:
            if current < threshold:
                increment = inc
                break
        current += increment * direction
        current = max(1.01, min(1000.0, current))

    return round(current, 2)


def calculate_overround(probabilities: list[float]) -> float:
    """Calculate the market overround (margin).

    A fair market sums to 1.0. Betfair markets are close to fair
    but the spread creates a small overround.
    """
    return sum(probabilities) - 1.0


def remove_overround(
    probabilities: list[float], method: str = "proportional"
) -> list[float]:
    """Remove overround to get true probabilities.

    Args:
        probabilities: List of implied probabilities (sum > 1.0)
        method: "proportional" (default) or "shin"

    Returns:
        Adjusted probabilities summing to 1.0
    """
    total = sum(probabilities)
    if total <= 0:
        return probabilities
    return [p / total for p in probabilities]


def kelly_stake(
    probability: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """Calculate Kelly Criterion stake.

    Args:
        probability: Estimated true probability of winning
        odds: Decimal odds available
        bankroll: Current bankroll
        fraction: Kelly fraction (0.25 = quarter-Kelly)

    Returns:
        Optimal stake in currency units
    """
    b = odds - 1.0
    q = 1.0 - probability

    if b <= 0 or probability <= 0:
        return 0.0

    kelly = (b * probability - q) / b
    kelly = max(0, kelly)

    return bankroll * kelly * fraction
