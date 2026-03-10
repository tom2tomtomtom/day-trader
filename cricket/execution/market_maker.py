"""
Market-Making / Liquidity Provision Engine.

Instead of directional betting, this module acts as the "house" by
continuously quoting both back and lay prices around the model's fair
value, capturing the spread. This is fundamentally different from
directional trading:

- Directional: "I think Team A will win" → back Team A
- Market-making: "I know fair value is 1.80" → offer to back at 1.78
  AND lay at 1.82, profiting from the spread regardless of outcome

Key principles:
1. Always be two-sided: quote both back and lay
2. Manage inventory: don't accumulate too much exposure on one side
3. Widen spreads in volatility, tighten in stability
4. Use the AI pricing model as the "mid" around which to quote
5. Capture many small profits rather than few large ones
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from cricket.config import RiskConfig
from cricket.models.ensemble import EnsemblePrediction
from cricket.signals.signals import SignalDirection, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """A two-sided quote (back + lay) for a selection."""

    selection_name: str
    back_price: float  # Price we're willing to back at (buy)
    lay_price: float  # Price we're willing to lay at (sell)
    back_stake: float  # Size on the back side
    lay_stake: float  # Size on the lay side
    mid_price: float  # Model fair price (center of our quote)
    spread: float  # lay_price - back_price
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        return self.spread / self.mid_price * 100 if self.mid_price > 0 else 0

    @property
    def back_edge(self) -> float:
        """Edge captured on back side (model prob - implied prob)."""
        return 1 / self.mid_price - 1 / self.back_price if self.back_price > 0 else 0

    @property
    def lay_edge(self) -> float:
        """Edge captured on lay side."""
        return 1 / self.lay_price - 1 / self.mid_price if self.lay_price > 0 else 0


@dataclass
class InventoryPosition:
    """Tracks net inventory (exposure) on a selection.

    Positive = net backed (long), Negative = net laid (short).
    Market maker wants this near zero.
    """

    selection_name: str
    net_exposure: float = 0.0  # Positive = long, negative = short
    total_back_volume: float = 0.0
    total_lay_volume: float = 0.0
    total_spread_captured: float = 0.0
    trade_count: int = 0

    @property
    def imbalance(self) -> float:
        """Absolute imbalance in GBP."""
        return abs(self.net_exposure)

    @property
    def is_long(self) -> bool:
        return self.net_exposure > 0

    @property
    def is_short(self) -> bool:
        return self.net_exposure < 0


@dataclass
class MarketMakerConfig:
    """Configuration for market-making strategy."""

    # Spread parameters
    base_spread_ticks: int = 4  # Minimum spread in ticks (0.02 per tick)
    max_spread_ticks: int = 12  # Maximum spread during high volatility
    tick_size: float = 0.02  # Betfair minimum tick

    # Inventory management
    max_inventory: float = 500.0  # Max one-sided exposure in GBP
    inventory_skew_factor: float = 0.3  # How much to skew quotes on imbalance
    max_inventory_age_balls: int = 30  # Force-flatten inventory after N balls

    # Sizing
    quote_size_pct: float = 0.005  # 0.5% of bankroll per quote side
    min_quote_size: float = 5.0  # Minimum GBP per side

    # Volatility
    volatility_lookback: int = 12  # Balls to compute volatility
    volatility_spread_multiplier: float = 2.0  # Spread multiplier at high vol

    # Quote management
    requote_threshold: float = 0.03  # Requote if mid moves > 3%
    cancel_distance_ticks: int = 20  # Cancel if market moves > 20 ticks

    # Risk
    max_loss_per_match: float = 200.0  # Max loss before stopping MM on a match
    min_liquidity: float = 5000.0  # Don't MM if market < this volume


class MarketMaker:
    """Automated market-making engine for cricket exchange markets.

    Continuously quotes two-sided prices around the model's fair value,
    capturing the bid-ask spread as profit. Manages inventory risk by
    skewing quotes and force-flattening when exposure gets too large.

    The key insight: on Betfair, we're competing with other market makers
    and informed traders. Our edge comes from the AI pricing model giving
    us a more accurate fair value than competitors, allowing us to:
    1. Quote tighter spreads confidently (more volume captured)
    2. Avoid adverse selection (widen/pull quotes before events)
    3. Manage inventory better (know which side is correctly priced)
    """

    def __init__(
        self,
        config: Optional[MarketMakerConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        bankroll: float = 10_000.0,
    ):
        self._config = config or MarketMakerConfig()
        self._risk_config = risk_config or RiskConfig()
        self._bankroll = bankroll

        # Per-selection inventory
        self._inventory: dict[str, InventoryPosition] = {}

        # Volatility tracking
        self._price_history: dict[str, deque] = {}

        # Active quotes
        self._active_quotes: dict[str, Quote] = {}

        # Session P&L
        self._match_pnl: dict[str, float] = {}
        self._total_spread_captured: float = 0.0
        self._total_trades: int = 0

    def generate_quotes(
        self,
        match_id: str,
        prediction: EnsemblePrediction,
        team_a: str,
        team_b: str,
        market_volume: float = 50_000.0,
    ) -> list[Quote]:
        """Generate two-sided quotes for both selections.

        Args:
            match_id: Match identifier
            prediction: Model's probability estimate
            team_a: Team A name
            team_b: Team B name
            market_volume: Current matched volume (for liquidity check)

        Returns:
            List of Quote objects (one per selection)
        """
        # Don't MM if insufficient liquidity
        if market_volume < self._config.min_liquidity:
            return []

        # Check match loss limit
        match_pnl = self._match_pnl.get(match_id, 0.0)
        if match_pnl <= -self._config.max_loss_per_match:
            logger.warning("Match %s MM stopped: loss limit reached (£%.2f)", match_id, match_pnl)
            return []

        quotes = []

        for selection_name, prob in [
            (team_a, prediction.team_a_win_prob),
            (team_b, prediction.team_b_win_prob),
        ]:
            if prob <= 0.01 or prob >= 0.99:
                continue

            fair_price = 1.0 / prob
            spread = self._calculate_spread(selection_name, prediction)
            skew = self._calculate_inventory_skew(selection_name)

            # Apply skew: if we're long, lower the back price (discourage more buying)
            # and lower the lay price (encourage selling to us)
            back_price = fair_price - spread / 2 + skew
            lay_price = fair_price + spread / 2 + skew

            # Round to tick size
            back_price = self._round_to_tick(back_price)
            lay_price = self._round_to_tick(lay_price)

            # Ensure valid prices
            back_price = max(1.01, back_price)
            lay_price = max(back_price + self._config.tick_size, lay_price)

            # Size based on bankroll and inventory
            base_size = max(
                self._config.min_quote_size,
                self._bankroll * self._config.quote_size_pct,
            )
            back_size, lay_size = self._calculate_quote_sizes(
                selection_name, base_size
            )

            quote = Quote(
                selection_name=selection_name,
                back_price=back_price,
                lay_price=lay_price,
                back_stake=back_size,
                lay_stake=lay_size,
                mid_price=fair_price,
                spread=lay_price - back_price,
            )

            self._active_quotes[selection_name] = quote
            self._track_price(selection_name, fair_price)
            quotes.append(quote)

        return quotes

    def on_fill(
        self,
        match_id: str,
        selection_name: str,
        direction: SignalDirection,
        price: float,
        stake: float,
    ) -> None:
        """Handle a fill (matched bet) from the exchange.

        Updates inventory and tracks P&L contribution.
        """
        inv = self._get_inventory(selection_name)
        quote = self._active_quotes.get(selection_name)
        mid = quote.mid_price if quote else price

        if direction == SignalDirection.BACK:
            inv.net_exposure += stake
            inv.total_back_volume += stake
            # Spread captured = distance from mid to our price
            spread_earned = (mid - price) / mid * stake if mid > 0 else 0
        else:
            inv.net_exposure -= stake
            inv.total_lay_volume += stake
            spread_earned = (price - mid) / mid * stake if mid > 0 else 0

        inv.total_spread_captured += max(0, spread_earned)
        inv.trade_count += 1
        self._total_spread_captured += max(0, spread_earned)
        self._total_trades += 1

        self._match_pnl[match_id] = self._match_pnl.get(match_id, 0.0) + spread_earned

        logger.info(
            "MM fill: %s %s %s @ %.2f, £%.2f | inventory=£%.2f | spread_earned=£%.2f",
            direction.value, selection_name, match_id, price, stake,
            inv.net_exposure, spread_earned,
        )

    def should_flatten(self, selection_name: str) -> bool:
        """Check if inventory should be force-flattened."""
        inv = self._get_inventory(selection_name)
        return inv.imbalance > self._config.max_inventory

    def get_flatten_signal(
        self,
        match_id: str,
        selection_name: str,
        current_fair_price: float,
    ) -> Optional[TradeSignal]:
        """Generate a signal to flatten excess inventory.

        When inventory gets too large on one side, generate a
        directional signal to reduce exposure.
        """
        inv = self._get_inventory(selection_name)

        if inv.imbalance <= self._config.max_inventory * 0.7:
            return None

        # Flatten direction: if long, lay to reduce; if short, back to reduce
        if inv.is_long:
            direction = SignalDirection.LAY
            flatten_stake = min(inv.net_exposure * 0.5, self._config.max_inventory * 0.3)
        else:
            direction = SignalDirection.BACK
            flatten_stake = min(abs(inv.net_exposure) * 0.5, self._config.max_inventory * 0.3)

        return TradeSignal(
            signal_id=f"MM-FLAT-{uuid.uuid4().hex[:6]}",
            signal_type=_signal_type_mm(),
            direction=direction,
            selection_name=selection_name,
            model_probability=1 / current_fair_price if current_fair_price > 0 else 0.5,
            market_probability=1 / current_fair_price if current_fair_price > 0 else 0.5,
            model_fair_odds=current_fair_price,
            market_odds=current_fair_price,
            edge_probability=0.0,
            edge_ticks=0,
            match_id=match_id,
            innings=0,
            over=0,
            score=0,
            wickets=0,
            confidence="HIGH",
            suggested_stake_pct=flatten_stake / self._bankroll,
            reason=f"Inventory flatten: net_exposure=£{inv.net_exposure:.2f}",
        )

    def _calculate_spread(
        self,
        selection_name: str,
        prediction: EnsemblePrediction,
    ) -> float:
        """Calculate optimal spread based on volatility and confidence."""
        base_spread = self._config.base_spread_ticks * self._config.tick_size

        # Widen spread if model confidence is low
        if prediction.confidence == "LOW":
            base_spread *= 2.0
        elif prediction.confidence == "MEDIUM":
            base_spread *= 1.3

        # Widen spread based on recent price volatility
        vol = self._get_volatility(selection_name)
        if vol > 0.05:
            vol_multiplier = min(
                self._config.volatility_spread_multiplier,
                1.0 + vol * 10,
            )
            base_spread *= vol_multiplier

        # Cap at max spread
        max_spread = self._config.max_spread_ticks * self._config.tick_size
        return min(base_spread, max_spread)

    def _calculate_inventory_skew(self, selection_name: str) -> float:
        """Calculate price skew based on inventory imbalance.

        If we're long (too much back exposure), skew prices down
        to attract lay orders and discourage back orders.
        """
        inv = self._get_inventory(selection_name)
        if inv.imbalance < self._config.min_quote_size:
            return 0.0

        # Skew proportional to inventory as % of max
        imbalance_pct = inv.net_exposure / self._config.max_inventory
        skew = -imbalance_pct * self._config.inventory_skew_factor

        # Cap skew
        max_skew = self._config.max_spread_ticks * self._config.tick_size * 0.5
        return max(-max_skew, min(max_skew, skew))

    def _calculate_quote_sizes(
        self, selection_name: str, base_size: float
    ) -> tuple[float, float]:
        """Calculate asymmetric quote sizes based on inventory.

        Offer more size on the side that reduces inventory.
        """
        inv = self._get_inventory(selection_name)

        if inv.imbalance < self._config.min_quote_size:
            return base_size, base_size

        if inv.is_long:
            # More lay size (to sell), less back size
            lay_size = base_size * 1.5
            back_size = base_size * 0.5
        else:
            back_size = base_size * 1.5
            lay_size = base_size * 0.5

        return round(back_size, 2), round(lay_size, 2)

    def _get_volatility(self, selection_name: str) -> float:
        """Calculate recent price volatility."""
        history = self._price_history.get(selection_name)
        if not history or len(history) < 3:
            return 0.0

        prices = list(history)
        if len(prices) < 2:
            return 0.0

        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def _track_price(self, selection_name: str, price: float) -> None:
        """Track price for volatility calculation."""
        if selection_name not in self._price_history:
            self._price_history[selection_name] = deque(
                maxlen=self._config.volatility_lookback
            )
        self._price_history[selection_name].append(price)

    def _get_inventory(self, selection_name: str) -> InventoryPosition:
        """Get or create inventory tracker for a selection."""
        if selection_name not in self._inventory:
            self._inventory[selection_name] = InventoryPosition(
                selection_name=selection_name
            )
        return self._inventory[selection_name]

    def _round_to_tick(self, price: float) -> float:
        """Round to nearest valid Betfair tick."""
        tick = self._config.tick_size
        return round(round(price / tick) * tick, 2)

    def get_performance(self) -> dict:
        """Get market-making performance metrics."""
        total_volume = sum(
            inv.total_back_volume + inv.total_lay_volume
            for inv in self._inventory.values()
        )
        net_inventory = sum(inv.net_exposure for inv in self._inventory.values())

        return {
            "total_trades": self._total_trades,
            "total_volume": round(total_volume, 2),
            "total_spread_captured": round(self._total_spread_captured, 2),
            "net_inventory": round(net_inventory, 2),
            "avg_spread_per_trade": (
                round(self._total_spread_captured / self._total_trades, 4)
                if self._total_trades > 0 else 0.0
            ),
            "match_pnl": {k: round(v, 2) for k, v in self._match_pnl.items()},
            "inventory": {
                name: {
                    "net_exposure": round(inv.net_exposure, 2),
                    "back_volume": round(inv.total_back_volume, 2),
                    "lay_volume": round(inv.total_lay_volume, 2),
                    "trades": inv.trade_count,
                }
                for name, inv in self._inventory.items()
            },
        }


def _signal_type_mm():
    """Return a signal type for market-making. Uses PATTERN as closest match."""
    from cricket.signals.signals import SignalType
    return SignalType.PATTERN
