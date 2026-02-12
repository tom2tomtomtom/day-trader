"""
Exchange price feed interface.

Provides an abstraction over the Betfair Streaming API for real-time
exchange price data. Includes a simulated feed for backtesting.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """A single price level in the order book."""
    price: float
    size: float  # Available volume at this price


@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of exchange prices for a selection."""

    market_id: str
    selection_id: int
    selection_name: str
    timestamp: datetime

    # Best 3 back (buy) prices - descending
    back_prices: list[PriceLevel] = field(default_factory=list)
    # Best 3 lay (sell) prices - ascending
    lay_prices: list[PriceLevel] = field(default_factory=list)

    last_price_traded: Optional[float] = None
    total_matched: float = 0.0
    market_status: str = "OPEN"

    @property
    def best_back(self) -> Optional[float]:
        return self.back_prices[0].price if self.back_prices else None

    @property
    def best_lay(self) -> Optional[float]:
        return self.lay_prices[0].price if self.lay_prices else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_back and self.best_lay:
            return self.best_lay - self.best_back
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_back and self.best_lay:
            return (self.best_back + self.best_lay) / 2.0
        return None

    @property
    def implied_probability(self) -> Optional[float]:
        """Implied probability from mid-price."""
        if self.mid_price and self.mid_price > 0:
            return 1.0 / self.mid_price
        return None


@dataclass
class MarketState:
    """Complete market state for a match."""

    match_id: str
    market_id: str
    market_type: str  # MATCH_ODDS, etc.
    selections: dict[int, MarketSnapshot] = field(default_factory=dict)
    in_play: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_selection(self, name: str) -> Optional[MarketSnapshot]:
        """Find a selection by team name."""
        for snap in self.selections.values():
            if snap.selection_name == name:
                return snap
        return None


class ExchangeFeed(ABC):
    """Abstract base class for exchange data feeds."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the exchange."""

    @abstractmethod
    def subscribe_market(self, market_id: str) -> None:
        """Subscribe to price updates for a market."""

    @abstractmethod
    def get_market_state(self, market_id: str) -> Optional[MarketState]:
        """Get current market state."""

    @abstractmethod
    def on_price_update(self, callback: Callable[[MarketState], None]) -> None:
        """Register a callback for price updates."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""


class SimulatedExchangeFeed(ExchangeFeed):
    """Simulated exchange feed for backtesting and paper trading.

    Generates realistic price movements based on model probabilities
    and historical volatility patterns.
    """

    def __init__(self, commission_rate: float = 0.02):
        self._markets: dict[str, MarketState] = {}
        self._callbacks: list[Callable[[MarketState], None]] = []
        self._commission_rate = commission_rate
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info("Simulated exchange feed connected")

    def subscribe_market(self, market_id: str) -> None:
        if market_id not in self._markets:
            self._markets[market_id] = MarketState(
                match_id="", market_id=market_id, market_type="MATCH_ODDS"
            )
        logger.info("Subscribed to simulated market %s", market_id)

    def get_market_state(self, market_id: str) -> Optional[MarketState]:
        return self._markets.get(market_id)

    def on_price_update(self, callback: Callable[[MarketState], None]) -> None:
        self._callbacks.append(callback)

    def disconnect(self) -> None:
        self._connected = False
        self._markets.clear()
        logger.info("Simulated exchange feed disconnected")

    def inject_prices(
        self,
        market_id: str,
        match_id: str,
        selections: dict[str, float],
        volume: float = 50_000.0,
    ) -> None:
        """Inject prices into the simulated market.

        Args:
            market_id: Market identifier
            match_id: Match identifier
            selections: Dict of {team_name: implied_probability}
            volume: Simulated matched volume
        """
        state = self._markets.get(market_id)
        if not state:
            state = MarketState(
                match_id=match_id, market_id=market_id, market_type="MATCH_ODDS"
            )
            self._markets[market_id] = state

        state.match_id = match_id
        state.in_play = True
        state.timestamp = datetime.utcnow()

        for idx, (name, prob) in enumerate(selections.items()):
            if prob <= 0 or prob >= 1:
                continue
            fair_price = 1.0 / prob
            # Add realistic spread (1-2 ticks)
            spread = max(0.02, fair_price * 0.005)
            back_price = round(fair_price - spread / 2, 2)
            lay_price = round(fair_price + spread / 2, 2)

            snap = MarketSnapshot(
                market_id=market_id,
                selection_id=idx + 1,
                selection_name=name,
                timestamp=state.timestamp,
                back_prices=[
                    PriceLevel(back_price, volume * 0.3),
                    PriceLevel(round(back_price - 0.02, 2), volume * 0.2),
                    PriceLevel(round(back_price - 0.04, 2), volume * 0.1),
                ],
                lay_prices=[
                    PriceLevel(lay_price, volume * 0.3),
                    PriceLevel(round(lay_price + 0.02, 2), volume * 0.2),
                    PriceLevel(round(lay_price + 0.04, 2), volume * 0.1),
                ],
                last_price_traded=fair_price,
                total_matched=volume,
            )
            state.selections[idx + 1] = snap

        for cb in self._callbacks:
            cb(state)


class BetfairFeed(ExchangeFeed):
    """Live Betfair Streaming API feed.

    Requires betfairlightweight library and valid Betfair credentials.
    This is a placeholder for live deployment - the SimulatedExchangeFeed
    is used for backtesting and paper trading.
    """

    def __init__(self, app_key: str, username: str, password: str, cert_path: str):
        self._app_key = app_key
        self._username = username
        self._password = password
        self._cert_path = cert_path
        self._client = None
        self._callbacks: list[Callable[[MarketState], None]] = []

    def connect(self) -> None:
        try:
            import betfairlightweight

            self._client = betfairlightweight.APIClient(
                username=self._username,
                password=self._password,
                app_key=self._app_key,
                certs=self._cert_path,
            )
            self._client.login()
            logger.info("Connected to Betfair API")
        except ImportError:
            raise RuntimeError(
                "betfairlightweight is required for live Betfair feed. "
                "Install with: pip install betfairlightweight"
            )

    def subscribe_market(self, market_id: str) -> None:
        logger.info("Betfair market subscription: %s (not yet implemented)", market_id)

    def get_market_state(self, market_id: str) -> Optional[MarketState]:
        logger.warning("Live market state not yet implemented")
        return None

    def on_price_update(self, callback: Callable[[MarketState], None]) -> None:
        self._callbacks.append(callback)

    def disconnect(self) -> None:
        if self._client:
            self._client.logout()
        logger.info("Disconnected from Betfair API")
