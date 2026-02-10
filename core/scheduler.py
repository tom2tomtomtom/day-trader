#!/usr/bin/env python3
"""
AUTONOMOUS SCHEDULER — Runs the trading system on schedule.

Schedules:
- Stocks: every 5 min during market hours (9:30 AM - 4:00 PM ET)
- Crypto: every 15 min, 24/7
- ML retrain: daily at midnight ET
- Intelligence briefing: 9:00 AM + 4:30 PM ET

Run as: python -m core.scheduler
"""

import time
import logging
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from .config import get_config
from .db import get_db

logger = logging.getLogger(__name__)

# Eastern timezone offset (simplified — doesn't handle DST transitions mid-run)
ET_OFFSET = timedelta(hours=-5)


def _now_et() -> datetime:
    return datetime.now(timezone.utc) + ET_OFFSET


def _is_market_hours() -> bool:
    """Check if US stock market is open (9:30 AM - 4:00 PM ET, weekdays)."""
    now = _now_et()
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


# Stock universe
STOCK_UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD",
    "MSTR", "COIN",
]

# Crypto universe (24/7)
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD",
    "DOGE-USD", "SHIB-USD", "AVAX-USD",
]


def _fetch_price_data(symbols: List[str]) -> dict:
    """Fetch price data for symbols via yfinance."""
    import yfinance as yf
    price_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
            if not hist.empty and len(hist) >= 20:
                price_data[symbol] = {
                    "prices": hist["Close"].tolist(),
                    "highs": hist["High"].tolist(),
                    "lows": hist["Low"].tolist(),
                    "volumes": hist["Volume"].tolist(),
                }
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
    return price_data


def run_ml_retrain():
    """Retrain ML models if enough new data."""
    try:
        from .ml_pipeline import get_ml_pipeline
        pipeline = get_ml_pipeline()
        if pipeline.needs_retrain():
            logger.info("Retraining ML models...")
            success = pipeline.train()
            if success:
                logger.info("ML retrain complete")
            else:
                logger.info("ML retrain skipped (not enough data)")
    except Exception as e:
        logger.error(f"ML retrain failed: {e}")

    # Always run signal evaluation during nightly cycle
    try:
        from .signal_evaluator import SignalEvaluator
        evaluator = SignalEvaluator()
        results = evaluator.evaluate()
        if results:
            logger.info(f"Signal evaluation: {len(results)} groups evaluated")
    except Exception as e:
        logger.error(f"Signal evaluation failed: {e}")


def run_intelligence_briefing():
    """Run full intelligence briefing."""
    try:
        from .orchestrator import TradingOrchestrator, DEFAULT_UNIVERSE
        logger.info("Running intelligence briefing...")
        orch = TradingOrchestrator()
        briefing = orch.run_intelligence_briefing(DEFAULT_UNIVERSE)
        logger.info(f"Briefing complete: {briefing.actionable_signals} actionable signals")
    except Exception as e:
        logger.error(f"Intelligence briefing failed: {e}")


class TradingScheduler:
    """Main scheduler that coordinates all automated tasks."""

    def __init__(self):
        self.cfg = get_config()
        self.running = True
        self._last_stock_scan = 0.0
        self._last_crypto_scan = 0.0
        self._last_ml_retrain = 0.0
        self._last_briefing_am = ""
        self._last_briefing_pm = ""
        self._loop_count = 0

        # Persistent trader instance (loads state from DB once)
        from .paper_trader import PaperTrader
        self._trader = PaperTrader(initial_capital=100000)

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Scheduler shutting down...")
        self.running = False

    def _run_stock_scan(self):
        """Run paper trading scan on stock universe."""
        logger.info("Running stock scan...")
        price_data = _fetch_price_data(STOCK_UNIVERSE)
        if not price_data:
            logger.warning("No stock price data fetched")
            return
        portfolio = self._trader.analyze_and_trade(STOCK_UNIVERSE, price_data)
        logger.info(
            f"Stock scan complete: ${portfolio.portfolio_value:,.2f} "
            f"({portfolio.total_return_pct:+.2f}%) | "
            f"{len(portfolio.positions)} open positions"
        )

    def _run_crypto_scan(self):
        """Run paper trading scan on crypto universe."""
        logger.info("Running crypto scan...")
        price_data = _fetch_price_data(CRYPTO_UNIVERSE)
        if not price_data:
            logger.warning("No crypto price data fetched")
            return
        portfolio = self._trader.analyze_and_trade(CRYPTO_UNIVERSE, price_data)
        logger.info(
            f"Crypto scan complete: ${portfolio.portfolio_value:,.2f} "
            f"({portfolio.total_return_pct:+.2f}%) | "
            f"{len(portfolio.positions)} open positions"
        )

    def run(self):
        """Main loop — runs forever until stopped."""
        logger.info("Apex Trader Scheduler started")
        logger.info(f"  Stock interval: {self.cfg.trading.stock_scan_interval}s")
        logger.info(f"  Crypto interval: {self.cfg.trading.crypto_scan_interval}s")
        logger.info(f"  Loaded {len(self._trader.positions)} existing positions")
        logger.info(f"  Portfolio cash: ${self._trader.cash:,.2f}")

        while self.running:
            try:
                now = time.time()
                now_et = _now_et()
                today = now_et.strftime("%Y-%m-%d")
                self._loop_count += 1

                # Heartbeat every 10 loops (~5 min)
                if self._loop_count % 10 == 0:
                    logger.info(
                        f"Heartbeat: loop {self._loop_count} | "
                        f"positions={len(self._trader.positions)} | "
                        f"cash=${self._trader.cash:,.2f} | "
                        f"ET={now_et.strftime('%H:%M')}"
                    )

                # Stock scan (during market hours)
                if _is_market_hours():
                    if now - self._last_stock_scan >= self.cfg.trading.stock_scan_interval:
                        self._run_stock_scan()
                        self._last_stock_scan = now

                # Crypto scan (24/7)
                if now - self._last_crypto_scan >= self.cfg.trading.crypto_scan_interval:
                    self._run_crypto_scan()
                    self._last_crypto_scan = now

                # ML retrain (daily at midnight ET)
                if now_et.hour == 0 and now - self._last_ml_retrain >= 82800:
                    run_ml_retrain()
                    self._last_ml_retrain = now

                # Morning briefing (9:00 AM ET)
                if now_et.hour == 9 and now_et.minute < 5 and self._last_briefing_am != today:
                    run_intelligence_briefing()
                    self._last_briefing_am = today

                # Closing briefing (4:30 PM ET)
                if now_et.hour == 16 and 30 <= now_et.minute < 35 and self._last_briefing_pm != today:
                    run_intelligence_briefing()
                    self._last_briefing_pm = today

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)

            # Sleep between checks
            time.sleep(30)

        logger.info("Scheduler stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    scheduler = TradingScheduler()
    scheduler.run()
