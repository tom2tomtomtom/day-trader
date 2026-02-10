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
    "DOGE-USD", "SHIB-USD", "PEPE-USD",
]


def run_stock_scan():
    """Run paper trading scan on stock universe."""
    try:
        from .paper_trader import run_paper_trading
        logger.info("Running stock scan...")
        portfolio = run_paper_trading(STOCK_UNIVERSE)
        logger.info(
            f"Stock scan complete: ${portfolio.portfolio_value:,.2f} "
            f"({portfolio.total_return_pct:+.2f}%)"
        )
    except Exception as e:
        logger.error(f"Stock scan failed: {e}")


def run_crypto_scan():
    """Run paper trading scan on crypto universe."""
    try:
        from .paper_trader import run_paper_trading
        logger.info("Running crypto scan...")
        portfolio = run_paper_trading(CRYPTO_UNIVERSE)
        logger.info(
            f"Crypto scan complete: ${portfolio.portfolio_value:,.2f} "
            f"({portfolio.total_return_pct:+.2f}%)"
        )
    except Exception as e:
        logger.error(f"Crypto scan failed: {e}")


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

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Scheduler shutting down...")
        self.running = False

    def run(self):
        """Main loop — runs forever until stopped."""
        logger.info("Apex Trader Scheduler started")
        logger.info(f"  Stock interval: {self.cfg.trading.stock_scan_interval}s")
        logger.info(f"  Crypto interval: {self.cfg.trading.crypto_scan_interval}s")

        while self.running:
            now = time.time()
            now_et = _now_et()
            today = now_et.strftime("%Y-%m-%d")

            # Stock scan (during market hours)
            if _is_market_hours():
                if now - self._last_stock_scan >= self.cfg.trading.stock_scan_interval:
                    run_stock_scan()
                    self._last_stock_scan = now

            # Crypto scan (24/7)
            if now - self._last_crypto_scan >= self.cfg.trading.crypto_scan_interval:
                run_crypto_scan()
                self._last_crypto_scan = now

            # ML retrain (daily at midnight ET)
            if now_et.hour == 0 and now - self._last_ml_retrain >= 82800:  # ~23 hours guard
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
