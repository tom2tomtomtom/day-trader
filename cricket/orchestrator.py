"""
Cricket Trading Engine Orchestrator.

Main entry point that coordinates the full pipeline:
Data Ingestion → State Engine → Pricing Model → Signal Generator → Execution

Supports three modes:
1. Backtest: Replay historical data to validate strategy
2. Paper Trade: Process live data with simulated execution
3. Live: Real execution on Betfair exchange (future)

Usage:
    python -m cricket.orchestrator --backtest --data-dir data/cricsheet/t20s/
    python -m cricket.orchestrator --paper
    python -m cricket.orchestrator --live
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from cricket.config import EngineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cricket.orchestrator")


def run_backtest(
    config: EngineConfig,
    data_dir: str,
    match_format: Optional[str] = None,
    max_matches: Optional[int] = None,
    enable_directional: bool = True,
    enable_mm: bool = True,
) -> None:
    """Run backtesting mode."""
    from cricket.backtest.backtester import Backtester

    logger.info("=" * 60)
    logger.info("CRICKET TRADING ENGINE - BACKTEST MODE")
    logger.info("=" * 60)
    logger.info("Data directory: %s", data_dir)
    logger.info("Format filter: %s", match_format or "all")
    logger.info("Max matches: %s", max_matches or "unlimited")
    logger.info("Directional: %s | Market-Making: %s", enable_directional, enable_mm)
    logger.info("Bankroll: £%.2f", config.bankroll)
    logger.info("")

    backtester = Backtester(
        config=config,
        enable_directional=enable_directional,
        enable_market_making=enable_mm,
    )

    summary = backtester.run_directory(
        data_dir,
        match_format=match_format,
        max_matches=max_matches,
    )

    print("\n" + summary.summary_str())


def run_paper_trading(config: EngineConfig) -> None:
    """Run paper trading mode (simulated execution on live data)."""
    from cricket.data.exchange_feed import SimulatedExchangeFeed
    from cricket.execution.engine import ExecutionEngine
    from cricket.execution.market_maker import MarketMaker
    from cricket.models.ensemble import EnsemblePricingModel
    from cricket.signals.signals import SignalGenerator
    from cricket.state.match_state import MatchStateEngine

    logger.info("=" * 60)
    logger.info("CRICKET TRADING ENGINE - PAPER TRADING MODE")
    logger.info("=" * 60)
    logger.info("Bankroll: £%.2f", config.bankroll)
    logger.info("Risk per trade: %.1f%%", config.risk.max_stake_pct * 100)
    logger.info("Max exposure per match: %.1f%%", config.risk.max_exposure_pct * 100)
    logger.info("")
    logger.info("Paper trading mode ready.")
    logger.info("Connect a data feed to begin processing live matches.")
    logger.info("")
    logger.info("Components initialized:")

    # Initialize all components
    exchange = SimulatedExchangeFeed()
    exchange.connect()

    pricing_model = EnsemblePricingModel(config=config.model)
    signal_gen = SignalGenerator(config.signal)
    execution = ExecutionEngine(
        risk_config=config.risk,
        bankroll=config.bankroll,
        paper_mode=True,
    )
    market_maker = MarketMaker(bankroll=config.bankroll)

    logger.info("  - Exchange feed: connected (simulated)")
    logger.info("  - Pricing model: ensemble (stat + xgboost heuristic)")
    logger.info("  - Signal generator: 4 signal types active")
    logger.info("  - Execution engine: paper mode")
    logger.info("  - Market maker: active")
    logger.info("")
    logger.info(
        "Waiting for match data... (use --backtest mode for historical testing)"
    )


def run_demo(config: EngineConfig) -> None:
    """Run a quick demo with synthetic match data to verify the pipeline."""
    from cricket.data.ball_event import BallEvent, MatchInfo
    from cricket.data.exchange_feed import SimulatedExchangeFeed
    from cricket.execution.engine import ExecutionEngine
    from cricket.execution.market_maker import MarketMaker
    from cricket.models.ensemble import EnsemblePricingModel
    from cricket.signals.signals import SignalGenerator
    from cricket.state.match_state import MatchStateEngine

    import random

    logger.info("=" * 60)
    logger.info("CRICKET TRADING ENGINE - DEMO MODE")
    logger.info("=" * 60)

    # Synthetic match
    match_info = MatchInfo(
        match_id="demo_001",
        format="t20",
        team_a="Thunder",
        team_b="Strikers",
        venue="Demo Stadium",
        venue_avg_first_innings_score=165.0,
        team_a_elo=1550,
        team_b_elo=1480,
    )

    # Initialize pipeline
    state_engine = MatchStateEngine(match_info)
    pricing = EnsemblePricingModel(config=config.model, match_format="t20")
    signal_gen = SignalGenerator(config.signal)
    execution = ExecutionEngine(
        risk_config=config.risk, bankroll=config.bankroll, paper_mode=True
    )
    market_maker = MarketMaker(bankroll=config.bankroll)
    exchange = SimulatedExchangeFeed()
    exchange.connect()

    market_id = "demo_mkt"
    exchange.subscribe_market(market_id)

    # Simulate 40 balls (roughly 6-7 overs)
    score = 0
    wickets = 0
    balls = 0

    logger.info("\n--- Simulating T20 match: Thunder vs Strikers ---\n")

    for over_num in range(7):
        for ball_num in range(1, 7):
            if wickets >= 10:
                break

            # Random ball outcome
            r = random.random()
            runs = 0
            is_wicket = False
            if r < 0.30:
                runs = 0  # Dot
            elif r < 0.55:
                runs = 1
            elif r < 0.70:
                runs = 2
            elif r < 0.80:
                runs = 4
            elif r < 0.85:
                runs = 6
            elif r < 0.93:
                runs = 1
            else:
                is_wicket = True

            score += runs
            balls += 1
            if is_wicket:
                wickets += 1

            event = BallEvent(
                match_id="demo_001",
                innings=1,
                over=over_num,
                ball=ball_num,
                batting_team="Thunder",
                bowling_team="Strikers",
                striker=f"Bat_{wickets+1}",
                non_striker=f"Bat_{wickets+2}",
                bowler=f"Bowl_{over_num % 4 + 1}",
                runs_off_bat=runs,
                total_runs=runs,
                is_wicket=is_wicket,
                is_boundary_four=(runs == 4),
                is_boundary_six=(runs == 6),
                cumulative_score=score,
                cumulative_wickets=wickets,
                cumulative_overs=over_num + ball_num / 10.0,
            )

            # Process through pipeline
            match_state = state_engine.process_ball(event)
            features = state_engine.get_features()
            prediction = pricing.predict(features, batting_is_team_a=True)

            # Inject market prices
            noise = random.gauss(0, 0.03)
            mkt_prob_a = max(0.05, min(0.95, prediction.team_a_win_prob + noise))
            exchange.inject_prices(
                market_id, "demo_001",
                {"Thunder": mkt_prob_a, "Strikers": 1.0 - mkt_prob_a},
            )
            market_state = exchange.get_market_state(market_id)

            # Generate signals
            signals = signal_gen.generate_signals(match_state, market_state, prediction)

            # Market-making quotes
            quotes = market_maker.generate_quotes(
                "demo_001", prediction, "Thunder", "Strikers"
            )

            # Execute signals
            for sig in signals:
                pos = execution.execute_signal(sig)
                if pos:
                    logger.info(
                        "  SIGNAL: %s %s %s @ %.2f (edge: %.1f%%)",
                        sig.signal_type.value,
                        sig.direction.value.upper(),
                        sig.selection_name,
                        sig.market_odds,
                        sig.edge_pct,
                    )

            if is_wicket:
                logger.info(
                    "  WICKET! %d/%d after %d.%d overs | Model: Thunder %.1f%% | Market: %.1f%%",
                    score, wickets, over_num, ball_num,
                    prediction.team_a_win_prob * 100,
                    mkt_prob_a * 100,
                )

        if over_num % 2 == 0 and over_num > 0:
            logger.info(
                "  Over %d: %d/%d (RR: %.1f) | Thunder win prob: %.1f%%",
                over_num, score, wickets,
                score / (balls / 6) if balls > 0 else 0,
                prediction.team_a_win_prob * 100,
            )

    # Summary
    print("\n" + "=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"Score: Thunder {score}/{wickets} after {balls//6}.{balls%6} overs")
    print(f"Model win probability: {prediction.team_a_win_prob*100:.1f}%")
    print()

    perf = execution.get_performance_summary()
    print(f"Directional Trading:")
    print(f"  Trades: {perf['total_trades']}")
    print(f"  P&L: £{perf['total_pnl']:.2f}")
    print(f"  Win Rate: {perf['win_rate']*100:.0f}%")
    print()

    mm_perf = market_maker.get_performance()
    print(f"Market Making:")
    print(f"  Trades: {mm_perf['total_trades']}")
    print(f"  Spread Captured: £{mm_perf['total_spread_captured']:.2f}")
    print(f"  Volume: £{mm_perf['total_volume']:.2f}")
    print()

    exchange.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cricket Exchange AI Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cricket.orchestrator --demo
  python -m cricket.orchestrator --backtest --data-dir data/cricsheet/t20s/
  python -m cricket.orchestrator --paper
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backtest", action="store_true", help="Run backtesting mode")
    mode.add_argument("--paper", action="store_true", help="Run paper trading mode")
    mode.add_argument("--live", action="store_true", help="Run live trading mode")
    mode.add_argument("--demo", action="store_true", help="Run demo with synthetic data")

    parser.add_argument("--data-dir", type=str, default="data/cricsheet", help="Directory for Cricsheet CSV files")
    parser.add_argument("--format", type=str, choices=["t20", "odi", "test"], help="Match format filter")
    parser.add_argument("--max-matches", type=int, help="Maximum matches to backtest")
    parser.add_argument("--bankroll", type=float, default=10000.0, help="Starting bankroll in GBP")
    parser.add_argument("--no-directional", action="store_true", help="Disable directional signals")
    parser.add_argument("--no-mm", action="store_true", help="Disable market-making")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = EngineConfig.from_env()
    config = EngineConfig(
        exchange=config.exchange,
        cricket_data=config.cricket_data,
        risk=config.risk,
        model=config.model,
        signal=config.signal,
        bankroll=args.bankroll,
        paper_trading=not args.live,
    )

    if args.demo:
        run_demo(config)
    elif args.backtest:
        run_backtest(
            config,
            data_dir=args.data_dir,
            match_format=args.format,
            max_matches=args.max_matches,
            enable_directional=not args.no_directional,
            enable_mm=not args.no_mm,
        )
    elif args.paper:
        run_paper_trading(config)
    elif args.live:
        logger.error("Live trading requires Betfair credentials and is not yet implemented.")
        logger.error("Use --paper mode for simulated trading or --backtest for historical testing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
