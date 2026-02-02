#!/usr/bin/env python3
"""
RISK ENGINE - Position sizing and risk management

Key components:
1. Kelly Criterion - Optimal bet size based on edge
2. Correlation Check - Don't over-concentrate
3. Max Drawdown Protection - Circuit breakers
4. Regime-Adjusted Stops - Wider in volatile markets

The #1 factor in trading success is NOT signal quality - it's risk management.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Position:
    symbol: str
    entry_price: float
    current_price: float
    quantity: float
    direction: str  # "long" or "short"
    entry_date: str
    stop_loss: float
    take_profit: float
    
    @property
    def pnl_pct(self) -> float:
        if self.direction == "long":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100
    
    @property
    def pnl_dollars(self) -> float:
        return self.pnl_pct / 100 * self.entry_price * self.quantity


@dataclass
class RiskMetrics:
    kelly_fraction: float
    max_position_pct: float
    recommended_shares: int
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    portfolio_heat: float
    can_trade: bool
    reason: str


class RiskEngine:
    """
    Manages all risk-related decisions
    """
    
    # Risk limits
    MAX_PORTFOLIO_HEAT = 0.20  # Max 20% of portfolio at risk
    MAX_SINGLE_POSITION = 0.10  # Max 10% in any single position
    MAX_CORRELATED_EXPOSURE = 0.25  # Max 25% in correlated assets
    MAX_DRAWDOWN_HALT = 0.15  # Stop trading if down 15%
    
    # Kelly parameters
    KELLY_FRACTION = 0.25  # Use 25% of Kelly (conservative)
    
    def __init__(self, portfolio_value: float = 100000):
        self.portfolio_value = portfolio_value
        self.starting_value = portfolio_value
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl: List[float] = []
    
    def calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b
        where:
        - b = ratio of win to loss (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1 - p)
        
        We use fractional Kelly (25%) for safety.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Fractional Kelly
        return max(0, min(kelly * self.KELLY_FRACTION, self.MAX_SINGLE_POSITION))
    
    def calculate_position_size(self, 
                                symbol: str,
                                entry_price: float,
                                stop_loss_pct: float,
                                confidence: float,
                                win_rate: float = 0.55,
                                avg_win_pct: float = 3.0,
                                avg_loss_pct: float = 2.0) -> RiskMetrics:
        """
        Calculate optimal position size considering all risk factors
        """
        # Check if we can trade at all
        current_drawdown = (self.starting_value - self.portfolio_value) / self.starting_value
        if current_drawdown >= self.MAX_DRAWDOWN_HALT:
            return RiskMetrics(
                kelly_fraction=0,
                max_position_pct=0,
                recommended_shares=0,
                stop_loss_price=0,
                take_profit_price=0,
                risk_reward_ratio=0,
                portfolio_heat=self._calculate_heat(),
                can_trade=False,
                reason=f"Drawdown halt: {current_drawdown:.1%} >= {self.MAX_DRAWDOWN_HALT:.1%}"
            )
        
        # Check portfolio heat
        current_heat = self._calculate_heat()
        if current_heat >= self.MAX_PORTFOLIO_HEAT:
            return RiskMetrics(
                kelly_fraction=0,
                max_position_pct=0,
                recommended_shares=0,
                stop_loss_price=0,
                take_profit_price=0,
                risk_reward_ratio=0,
                portfolio_heat=current_heat,
                can_trade=False,
                reason=f"Portfolio heat too high: {current_heat:.1%} >= {self.MAX_PORTFOLIO_HEAT:.1%}"
            )
        
        # Kelly calculation
        kelly = self.calculate_kelly(win_rate, avg_win_pct, avg_loss_pct)
        
        # Adjust by confidence
        adjusted_size = kelly * confidence
        
        # Cap at max position size
        position_pct = min(adjusted_size, self.MAX_SINGLE_POSITION)
        
        # Calculate dollars and shares
        position_dollars = self.portfolio_value * position_pct
        shares = int(position_dollars / entry_price)
        
        # Calculate stop and target prices
        stop_price = entry_price * (1 - stop_loss_pct)
        target_price = entry_price * (1 + avg_win_pct / 100)
        
        # Risk/reward ratio
        risk = entry_price - stop_price
        reward = target_price - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return RiskMetrics(
            kelly_fraction=round(kelly, 4),
            max_position_pct=round(position_pct, 4),
            recommended_shares=shares,
            stop_loss_price=round(stop_price, 2),
            take_profit_price=round(target_price, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            portfolio_heat=round(current_heat, 4),
            can_trade=True,
            reason=f"Position sized at {position_pct:.1%} of portfolio"
        )
    
    def _calculate_heat(self) -> float:
        """
        Calculate portfolio heat (% of portfolio at risk)
        Heat = sum of (position size * distance to stop loss)
        """
        if not self.positions:
            return 0
        
        total_risk = 0
        for pos in self.positions.values():
            position_value = pos.quantity * pos.current_price
            if pos.direction == "long":
                risk_pct = (pos.current_price - pos.stop_loss) / pos.current_price
            else:
                risk_pct = (pos.stop_loss - pos.current_price) / pos.current_price
            
            total_risk += position_value * risk_pct
        
        return total_risk / self.portfolio_value
    
    def check_correlation(self, symbol: str, sector: str = None) -> Tuple[bool, str]:
        """
        Check if adding this position would over-concentrate in correlated assets
        """
        # Simplified correlation check based on sectors
        CORRELATIONS = {
            "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "AMD", "META", "QQQ"],
            "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "COIN", "MSTR"],
            "finance": ["JPM", "BAC", "GS", "MS", "XLF"],
            "energy": ["XOM", "CVX", "XLE"],
        }
        
        # Find symbol's sector
        symbol_sector = None
        for sec, symbols in CORRELATIONS.items():
            if symbol in symbols:
                symbol_sector = sec
                break
        
        if not symbol_sector:
            return True, "No correlation concerns"
        
        # Calculate exposure to this sector
        sector_exposure = 0
        for pos_symbol, pos in self.positions.items():
            if pos_symbol in CORRELATIONS.get(symbol_sector, []):
                sector_exposure += pos.quantity * pos.current_price
        
        sector_exposure_pct = sector_exposure / self.portfolio_value
        
        if sector_exposure_pct >= self.MAX_CORRELATED_EXPOSURE:
            return False, f"Correlated exposure too high: {sector_exposure_pct:.1%} in {symbol_sector}"
        
        return True, f"Sector exposure OK: {sector_exposure_pct:.1%} in {symbol_sector}"
    
    def calculate_dynamic_stops(self, 
                               entry_price: float,
                               atr: float,
                               regime: str,
                               direction: str = "long") -> Tuple[float, float]:
        """
        Calculate stops and targets based on volatility and regime
        """
        # ATR multipliers by regime
        STOP_MULTIPLIERS = {
            "trending_up": 2.0,
            "trending_down": 1.5,
            "ranging": 1.5,
            "high_vol": 2.5,
            "crisis": 3.0,
        }
        
        TARGET_MULTIPLIERS = {
            "trending_up": 3.0,  # Let winners run in trends
            "trending_down": 1.5,
            "ranging": 2.0,
            "high_vol": 2.0,
            "crisis": 1.5,
        }
        
        stop_mult = STOP_MULTIPLIERS.get(regime, 2.0)
        target_mult = TARGET_MULTIPLIERS.get(regime, 2.0)
        
        if direction == "long":
            stop_loss = entry_price - (atr * stop_mult)
            take_profit = entry_price + (atr * target_mult)
        else:
            stop_loss = entry_price + (atr * stop_mult)
            take_profit = entry_price - (atr * target_mult)
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    def should_exit(self, position: Position, current_price: float, 
                   trailing_stop: bool = False) -> Tuple[bool, str]:
        """
        Check if position should be exited
        """
        position.current_price = current_price
        
        # Stop loss
        if position.direction == "long":
            if current_price <= position.stop_loss:
                return True, f"Stop loss hit at ${current_price}"
            if current_price >= position.take_profit:
                return True, f"Take profit hit at ${current_price}"
        else:
            if current_price >= position.stop_loss:
                return True, f"Stop loss hit at ${current_price}"
            if current_price <= position.take_profit:
                return True, f"Take profit hit at ${current_price}"
        
        # Trailing stop (optional)
        if trailing_stop:
            # Update stop to lock in profits
            if position.direction == "long" and position.pnl_pct > 2:
                new_stop = position.entry_price * 1.01  # Move to break-even + 1%
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
        
        return False, "Position open"
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_value = self.portfolio_value
        positions_value = sum(p.quantity * p.current_price for p in self.positions.values())
        cash = total_value - positions_value
        
        total_pnl = sum(p.pnl_dollars for p in self.positions.values())
        
        return {
            "portfolio_value": round(total_value, 2),
            "cash": round(cash, 2),
            "invested": round(positions_value, 2),
            "invested_pct": round(positions_value / total_value * 100, 1),
            "open_pnl": round(total_pnl, 2),
            "open_pnl_pct": round(total_pnl / total_value * 100, 2),
            "num_positions": len(self.positions),
            "portfolio_heat": round(self._calculate_heat() * 100, 2),
            "max_heat": self.MAX_PORTFOLIO_HEAT * 100,
            "drawdown": round((self.starting_value - total_value) / self.starting_value * 100, 2)
        }


# Test
if __name__ == "__main__":
    engine = RiskEngine(portfolio_value=100000)
    
    # Calculate position size for AAPL
    risk = engine.calculate_position_size(
        symbol="AAPL",
        entry_price=180,
        stop_loss_pct=0.02,
        confidence=0.7,
        win_rate=0.55,
        avg_win_pct=3.0,
        avg_loss_pct=2.0
    )
    
    print("=== Risk Metrics for AAPL ===")
    print(f"Kelly Fraction: {risk.kelly_fraction:.2%}")
    print(f"Position Size: {risk.max_position_pct:.2%}")
    print(f"Recommended Shares: {risk.recommended_shares}")
    print(f"Stop Loss: ${risk.stop_loss_price}")
    print(f"Take Profit: ${risk.take_profit_price}")
    print(f"Risk/Reward: {risk.risk_reward_ratio}")
    print(f"Can Trade: {risk.can_trade} - {risk.reason}")
    
    print("\n=== Portfolio Summary ===")
    summary = engine.get_portfolio_summary()
    for k, v in summary.items():
        print(f"{k}: {v}")
