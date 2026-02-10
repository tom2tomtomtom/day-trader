#!/usr/bin/env python3
"""
RL TRADING ENVIRONMENT — Gym-compatible environment for PPO agent.

Observation: Feature vector (70 features) + position state (3 fields)
Action: Discrete(3) — 0=hold, 1=buy, 2=sell
Reward: Log returns with drawdown penalty

The RL agent is trained OFFLINE on historical data and deployed as a
signal source in the signal ensemble (weight 0.15-0.20).
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Feature count from FeatureVector.to_ml_array()
N_FEATURES = 70
# Extra obs: position_held (-1/0/1), unrealized_pnl_pct, bars_in_position
N_POSITION_STATE = 3
OBS_DIM = N_FEATURES + N_POSITION_STATE


class TradingEnv(gym.Env):
    """
    Gym-compatible trading environment for RL training.

    State: [feature_vector, position_held, unrealized_pnl_pct, bars_in_position]
    Actions: 0=hold, 1=buy(long), 2=sell(short)
    Reward: log return per step with drawdown penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,       # (T, N_FEATURES) array of feature vectors
        prices: np.ndarray,          # (T,) array of prices
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_drawdown_penalty: float = 2.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        assert features.shape[0] == prices.shape[0], "features and prices must have same length"
        assert features.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {features.shape[1]}"

        self.features = features
        self.prices = prices
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_drawdown_penalty = max_drawdown_penalty
        self.render_mode = render_mode

        self.n_steps = len(prices)

        # Action space: hold, buy, sell
        self.action_space = spaces.Discrete(3)

        # Observation space: features + position state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # State
        self.current_step = 0
        self.position = 0          # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.bars_in_position = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_in_position = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        current_price = self.prices[self.current_step]
        reward = 0.0
        info = {}

        # Execute action
        if action == 1 and self.position != 1:  # Buy
            if self.position == -1:  # Close short first
                pnl = (self.entry_price - current_price) / self.entry_price
                reward += pnl - self.transaction_cost
                self.balance *= (1 + pnl - self.transaction_cost)
            # Open long
            self.position = 1
            self.entry_price = current_price
            self.bars_in_position = 0
            reward -= self.transaction_cost  # Entry cost

        elif action == 2 and self.position != -1:  # Sell
            if self.position == 1:  # Close long first
                pnl = (current_price - self.entry_price) / self.entry_price
                reward += pnl - self.transaction_cost
                self.balance *= (1 + pnl - self.transaction_cost)
            # Open short
            self.position = -1
            self.entry_price = current_price
            self.bars_in_position = 0
            reward -= self.transaction_cost

        elif action == 0:  # Hold
            if self.position != 0:
                self.bars_in_position += 1
                # Mark-to-market reward
                if self.position == 1:
                    step_return = (current_price - self.prices[max(0, self.current_step - 1)]) / self.prices[max(0, self.current_step - 1)]
                else:
                    step_return = (self.prices[max(0, self.current_step - 1)] - current_price) / self.prices[max(0, self.current_step - 1)]
                reward += step_return

        # Update balance for mark-to-market
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                unrealized = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized = (self.entry_price - current_price) / self.entry_price
            current_equity = self.initial_balance * (1 + unrealized)
        else:
            current_equity = self.balance

        # Drawdown penalty
        self.peak_balance = max(self.peak_balance, current_equity)
        drawdown = (self.peak_balance - current_equity) / self.peak_balance
        if drawdown > 0.05:  # Penalty kicks in above 5% drawdown
            reward -= drawdown * self.max_drawdown_penalty

        # Use log returns for better gradient properties
        if reward != 0:
            reward = np.sign(reward) * np.log1p(abs(reward))

        # Advance
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Force close at end
        if terminated and self.position != 0:
            if self.position == 1:
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            reward += pnl - self.transaction_cost

        obs = self._get_obs()
        info["balance"] = float(current_equity)
        info["position"] = self.position
        info["drawdown"] = drawdown

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        features = self.features[min(self.current_step, self.n_steps - 1)]

        # Position state
        unrealized_pnl_pct = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self.prices[min(self.current_step, self.n_steps - 1)]
            if self.position == 1:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            else:
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100

        position_state = np.array([
            float(self.position),
            unrealized_pnl_pct,
            float(self.bars_in_position),
        ], dtype=np.float32)

        return np.concatenate([features.astype(np.float32), position_state])


def build_training_env(
    prices: List[float],
    features_list: List[np.ndarray],
    initial_balance: float = 100000.0,
) -> TradingEnv:
    """Build a TradingEnv from price and feature data."""
    prices_arr = np.array(prices, dtype=np.float64)
    features_arr = np.array(features_list, dtype=np.float64)
    return TradingEnv(
        features=features_arr,
        prices=prices_arr,
        initial_balance=initial_balance,
    )
