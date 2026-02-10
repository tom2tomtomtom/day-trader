#!/usr/bin/env python3
"""
RL AGENT — PPO agent for trading signal generation.

Trained offline on historical data via stable-baselines3.
Deployed as a signal source in the signal ensemble with weight 0.15-0.20.

Feature-gated via FeatureFlags.rl_agent_enabled.
"""

import logging
import io
import base64
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

from .config import get_feature_flags
from .db import get_db
from .feature_engine import FeatureVector

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Import SB3 lazily to avoid pulling PyTorch on every import
_sb3_available = None


def _check_sb3():
    global _sb3_available
    if _sb3_available is None:
        try:
            from stable_baselines3 import PPO
            _sb3_available = True
        except ImportError:
            _sb3_available = False
    return _sb3_available


@dataclass
class RLPrediction:
    """Output from RL agent."""
    action: int           # 0=hold, 1=buy, 2=sell
    direction: float      # -1 to +1
    confidence: float     # 0-1
    using_rl: bool


class RLAgent:
    """PPO-based trading agent.

    Wraps stable-baselines3 PPO for training and inference.
    Model stored in Supabase for Railway persistence.
    """

    def __init__(self):
        self.model = None
        self.model_version: int = 0
        self.training_reward: float = 0.0
        self._load_model()

    def predict(self, features: FeatureVector, position: int = 0) -> RLPrediction:
        """Predict action from feature vector.

        Args:
            features: Current feature vector
            position: Current position (-1=short, 0=flat, 1=long)

        Returns:
            RLPrediction with action, direction, confidence
        """
        flags = get_feature_flags()
        if not flags.rl_agent_enabled or self.model is None:
            return RLPrediction(action=0, direction=0.0, confidence=0.0, using_rl=False)

        try:
            # Build observation: features + position state
            feature_arr = features.to_ml_array()
            position_state = np.array([float(position), 0.0, 0.0], dtype=np.float32)
            obs = np.concatenate([feature_arr.astype(np.float32), position_state])

            action, _states = self.model.predict(obs, deterministic=True)
            action = int(action)

            # Get action probabilities for confidence
            from stable_baselines3.common.utils import obs_as_tensor
            import torch

            obs_tensor = obs_as_tensor(obs.reshape(1, -1), self.model.policy.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]

            confidence = float(probs[action])

            # Map action to direction
            direction_map = {0: 0.0, 1: 1.0, 2: -1.0}
            direction = direction_map[action]

            return RLPrediction(
                action=action,
                direction=direction,
                confidence=round(confidence, 4),
                using_rl=True,
            )
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            return RLPrediction(action=0, direction=0.0, confidence=0.0, using_rl=False)

    def train(self, prices: list, features_list: list, total_timesteps: int = 50000) -> bool:
        """Train PPO agent on historical data.

        Args:
            prices: List of prices (T,)
            features_list: List of feature arrays (T, N_FEATURES)
            total_timesteps: Training steps

        Returns True if training succeeded.
        """
        if not _check_sb3():
            logger.warning("stable-baselines3 not available — cannot train RL agent")
            return False

        from stable_baselines3 import PPO
        from .rl_env import TradingEnv
        import numpy as np

        if len(prices) < 100 or len(features_list) < 100:
            logger.warning(f"Insufficient data for RL training: {len(prices)} bars")
            return False

        try:
            prices_arr = np.array(prices, dtype=np.float64)
            features_arr = np.array(features_list, dtype=np.float64)

            env = TradingEnv(features=features_arr, prices=prices_arr)

            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=0,
            )

            logger.info(f"Training PPO agent for {total_timesteps} steps on {len(prices)} bars...")
            self.model.learn(total_timesteps=total_timesteps)

            # Evaluate
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                done = terminated or truncated

            self.training_reward = total_reward
            self.model_version += 1

            # Save
            self._save_model()

            logger.info(
                f"RL agent trained v{self.model_version}: "
                f"reward={total_reward:.4f} "
                f"final_balance={info.get('balance', 0):.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"RL training failed: {e}")
            return False

    def _save_model(self):
        """Save model to disk and Supabase."""
        if self.model is None:
            return

        # Save to disk
        try:
            model_path = MODELS_DIR / "rl_ppo_model"
            self.model.save(str(model_path))
            meta = {
                "version": self.model_version,
                "training_reward": self.training_reward,
            }
            (MODELS_DIR / "rl_meta.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.error(f"Failed to save RL model to disk: {e}")

        # Save to Supabase
        try:
            model_path = MODELS_DIR / "rl_ppo_model.zip"
            if model_path.exists():
                model_bytes = model_path.read_bytes()
                b64 = base64.b64encode(model_bytes).decode("utf-8")

                db = get_db()
                db.client.table("ml_model_artifacts").upsert({
                    "model_name": "rl_ppo",
                    "version": self.model_version,
                    "model_type": "ppo",
                    "artifacts": {"model_zip": b64},
                    "metrics": {
                        "training_reward": self.training_reward,
                    },
                }, on_conflict="model_name").execute()
                logger.info(f"RL model saved to Supabase v{self.model_version}")
        except Exception as e:
            logger.warning(f"Could not save RL model to Supabase: {e}")

    def _load_model(self):
        """Load model from Supabase or disk."""
        if not _check_sb3():
            return

        from stable_baselines3 import PPO

        # Try Supabase
        try:
            db = get_db()
            result = db.client.table("ml_model_artifacts").select("*").eq(
                "model_name", "rl_ppo"
            ).limit(1).execute()

            if result.data:
                row = result.data[0]
                artifacts = row.get("artifacts", {})
                if "model_zip" in artifacts:
                    model_bytes = base64.b64decode(artifacts["model_zip"])
                    model_path = MODELS_DIR / "rl_ppo_model.zip"
                    model_path.write_bytes(model_bytes)
                    self.model = PPO.load(str(model_path))
                    self.model_version = row.get("version", 1)
                    logger.info(f"Loaded RL model from Supabase v{self.model_version}")
                    return
        except Exception as e:
            logger.debug(f"Could not load RL model from Supabase: {e}")

        # Try disk
        try:
            model_path = MODELS_DIR / "rl_ppo_model.zip"
            meta_path = MODELS_DIR / "rl_meta.json"
            if model_path.exists():
                self.model = PPO.load(str(model_path))
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self.model_version = meta.get("version", 1)
                    self.training_reward = meta.get("training_reward", 0)
                logger.info(f"Loaded RL model from disk v{self.model_version}")
        except Exception as e:
            logger.debug(f"Could not load RL model from disk: {e}")


# Singleton
_agent: Optional[RLAgent] = None


def get_rl_agent() -> RLAgent:
    global _agent
    if _agent is None:
        _agent = RLAgent()
    return _agent
