"""
Dynamic ART-DRL — DQN Agent
=============================
Deep Q-Network agent using Stable-Baselines3.

Action space: Discrete(5)
    0 = sell_all, 1 = sell_half, 2 = hold, 3 = buy_half, 4 = buy_all

Key features:
    • Experience replay buffer
    • Target network with soft updates (τ)
    • ε-greedy exploration with linear decay
    • Configurable MlpPolicy net architecture
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from agents.base_agent import BaseAgent
from config import settings


class DQNAgent(BaseAgent):
    """DQN agent for discrete-action commodity trading.

    Parameters
    ----------
    env : gymnasium.Env | None
        Environment to bind at construction (can also be passed to ``train``).
    learning_rate : float
        Adam learning rate.
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size for gradient updates.
    tau : float
        Soft-update coefficient for target network.
    exploration_fraction : float
        Fraction of total timesteps over which ε decays.
    exploration_final_eps : float
        Terminal ε value.
    gamma : float
        Discount factor.
    net_arch : list[int]
        Hidden-layer sizes for the Q-network MLP.
    tensorboard_log : str | Path | None
        TensorBoard log directory.
    device : str
        PyTorch device string ("auto", "cpu", "cuda").
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        learning_rate: float = settings.DQN_LEARNING_RATE,
        buffer_size: int = settings.DQN_BUFFER_SIZE,
        batch_size: int = settings.DQN_BATCH_SIZE,
        tau: float = settings.DQN_TAU,
        exploration_fraction: float = settings.DQN_EXPLORATION_FRACTION,
        exploration_final_eps: float = settings.DQN_EXPLORATION_FINAL_EPS,
        gamma: float = settings.GAMMA,
        net_arch: Optional[List[int]] = None,
        tensorboard_log: Optional[Union[str, Path]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="DQN", is_discrete=True)

        self._hyperparams = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "tau": tau,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "gamma": gamma,
        }
        self._net_arch = net_arch or [256, 256]
        try:
            import tensorboard
            self._tb_log = str(tensorboard_log or settings.TENSORBOARD_DIR)
        except ImportError:
            logger.warning("Tensorboard not installed. Disabling Tensorboard logging.")
            self._tb_log = None
        self._device = device
        self._seed = seed

        if env is not None:
            self._build_model(env)

        logger.info(
            "DQN hyperparams: lr={}, buffer={}, batch={}, τ={}, "
            "ε_frac={}, ε_final={}, γ={}, arch={}",
            learning_rate, buffer_size, batch_size, tau,
            exploration_fraction, exploration_final_eps, gamma,
            self._net_arch,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _build_model(self, env: gym.Env) -> None:
        """Instantiate the SB3 DQN model."""
        policy_kwargs = {"net_arch": self._net_arch}
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            **self._hyperparams,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self._tb_log,
            device=self._device,
            seed=self._seed,
            verbose=0,
        )
        logger.debug("DQN SB3 model built | env={}", env)

    # ──────────────────────────────────────────────────────────────────────
    # BaseAgent Interface
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        env: gym.Env,
        total_timesteps: int = settings.TRAINING_TIMESTEPS,
        callback: Optional[BaseCallback] = None,
        **kwargs: Any,
    ) -> None:
        """Train the DQN agent.

        Parameters
        ----------
        env : gymnasium.Env
            A DiscreteTradingEnv instance.
        total_timesteps : int
            Total environment interactions.
        callback : BaseCallback | None
            SB3 callback (e.g. EvalCallback, CheckpointCallback).
        """
        if self.model is None:
            self._build_model(env)
        else:
            self.model.set_env(env)

        logger.info("DQN training start | timesteps={:,}", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
            **kwargs,
        )
        logger.info("DQN training complete")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Select a discrete action.

        Returns
        -------
        action : int
            Action index in {0, 1, 2, 3, 4}.
        info : dict | None
            Contains ``q_values`` when available.
        """
        if self.model is None:
            raise RuntimeError("Model not initialised. Call train() or load() first.")

        action, _states = self.model.predict(
            observation, deterministic=deterministic,
        )
        return int(action), None

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk (SB3 adds .zip extension)."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info("DQN model saved → {}", save_path)

    def load(
        self, path: Union[str, Path], env: Optional[gym.Env] = None,
    ) -> None:
        """Load a saved DQN model."""
        load_path = Path(path)
        self.model = DQN.load(str(load_path), env=env, device=self._device)
        logger.info("DQN model loaded ← {}", load_path)
