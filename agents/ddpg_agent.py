"""
Dynamic ART-DRL — DDPG Agent
===============================
Deep Deterministic Policy Gradient agent using Stable-Baselines3.

Action space: Box(-1, 1, shape=(1,))  — continuous
    -1 = full sell, 0 = hold, +1 = full buy

Key features:
    • Deterministic actor-critic architecture
    • Off-policy replay buffer
    • Ornstein–Uhlenbeck (OU) action noise for exploration
    • Soft target-network updates (τ)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from agents.base_agent import BaseAgent
from config import settings


class DDPGAgent(BaseAgent):
    """DDPG agent for continuous-action commodity trading.

    Parameters
    ----------
    env : gymnasium.Env | None
        Environment to bind at construction.
    learning_rate : float
        Adam learning rate (shared actor-critic by default).
    buffer_size : int
        Replay buffer capacity.
    batch_size : int
        Mini-batch size for gradient updates.
    tau : float
        Soft-update coefficient for target networks.
    gamma : float
        Discount factor.
    noise_sigma : float
        Standard deviation of OU noise.
    net_arch : list[int] | None
        Hidden-layer sizes for actor and critic MLPs.
    tensorboard_log : str | Path | None
        TensorBoard log directory.
    device : str
        PyTorch device string.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        learning_rate: float = settings.DDPG_LEARNING_RATE,
        buffer_size: int = settings.DDPG_BUFFER_SIZE,
        batch_size: int = settings.DDPG_BATCH_SIZE,
        tau: float = settings.DDPG_TAU,
        gamma: float = settings.GAMMA,
        noise_sigma: float = settings.DDPG_NOISE_SIGMA,
        net_arch: Optional[List[int]] = None,
        tensorboard_log: Optional[Union[str, Path]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="DDPG", is_discrete=False)

        self._hyperparams = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
        }
        self._noise_sigma = noise_sigma
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
            "DDPG hyperparams: lr={}, buffer={}, batch={}, τ={}, "
            "γ={}, OU_σ={}, arch={}",
            learning_rate, buffer_size, batch_size, tau,
            gamma, noise_sigma, self._net_arch,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _make_noise(self, action_dim: int) -> OrnsteinUhlenbeckActionNoise:
        """Create an Ornstein–Uhlenbeck noise process."""
        return OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(action_dim, dtype=np.float32),
            sigma=self._noise_sigma * np.ones(action_dim, dtype=np.float32),
        )

    def _build_model(self, env: gym.Env) -> None:
        """Instantiate the SB3 DDPG model."""
        action_dim = env.action_space.shape[0]
        action_noise = self._make_noise(action_dim)

        policy_kwargs = {
            "net_arch": self._net_arch,
        }

        self.model = DDPG(
            policy="MlpPolicy",
            env=env,
            **self._hyperparams,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self._tb_log,
            device=self._device,
            seed=self._seed,
            verbose=0,
        )
        logger.debug("DDPG SB3 model built | env={} OU_σ={}", env, self._noise_sigma)

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
        """Train the DDPG agent.

        Parameters
        ----------
        env : gymnasium.Env
            A ContinuousTradingEnv instance.
        total_timesteps : int
            Total environment interactions.
        callback : BaseCallback | None
            SB3 callback.
        """
        if self.model is None:
            self._build_model(env)
        else:
            self.model.set_env(env)

        logger.info("DDPG training start | timesteps={:,}", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
            **kwargs,
        )
        logger.info("DDPG training complete")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Select a continuous action.

        Returns
        -------
        action : np.ndarray
            Action array of shape ``(1,)`` in ``[-1, 1]``.
        info : dict | None
        """
        if self.model is None:
            raise RuntimeError("Model not initialised. Call train() or load() first.")

        action, _states = self.model.predict(
            observation, deterministic=deterministic,
        )
        return action, None

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info("DDPG model saved → {}", save_path)

    def load(
        self, path: Union[str, Path], env: Optional[gym.Env] = None,
    ) -> None:
        """Load a saved DDPG model."""
        load_path = Path(path)
        self.model = DDPG.load(str(load_path), env=env, device=self._device)
        logger.info("DDPG model loaded ← {}", load_path)
