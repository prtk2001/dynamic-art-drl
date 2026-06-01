"""
Dynamic ART-DRL — PPO Agent
==============================
Proximal Policy Optimisation agent using Stable-Baselines3.

Action space: Box(-1, 1, shape=(1,))  — continuous
    -1 = full sell, 0 = hold, +1 = full buy

Key features:
    • Clipped surrogate objective
    • Generalised Advantage Estimation (GAE)
    • Entropy coefficient for exploration
    • On-policy rollout buffer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agents.base_agent import BaseAgent
from config import settings


class PPOAgent(BaseAgent):
    """PPO agent for continuous-action commodity trading.

    Parameters
    ----------
    env : gymnasium.Env | None
        Environment to bind at construction.
    learning_rate : float
        Adam learning rate.
    n_steps : int
        Rollout length before each policy update.
    batch_size : int
        Mini-batch size for SGD updates.
    n_epochs : int
        Number of SGD passes per rollout.
    clip_range : float
        PPO clipping parameter ε.
    gae_lambda : float
        GAE λ for advantage estimation.
    ent_coef : float
        Entropy bonus coefficient.
    gamma : float
        Discount factor.
    net_arch : list[int] | None
        Shared MLP hidden-layer sizes; defaults to [256, 256].
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
        learning_rate: float = settings.PPO_LEARNING_RATE,
        n_steps: int = settings.PPO_N_STEPS,
        batch_size: int = settings.PPO_BATCH_SIZE,
        n_epochs: int = settings.PPO_N_EPOCHS,
        clip_range: float = settings.PPO_CLIP_RANGE,
        gae_lambda: float = settings.PPO_GAE_LAMBDA,
        ent_coef: float = settings.PPO_ENT_COEF,
        gamma: float = settings.GAMMA,
        net_arch: Optional[List[int]] = None,
        tensorboard_log: Optional[Union[str, Path]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="PPO", is_discrete=False)

        self._hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "clip_range": clip_range,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
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
            "PPO hyperparams: lr={}, n_steps={}, batch={}, epochs={}, "
            "clip={}, λ_GAE={}, ent_coef={}, γ={}",
            learning_rate, n_steps, batch_size, n_epochs,
            clip_range, gae_lambda, ent_coef, gamma,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _build_model(self, env: gym.Env) -> None:
        """Instantiate the SB3 PPO model."""
        policy_kwargs = {
            "net_arch": dict(pi=self._net_arch, vf=self._net_arch),
        }
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            **self._hyperparams,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self._tb_log,
            device=self._device,
            seed=self._seed,
            verbose=0,
        )
        logger.debug("PPO SB3 model built | env={}", env)

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
        """Train the PPO agent.

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

        logger.info("PPO training start | timesteps={:,}", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
            **kwargs,
        )
        logger.info("PPO training complete")

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
        logger.info("PPO model saved → {}", save_path)

    def load(
        self, path: Union[str, Path], env: Optional[gym.Env] = None,
    ) -> None:
        """Load a saved PPO model."""
        load_path = Path(path)
        self.model = PPO.load(str(load_path), env=env, device=self._device)
        logger.info("PPO model loaded ← {}", load_path)
