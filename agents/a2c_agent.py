"""
Dynamic ART-DRL — A2C Agent
==============================
Advantage Actor-Critic agent using Stable-Baselines3.

Action space: Box(-1, 1, shape=(1,))  — continuous
    -1 = full sell, 0 = hold, +1 = full buy

Key features:
    • Synchronous advantage actor-critic
    • Generalised Advantage Estimation (GAE)
    • Entropy regularisation
    • Value-function coefficient for loss balancing
    • On-policy (short rollouts, n_steps=5 by default)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

from agents.base_agent import BaseAgent
from config import settings


class A2CAgent(BaseAgent):
    """A2C agent for continuous-action commodity trading.

    Parameters
    ----------
    env : gymnasium.Env | None
        Environment to bind at construction.
    learning_rate : float
        RMSProp / Adam learning rate.
    n_steps : int
        Number of steps per rollout before each update.
    gae_lambda : float
        GAE λ for advantage estimation.
    ent_coef : float
        Entropy bonus coefficient.
    vf_coef : float
        Value-function loss coefficient.
    gamma : float
        Discount factor.
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
        learning_rate: float = settings.A2C_LEARNING_RATE,
        n_steps: int = settings.A2C_N_STEPS,
        gae_lambda: float = settings.A2C_GAE_LAMBDA,
        ent_coef: float = settings.A2C_ENT_COEF,
        vf_coef: float = settings.A2C_VF_COEF,
        gamma: float = settings.GAMMA,
        net_arch: Optional[List[int]] = None,
        tensorboard_log: Optional[Union[str, Path]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="A2C", is_discrete=False)

        self._hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "gae_lambda": gae_lambda,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
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
            "A2C hyperparams: lr={}, n_steps={}, λ_GAE={}, "
            "ent_coef={}, vf_coef={}, γ={}, arch={}",
            learning_rate, n_steps, gae_lambda,
            ent_coef, vf_coef, gamma, self._net_arch,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _build_model(self, env: gym.Env) -> None:
        """Instantiate the SB3 A2C model."""
        policy_kwargs = {
            "net_arch": dict(pi=self._net_arch, vf=self._net_arch),
        }
        self.model = A2C(
            policy="MlpPolicy",
            env=env,
            **self._hyperparams,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self._tb_log,
            device=self._device,
            seed=self._seed,
            verbose=0,
        )
        logger.debug("A2C SB3 model built | env={}", env)

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
        """Train the A2C agent.

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

        logger.info("A2C training start | timesteps={:,}", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
            **kwargs,
        )
        logger.info("A2C training complete")

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
        logger.info("A2C model saved → {}", save_path)

    def load(
        self, path: Union[str, Path], env: Optional[gym.Env] = None,
    ) -> None:
        """Load a saved A2C model."""
        load_path = Path(path)
        self.model = A2C.load(str(load_path), env=env, device=self._device)
        logger.info("A2C model loaded ← {}", load_path)
