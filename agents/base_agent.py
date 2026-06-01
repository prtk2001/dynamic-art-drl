"""
Dynamic ART-DRL — Base Agent
==============================
Abstract base class for all Deep Reinforcement Learning agents.

Every concrete agent (DQN, PPO, DDPG, A2C) **must** inherit from
``BaseAgent`` and implement all abstract methods so that the
Dynamic Router can swap agents transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for DRL trading agents.

    Attributes
    ----------
    name : str
        Human-readable agent identifier (e.g. "DQN", "PPO").
    is_discrete : bool
        ``True`` if the agent uses a discrete action space.
    model : Any
        Underlying model object (set by subclasses).
    """

    def __init__(self, name: str, is_discrete: bool) -> None:
        self._name = name
        self._is_discrete = is_discrete
        self.model: Any = None
        logger.info("Agent '{}' initialised | discrete={}", name, is_discrete)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_discrete(self) -> bool:
        """Whether the agent operates on a discrete action space."""
        return self._is_discrete

    def get_name(self) -> str:
        """Return the human-readable agent name."""
        return self._name

    # ──────────────────────────────────────────────────────────────────────
    # Abstract Interface
    # ──────────────────────────────────────────────────────────────────────

    @abstractmethod
    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Train the agent on the given environment.

        Parameters
        ----------
        env : gymnasium.Env
            The trading environment instance.
        total_timesteps : int
            Total environment steps to train for.
        callback : optional
            Stable-Baselines3 callback or similar.
        **kwargs
            Additional arguments forwarded to the underlying learner.
        """
        ...

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Predict an action given a single observation.

        Parameters
        ----------
        observation : np.ndarray
            Current state vector.
        deterministic : bool
            If ``True``, use the greedy / mean action (no exploration).

        Returns
        -------
        action : Any
            The selected action (int for discrete, ndarray for continuous).
        info : dict | None
            Optional auxiliary information (e.g. Q-values, log-probs).
        """
        ...

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the agent's learned parameters to disk.

        Parameters
        ----------
        path : str | Path
            File path (without extension — SB3 adds ``.zip``).
        """
        ...

    @abstractmethod
    def load(self, path: Union[str, Path], env: Optional[gym.Env] = None) -> None:
        """Load a previously saved agent from disk.

        Parameters
        ----------
        path : str | Path
            File path used in ``save()``.
        env : gymnasium.Env | None
            Optional environment to attach (needed for continued training).
        """
        ...

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self._name}', "
            f"discrete={self._is_discrete})"
        )
