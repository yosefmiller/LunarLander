"""
Base Agent Abstract Class for Lunar Lander RL Agents

Provides a common interface for all reinforcement learning agents in the Lunar Lander
training platform. Ensures consistency across different algorithm implementations.

All agents must implement the abstract methods defined here.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Tuple, List

import numpy as np

from lunar_lander_env import LunarLanderEnv, EpisodeResult


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.

    Defines the common interface that all agents must implement to ensure
    consistency across the training platform.
    """

    @abstractmethod
    def act(self, state: Tuple|np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action to take given the current state of the environment.

        :param state: Current state of the environment
        :param evaluate: Whether to evaluate the agent and choose exploitation
        :return: action to take
        """
        pass

    @abstractmethod
    def train(self, env: LunarLanderEnv, episodes: int, chkpt_path: str, **kwargs) -> List[EpisodeResult]:
        """
        Train the agent on the given environment.

        Args:
            env: Environment instance or class (for vectorized training)
            episodes: Number of training episodes
            chkpt_path: Path to save checkpoints
            **kwargs: Additional algorithm-specific parameters
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state from checkpoint file.

        Args:
            path: Path to the checkpoint file
        """
        pass

    @abstractmethod
    def evaluate(self, env, episodes: int) -> List[EpisodeResult]:
        """
        Evaluate the trained agent on the environment.

        Args:
            env: Environment instance
            episodes: Number of evaluation episodes

        Returns:
            avg_rewards_per_interval: List of average rewards for each evaluation interval
            average_reward: Average reward over all episodes
            additional_metrics: Dictionary of any additional evaluation metrics
        """
        pass

    @abstractmethod
    def show_progress(self, env, episodes: int, save_gif: bool, gif_path: str) -> None:
        """
        Visualize agent performance using pygame.

        Args:
            env: Environment instance
            episodes: Number of episodes to visualize
            save_gif: Whether to save a GIF of the agent
            gif_path: Path to save the GIF of the agent
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent name/algorithm identifier."""
        pass

    @staticmethod
    def _calculate_stats(episode_history: List[EpisodeResult]) -> List[str]:
        avg_reward = np.mean([h['reward'] for h in episode_history])
        avg_duration = np.mean([h['steps'] for h in episode_history])
        avg_spent = np.mean([h['spent'] for h in episode_history])
        crashed = sum(h['crashed'] for h in episode_history)
        landed = sum(h['landed'] for h in episode_history)
        timeout = sum(h['timeout'] for h in episode_history)
        crash_counts = Counter(tuple(sorted(h['crash_reason'], key=lambda r: r.value))
                               for h in episode_history if not h.get('landed'))

        return [
            f"Avg Reward: {avg_reward:.2f}",
            f"Steps: {avg_duration:.1f}",
            f"Spent: {avg_spent:.2f} kg",
            f"Landed: {landed}",
            f"Crashed: {crashed}",
            f"Timeout: {timeout}",
            *[
                f"{','.join(reason.name for reason in sorted(combo, key=lambda x: x.value, reverse=True))}: {count}"
                for combo, count in sorted(
                    crash_counts.items(),
                    key=lambda x: sum(r.value for r in x[0]),
                    reverse=True
                ) if len(combo) > 0
              ]
        ]