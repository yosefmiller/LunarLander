import os
import random
import time
from collections import defaultdict, deque
from functools import reduce
from typing import List, Dict, Tuple

import numpy as np
import pygame
from tqdm import tqdm
from utils import plot_learning_curve, rand_argmax
from lunar_lander_env import (SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, RandomLunarLander,
    Renderer, EpisodeResult, LunarLanderState,  PAD_WIDTH)
from base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    def __init__(self, n_actions=4, n_step=1, alpha=0.2, gamma=0.995, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        """
        SARSA Agent for the Lunar Lander environment.

        How SARSA works:
        1. The agent observes the current state (discretized).
        2. It selects an action using an epsilon-greedy policy based on the Q-table.
        3. It takes the action, observes the reward and the next state.
        4. It selects the next action using the SAME epsilon-greedy policy (on-policy).
        5. It updates the Q-table using the SARSA update rule: Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S', A') - Q(S,A)]

        :param n_actions: Number of discrete actions available in the environment (e.g., 2 for main engine on/off, or 4 for main engine + left/right thrusters).
        :param n_step: Number of steps in trajectory to use for n-step SARSA updates. A value of 1 corresponds to standard SARSA, while higher values allow the agent to learn from longer sequences of experience.
        :param alpha: Learning rate (0 < alpha <= 1). Higher values mean the agent learns more from new experiences.
        :param gamma: Discount factor (0 <= gamma < 1). Higher values mean the agent values future rewards more.
        :param epsilon: Initial exploration rate (0 <= epsilon <= 1). Higher values mean the agent explores more at the start.
        :param epsilon_decay: Multiplicative factor for decaying epsilon after each episode (0 < epsilon_decay < 1). A value of 0.995 means epsilon will decay by 0.5% each episode.
        :param epsilon_min: Minimum exploration rate. Epsilon will not decay below this value, ensuring the agent continues to explore occasionally even after many episodes.
        """

        self.n_actions = n_actions
        self.n_step = n_step
        
        # Hyperparameters
        self.alpha = alpha                  # Learning rate
        self.gamma = gamma                  # Discount factor
        self.epsilon = epsilon              # Exploration rate
        self.epsilon_decay = epsilon_decay  # How fast exploration drops
        self.epsilon_min = epsilon_min

        # Q-table: dictionary mapping a state tuple -> numpy array of Q-values
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # --- Discretization Bins ---
        # np.digitize returns the bin index. Out of bounds values go into the first/last bins.
        # We increase resolution for y and vy as they are critical for the landing criteria.
        # x_bins = np.concatenate([
        #             np.array([-0.6, -0.3, -0.25]),
        #             np.linspace(-0.2, 0.2, 7),   # high resolution near pad
        #             np.array([0.25, 0.3, 0.6])
        #         ])
        # y_bins = np.concatenate([
        #             np.linspace(0, 0.01, 4),      # landing zone precision
        #             np.array([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.8])
        #         ])
        # vx_bins = np.array([-0.5, -0.3, -0.1, -0.02, 0.02, 0.1, 0.3, 0.5])
        # vy_bins = np.concatenate([
        #             np.array([-0.5, -0.3, -0.1, -0.05]),
        #             np.linspace(-0.05, 0.05, 5),   # critical landing region
        #             np.array([0.05, 0.1, 0.3, 0.5])
        #         ])
        # angle_bins = np.concatenate([
        #                 np.array([-1.0, -0.5, -0.2]),
        #                 np.linspace(-0.2, 0.2, 7),   # upright precision
        #                 np.array([0.2, 0.5, 1.0])
        #             ])
        # ang_vel_bins = np.array([-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0])

        x_bins = list(np.concatenate([
            -np.logspace(0, -0.8, base=10.0, num=5),
            [0.0],
            np.logspace(-0.8, 0, base=10.0, num=5),
        ]))
        y_bins = np.logspace(1, 3.4, base=3.0, num=6) / 50.0
        vx_bins = [-1.485, -0.993, -0.502, -0.011, 0.011, 0.502, 0.993, 1.485]
        vy_bins = [-0.633, -0.364, -0.256, -0.192, -0.138, -0.096, -0.04, 0.04, 0.096]
        angle_bins = [-1.329, -0.703, -0.36, -0.094, 0.094, 0.36, 0.703, 1.329]
        ang_vel_bins = [-0.327, -0.192, -0.096, -0.038, 0.038, 0.115, 0.25]

        self.bins = {
            'x': x_bins,
            'y': y_bins,
            'vx': vx_bins,
            'vy': vy_bins,
            'theta': angle_bins,
            'omega': ang_vel_bins
        }

    @property
    def name(self) -> str:
        return "SARSA"

    def get_number_of_states(self) -> int:
        """
        Calculates the total number of discrete states based on the defined bins for each state variable.
        :return: int
        """
        bin_sizes = [len(b) + 1 for b in self.bins.values()]  # +1 because digitize can return an index equal to len(bins)
        return reduce(lambda x, y: x * y, bin_sizes)

    def get_states(self) -> Dict[str, np.ndarray]:
        """
        Returns a list of discrete states based on the defined bins for each state variable.
        :return: dict
        """
        return {
            name: np.concatenate([[-np.inf], bins, [np.inf]])  # Include out-of-bounds bins
            for name, bins in self.bins.items()
        }

    def get_number_of_winning_states(self) -> int:
        """
        Calculates the number of winning states based on the defined bins for each state variable.
        :return: int
        """
        max_vy = 2.5 / 10
        max_vx = 2.0 / 10
        max_theta = 0.5
        max_y = 3.0 / 50
        max_x = PAD_WIDTH / 2 / 50

        # Calculate how many bins fall within the "safe" landing criteria
        safe_x_bins = sum(1 for b in self.bins['x'] if abs(b) <= max_x)
        safe_y_bins = sum(1 for b in self.bins['y'] if b <= max_y)
        safe_vx_bins = sum(1 for b in self.bins['vx'] if abs(b) <= max_vx)
        safe_vy_bins = sum(1 for b in self.bins['vy'] if abs(b) <= max_vy)
        safe_theta_bins = sum(1 for b in self.bins['theta'] if abs(b) <= max_theta)
        safe_omega_bins = len(self.bins['omega'])

        # The number of winning states is the product of the safe bins for each variable
        return safe_vy_bins * safe_vx_bins * safe_theta_bins * safe_y_bins * safe_x_bins * safe_omega_bins

    def discretize(self, state: LunarLanderState) -> Tuple[int, int, int, int, int, int]:
        """Converts the continuous state dataclass into a discrete tuple, ignoring fuel."""
        # Digitize returns an integer representing which bin the value falls into
        bin_x = np.digitize(state.x, self.bins['x'])
        bin_y = np.digitize(state.y, self.bins['y'])
        bin_vx = np.digitize(state.vx, self.bins['vx'])
        bin_vy = np.digitize(state.vy, self.bins['vy'])
        bin_theta = np.digitize(state.theta, self.bins['theta'])
        bin_omega = np.digitize(state.omega, self.bins['omega'])

        # Fuel and on_pad are explicitly ignored to reduce the state space
        return bin_x, bin_y, bin_vx, bin_vy, bin_theta, bin_omega

    def act(self, state_tuple, evaluate=False) -> int:
        """Selects an action using Epsilon-Greedy policy."""
        if not evaluate and np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions) # Explore
        
        # Exploit: return the action with the highest Q-value
        return rand_argmax(self.q_table[state_tuple])

    def update(self, trajectory: List[Tuple], next_state_tuple=None, next_action=None, done=False):
        """Applies the n-step SARSA update rule."""
        if not trajectory:
            return
        
        # Compute n-step return
        G = 0
        for i, (s, a, r) in enumerate(trajectory):
            G += (self.gamma ** i) * r
        if not done and next_state_tuple is not None and next_action is not None:
            G += (self.gamma ** self.n_step) * self.q_table[next_state_tuple][next_action]

        # Update the first state-action in trajectory
        s_update, a_update, _ = trajectory[0]
        current_q = self.q_table[s_update][a_update]
        new_q = current_q + self.alpha * (G - current_q)
        self.q_table[s_update][a_update] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(self.q_table, path)

    def load(self, path="SARSA_Agent_checkpoints/agent1/best_qtable_values.npy") -> SARSAAgent:
        qtable = np.load(path, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), qtable)
        return self

    def train(self,
              env: LunarLanderEnv,
              episodes=1000, 
              agent_type='SARSA', 
              chkpt_path="SARSA_Agent_checkpoints/agent1", 
              debug=False,
              logging_rate=500) -> List[EpisodeResult]:
        """Trains the agent on the given environment and returns episode history."""
        episode_history: List[EpisodeResult] = []
        total_bins = self.get_number_of_states()

        # Save Q table periodically
        os.makedirs(chkpt_path, exist_ok=True)
        ckpt_interval = max(1, episodes // 5)

        if debug:
            print(f"Training for {episodes} episodes...")

        # Training duration
        start = time.time()

        for ep in tqdm(range(episodes), disable=debug):
            state = env.reset()
            state_tuple = self.discretize(state)
            action = self.act(state_tuple)

            total_reward = 0.0
            done = False
            trajectory = deque(maxlen=self.n_step)  # Buffer for n-step

            while not done:
                # Step the environment
                next_state, reward, done, result = env.step(action)
                next_state_tuple = self.discretize(next_state)
                next_action = self.act(next_state_tuple)

                # Store in trajectory
                trajectory.append((state_tuple, action, reward))

                # If trajectory is full or episode ended, update
                if len(trajectory) == self.n_step or done:
                    # Update using n-step
                    self.update(list(trajectory), next_state_tuple, next_action, done)
                    trajectory.popleft()  # Remove the oldest experience to make room for new ones

                # Progress to next step
                state_tuple = next_state_tuple
                action = next_action
                total_reward += reward

            # Flush remaining transitions at episode end
            while trajectory:
                # noinspection PyUnboundLocalVariable
                self.update(list(trajectory), next_state_tuple, next_action, done=True)
                trajectory.popleft()

            self.decay_epsilon()

            # Save this episode
            # noinspection PyUnboundLocalVariable
            result['reward'] = total_reward
            episode_history.append(result)

            # Log stats
            if debug and (ep + 1) % logging_rate == 0:
                coverage = (len(self.q_table) / total_bins) * 100
                stats = [
                    f"Epsilon: {self.epsilon:.3f}",
                    f"Q-Table Coverage: {coverage:.2f}%",
                    *self._calculate_stats(episode_history[-logging_rate:])
                ]
                print(f"Episode {ep+1:04d}/{episodes}: {" | ".join(stats)}")

            # Save Q table checkpoints
            if (ep + 1) % ckpt_interval == 0:
                np.save(f"{chkpt_path}/ckpt_ep_{ep+1:06d}.npy", dict(self.q_table))
                # self.show_progress(env, episodes=1)

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # Save final checkpoint and training history
        np.save(f"{chkpt_path}/best_qtable_values.npy", dict(self.q_table))
        np.save(f"{chkpt_path}/training_history.npy", episode_history)

        return episode_history

    def evaluate(self, env: LunarLanderEnv, episodes=100, debug=False) -> List[EpisodeResult]:
        """Runs the trained agent and returns average reward."""
        episode_history: List[EpisodeResult] = []

        for ep in tqdm(range(episodes), disable=True):
            state = env.reset()
            state_tuple = self.discretize(state)
            done = False
            total_reward = 0.0

            while not done:
                action = self.act(state_tuple, evaluate=True)  # Greedy action selection
                next_state, reward, done, result = env.step(action)
                state_tuple = self.discretize(next_state)
                total_reward += reward

            # noinspection PyUnboundLocalVariable
            result['reward'] = total_reward
            episode_history.append(result)

        if debug:
            print("Evaluation complete.")
            print(" | ".join(self._calculate_stats(episode_history)))

        return episode_history

    def show_progress(self, env: LunarLanderEnv, episodes=5, save_gif=False, gif_path="SARSA_Agent_checkpoints/agent1", show_bins=False):
        """Runs the trained agent and renders the environment."""
        renderer = Renderer(env, self.name, save_gif)
        
        for ep in range(episodes):
            state = env.reset()
            state_tuple = self.discretize(state)
            done = False
            total_reward = 0.0
            
            print(f"\nRendering Episode {ep + 1}...")
            
            while not done:
                renderer.clock.tick(60) # Lock to 60 FPS for viewing
                
                # Keep pygame pumping so the window doesn't freeze
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Greedy action selection (epsilon=0)
                action = self.act(state_tuple, evaluate=True)
                
                next_state, reward, done, _ = env.step(action)
                state_tuple = self.discretize(next_state)
                total_reward += reward
                
                renderer.render(action, bins=self.bins if show_bins else None)

            renderer.save_gif(f"{gif_path}/{self.name}_episode_{ep + 1:02d}.gif")
            print(f"Episode {ep + 1}: Total Reward: {total_reward:.1f}")
            pygame.time.wait(1000) # Pause for a second before the next episode
            
        pygame.quit()

class SARSAAgent10Step(SARSAAgent):
    def __init__(self, **kwargs):
        super().__init__(n_step=10, **kwargs)

if __name__ == "__main__":
    # Environment
    num_actions = 4
    # lunar_env = SimpleLunarLanderEnv(num_actions=num_actions, debug=True)
    # lunar_env = LLE_XOffset(debug=True)
    # lunar_env = LLE_InitialVelocity(debug=True)
    lunar_env = LunarLanderEnv(debug=False, max_number_of_seconds=25)
    # lunar_env = RandomLunarLander(debug=True)
    print(f"Initial Reward Potential: {lunar_env._calculate_shaping():.2f}")

    # Agent
    sarsa_agent = SARSAAgent(n_actions=num_actions, n_step=10)  #.load()
    print(f"Number of discrete states: {sarsa_agent.get_number_of_states():,}")
    print(f"Number of winning states: {sarsa_agent.get_number_of_winning_states():,}")
    for bin, states in sarsa_agent.get_states().items():
        print(bin, [f"{s:.3f}" for s in states])

    # Train the agent and save the Q-table
    history = sarsa_agent.train(lunar_env, episodes=50000, debug=True)
    plot_learning_curve([h['reward'] for h in history], agent_type="SARSA")

    # Evaluate and render the trained agent
    sarsa_agent.evaluate(lunar_env, episodes=1000, debug=True)

    # Show a few episodes of the trained agent
    sarsa_agent.show_progress(lunar_env, episodes=5, show_bins=True)
