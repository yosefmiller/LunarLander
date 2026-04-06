import os
import time
import pickle
from collections import defaultdict
import numpy as np
import pygame
from tqdm import tqdm
from utils import plot_learning_curve
from lunar_lander_env import SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, Renderer


class SARSAAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        """
        SARSA Agent for the Lunar Lander environment.

        How SARSA works:
        1. The agent observes the current state (discretized).
        2. It selects an action using an epsilon-greedy policy based on the Q-table.
        3. It takes the action, observes the reward and the next state.
        4. It selects the next action using the SAME epsilon-greedy policy (on-policy).
        5. It updates the Q-table using the SARSA update rule: Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S', A') - Q(S,A)]

        :param n_actions: Number of discrete actions available in the environment (e.g., 2 for main engine on/off, or 4 for main engine + left/right thrusters).
        :param alpha: Learning rate (0 < alpha <= 1). Higher values mean the agent learns more from new experiences.
        :param gamma: Discount factor (0 <= gamma < 1). Higher values mean the agent values future rewards more.
        :param epsilon: Initial exploration rate (0 <= epsilon <= 1). Higher values mean the agent explores more at the start.
        :param epsilon_decay: Multiplicative factor for decaying epsilon after each episode (0 < epsilon_decay < 1). A value of 0.995 means epsilon will decay by 0.5% each episode.
        :param epsilon_min: Minimum exploration rate. Epsilon will not decay below this value, ensuring the agent continues to explore occasionally even after many episodes.
        """

        self.n_actions = n_actions
        
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
        x_bins = np.concatenate([
                    np.array([-0.6, -0.3, -0.15, -0.1]),
                    np.linspace(-0.1, 0.1, 5),   # high resolution near pad
                    np.array([0.1, 0.15, 0.3, 0.6])
                ])
        y_bins = np.concatenate([
                    np.linspace(0, 0.01, 4),      # landing zone precision
                    np.array([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.8])
                ])
        vx_bins = np.array([-0.3, -0.1, -0.02, 0.02, 0.1, 0.3])
        vy_bins = np.concatenate([
                    np.array([-0.3, -0.1, -0.05]),
                    np.linspace(-0.05, 0.05, 6),   # critical landing region
                    np.array([0.05, 0.1, 0.3])
                ])
        angle_bins = np.concatenate([
                        np.array([-0.5, -0.2]),
                        np.linspace(-0.2, 0.2, 7),   # upright precision
                        np.linspace(0.2, 0.5)
                    ])
        ang_vel_bins = np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5])

        self.bins = {
            'x': x_bins,
            'y': y_bins,
            'vx': vx_bins,
            'vy': vy_bins,
            'theta': angle_bins,
            'omega': ang_vel_bins
        }

        # self.bins = {
        #     'x': np.array([-0.6, -0.3, -0.15, 0.15, 0.3, 0.6]),  # 7 bins
        #     'y': np.array([0.02, 0.05, 0.1, 0.25, 0.5, 0.8]),  # 7 bins
        #     'vx': np.array([-0.3, -0.1, -0.02, 0.02, 0.1, 0.3]),  # 7 bins
        #     'vy': np.array([-0.3, -0.1, -0.02, 0.02, 0.1, 0.3]),  # 7 bins
        #     'theta': np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5]),  # 7 bins
        #     'omega': np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5])  # 7 bins
        # }

    def discretize(self, state):
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

    def act(self, state_tuple, evaluate=False):
        """Selects an action using Epsilon-Greedy policy."""
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) # Explore
        
        # Exploit: return the action with the highest Q-value
        return np.argmax(self.q_table[state_tuple])

    def update(self, state_tuple, action, reward, next_state_tuple, next_action, done):
        """Applies the SARSA update rule."""
        current_q = self.q_table[state_tuple][action]
        
        # If the episode is over, the expected future reward is 0
        next_q = 0 if done else self.q_table[next_state_tuple][next_action]

        # SARSA Formula: Q(S,A) = Q(S,A) + alpha * [R + gamma * Q(S', A') - Q(S,A)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state_tuple][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename="sarsa_q_table"):
        """Saves the Q-table to a file."""
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(dict(self.q_table), f)

    # TODO: Load q table from .npy file instead of pickle
    def load(self, filename="sarsa_q_table") -> 'SARSAAgent':
        """Loads the Q-table from a file."""
        if not os.path.exists(f"{filename}.pkl"):
            print("No saved Q-table found. Starting with an empty table.")
            return self
        with open(f"{filename}.pkl", "rb") as f:
            loaded_q_table = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), loaded_q_table)
        return self

    def train(self,
              env, 
              episodes=1000, 
              agent_type='SARSA', 
              chkpt_path="SARSA_Agent_checkpoints/agent1", 
              debug=False,
              log_every_episodes=500) -> list:
        rewards_history = []
        avg_rewards_per_interval = []
        result_history = {'crashed': 0, 'landed': 0, 'exceeded_max_steps': 0}
        episode_durations = []
        total_bins = len(self.bins['x']) * len(self.bins['y']) * len(self.bins['vx']) * len(self.bins['vy']) * len(self.bins['theta']) * len(self.bins['omega'])
        
        # Save Q table periodically
        os.makedirs(chkpt_path, exist_ok=True)
        ckpt_interval = max(1, episodes // 3)

        print(f"Training for {episodes} episodes...")

        # Training duration
        start = time.time()      

        for ep in range(episodes):
            state = env.reset()
            state_tuple = self.discretize(state)
            action = self.act(state_tuple)
            
            total_reward = 0
            done = False
            ep_duration = 0
            
            while not done:
                # Step the environment
                next_state, reward, done, result, _ = env.step(action)
                next_state_tuple = self.discretize(next_state)
                
                # SARSA requires choosing the next action BEFORE the update
                next_action = self.act(next_state_tuple)
                
                # Update Q-Table
                self.update(state_tuple, action, reward, next_state_tuple, next_action, done)
                
                # Progress to next step
                state_tuple = next_state_tuple
                action = next_action
                total_reward += reward
                ep_duration += 1

                # save result if done
                if done:
                    result_history['crashed'] += result[0]
                    result_history['landed'] += result[1]
                    result_history['exceeded_max_steps'] += result[2]
                
            episode_durations.append(ep_duration)
            self.decay_epsilon()
            
            # save this episode's reward
            rewards_history.append(total_reward)
            
            # Log stats
            if debug and (ep + 1) % log_every_episodes == 0:
                avg_reward = np.mean(rewards_history[-log_every_episodes:])
                avg_duration = np.mean(episode_durations[-log_every_episodes:])
                coverage = (len(self.q_table) / total_bins) * 100
                print(f"Episode {ep+1:04d}/{episodes}: Epsilon: {self.epsilon:.3f} | Last {log_every_episodes} Avg Reward: {avg_reward:.1f} | Q-Table Coverage: {coverage:.2f}% | Last {log_every_episodes} Avg Episode Duration: {avg_duration:.1f} steps")
                avg_rewards_per_interval.append(avg_reward)
                episode_durations = []

            # Save Q table checkpoints
            if (ep + 1) % ckpt_interval == 0:
                np.save(f"{chkpt_path}/{agent_type}_ep_{ep+1:06d}.npy", dict(self.q_table))

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Results:\n Max Steps Exceeded={result_history['exceeded_max_steps']}, Crashed={result_history['crashed']}, Landed={result_history['landed']}")

        # save final checkpoint and avg rewards
        np.save(f"{chkpt_path}/{agent_type}_trained.npy", dict(self.q_table))
        np.save(f"{chkpt_path}/{agent_type}_avg_rewards.npy", avg_rewards_per_interval)

        return avg_rewards_per_interval

    def evaluate(self, env, episodes=100):
        """Runs the trained agent and returns average reward."""
        total_rewards = []
        ep_durations = []
        results = {'exceeded_max_steps': 0, 'crashed': 0, 'landed': 0}
        
        for ep in tqdm(range(episodes)):
            state = env.reset()
            state_tuple = self.discretize(state)
            done = False
            total_reward = 0
            ep_duration = 0

            while not done:
                action = self.act(state_tuple, evaluate=True)  # Greedy action selection
                next_state, reward, done, result, _ = env.step(action)
                state_tuple = self.discretize(next_state)
                total_reward += reward
                ep_duration += 1
                
            ep_durations.append(ep_duration)
            results['crashed'] += int(result[0])
            results['landed'] += int(result[1])
            results['exceeded_max_steps'] += int(result[2])
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        avg_duration = np.mean(ep_durations)
        print(f"Average Reward over {episodes} episodes: {avg_reward:.1f}")
        print(f"Average Episode Duration: {avg_duration:.1f} steps")
        print(f"Results:\n Max Steps Exceeded={results['exceeded_max_steps']}, Crashed={results['crashed']}, Landed={results['landed']}")

    def show_progress(self, env, episodes=5, save_gif=False, gif_path="dqn_agent_progress.gif"):
        """Runs the trained agent and renders the environment."""
        print("Launching Pygame to evaluate trained agent...")
        renderer = Renderer(env)
        
        for ep in range(episodes):
            state = env.reset()
            state_tuple = self.discretize(state)
            done = False
            total_reward = 0
            
            print(f"\nEvaluating Episode {ep + 1}...")
            
            while not done:
                renderer.clock.tick(60) # Lock to 60 FPS for viewing
                
                # Keep pygame pumping so the window doesn't freeze
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Greedy action selection (epsilon=0)
                action = self.act(state_tuple, evaluate=True)
                
                next_state, reward, done, _, _ = env.step(action)
                state_tuple = self.discretize(next_state)
                total_reward += reward
                
                renderer.render(action)
                
            print(f"Episode Ended. Total Reward: {total_reward:.1f}")
            pygame.time.wait(1000) # Pause for a second before the next episode
            
        pygame.quit()

if __name__ == "__main__":
    # Environment
    num_actions = 4
    # lunar_env = SimpleLunarLanderEnv(num_actions=num_actions)
    # lunar_env = LLE_XOffset()
    # lunar_env = LLE_InitialVelocity()
    lunar_env = LunarLanderEnv()

    # Agent
    sarsa_agent = SARSAAgent(n_actions=num_actions)

    # Train the agent and save the Q-table
    logging_interval = 500
    rewards = sarsa_agent.train(lunar_env, episodes=50000, debug=True, log_every_episodes=logging_interval)
    episode_intervals = logging_interval = 500*np.ones(len(rewards))
    plot_learning_curve(rewards, episode_intervals=episode_intervals, title="Simple Lunar Lander")

    # Evaluate and render the trained agent
    sarsa_agent.evaluate(lunar_env, episodes=1000)

    # Show a few episodes of the trained agent
    sarsa_agent.show_progress(lunar_env, episodes=5)
