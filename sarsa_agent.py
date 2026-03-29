import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pygame
from tqdm import tqdm

from lunar_lander_env import SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, Renderer


class SARSAAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
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
        self.bins = {
            'x': np.array([-0.6, -0.3, -0.15, 0.15, 0.3, 0.6]),  # 7 bins
            'y': np.array([0.02, 0.05, 0.1, 0.25, 0.5, 0.8]),  # 7 bins
            'vx': np.array([-0.3, -0.1, -0.02, 0.02, 0.1, 0.3]),  # 7 bins
            'vy': np.array([-0.3, -0.1, -0.02, 0.02, 0.1, 0.3]),  # 7 bins
            'theta': np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5]),  # 7 bins
            'omega': np.array([-0.5, -0.2, -0.05, 0.05, 0.2, 0.5])  # 7 bins
        }

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

    def load(self, filename="sarsa_q_table") -> 'SARSAAgent':
        """Loads the Q-table from a file."""
        if not os.path.exists(f"{filename}.pkl"):
            print("No saved Q-table found. Starting with an empty table.")
            return self
        with open(f"{filename}.pkl", "rb") as f:
            loaded_q_table = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), loaded_q_table)
        return self


def train_agent(env, agent, episodes=800, max_steps=1000, agent_type='SARSA') -> list:
    rewards_history = []
    result_history = {'exceeded_max_steps': 0, 'crashed': 0, 'landed': 0}
    episode_durations = []
    total_bins = len(agent.bins['x']) * len(agent.bins['y']) * len(agent.bins['vx']) * len(agent.bins['vy']) * len(agent.bins['theta']) * len(agent.bins['omega'])
    
    print(f"Training for {episodes} episodes. (Rendering is disabled to run fast...)")
    for ep in tqdm(range(episodes), desc=f"Training {agent_type} Agent"):
        state = env.reset()
        state_tuple = agent.discretize(state)
        action = agent.act(state_tuple)
        
        total_reward = 0
        done = False
        ep_duration = 0
        
        while not done:
            # Step the environment
            next_state, reward, done, result, _ = env.step(action)
            next_state_tuple = agent.discretize(next_state)
            
            # SARSA requires choosing the next action BEFORE the update
            next_action = agent.act(next_state_tuple)
            
            # Update Q-Table
            agent.update(state_tuple, action, reward, next_state_tuple, next_action, done)
            
            # Progress to next step
            state_tuple = next_state_tuple
            action = next_action
            total_reward += reward
            ep_duration += 1

            # Limit the episode length
            result['exceeded_max_steps'] = False
            if ep_duration > max_steps:
                done = True
                result['exceeded_max_steps'] = True

                # Penalty for timeout
                penalty = 0.0 if agent.n_actions == 2 else -50.0 # No penalty for 2-action envs to encourage learning main engine usage
                agent.update(state_tuple, action, penalty, next_state_tuple, next_action, done)
            
        episode_durations.append(ep_duration)
        agent.decay_epsilon()
        
         # save this episodes reward and result (crashed/landed)
        rewards_history.append(total_reward)
        for k in result_history.keys():
            if result[k]:
                result_history[k] += 1
        
        if (ep + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_duration = np.mean(episode_durations[-50:])
            coverage = (len(agent.q_table) / total_bins) * 100
            print(f"Episode: {ep+1:04d} | Epsilon: {agent.epsilon:.3f} | Last 50 Avg Reward: {avg_reward:.1f} | Q-Table Coverage: {coverage:.2f}% | Last 50 Avg Episode Duration: {avg_duration:.1f} steps")
            episode_durations = []

    print("Training Complete!")
    print(f"Results:\n Max Steps Exceeded={result_history['exceeded_max_steps']}, Crashed={result_history['crashed']}, Landed={result_history['landed']}")
    return rewards_history

def plot_learning_curve(rewards_history: list, agent_type='SARSA', title="Lunar Lander"):
    # Plotting the learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.5, color='gray', label='Raw Reward')

    # Calculate a moving average
    window = 20
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards_history)), moving_avg, color='blue', label='Moving Average (20 ep)')

    plt.title(f"{agent_type} Agent Learning Curve - {title}")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()


def evaluate_agent(env, agent, num_episodes=5):
    """Runs the trained agent and renders the environment."""
    print("Launching Pygame to evaluate trained agent...")
    renderer = Renderer(env)
    
    for ep in range(num_episodes):
        state = env.reset()
        state_tuple = agent.discretize(state)
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
            action = agent.act(state_tuple, evaluate=True)
            
            next_state, reward, done, _, _ = env.step(action)
            state_tuple = agent.discretize(next_state)
            total_reward += reward
            
            renderer.render(action)
            
        print(f"Episode Ended. Total Reward: {total_reward:.1f}")
        pygame.time.wait(1000) # Pause for a second before the next episode
        
    pygame.quit()


if __name__ == "__main__":
    # Environment
    num_actions = 4
    lunar_env = SimpleLunarLanderEnv(num_actions=num_actions)
    # lunar_env = LLE_XOffset()
    # lunar_env = LLE_InitialVelocity()
    # lunar_env = LunarLanderEnv()

    # Agent
    sarsa_agent = SARSAAgent(n_actions=num_actions, epsilon_decay=0.9995, alpha=0.4, epsilon_min=0.05) #.load("sarsa_simple_lander")

    # Train the agent and save the Q-table
    rewards = train_agent(lunar_env, sarsa_agent, episodes=50000)
    plot_learning_curve(rewards, "Simple Lunar Lander")
    # sarsa_agent.save("sarsa_simple_lander")

    # Evaluate and render the trained agent
    evaluate_agent(lunar_env, sarsa_agent, num_episodes=5)
