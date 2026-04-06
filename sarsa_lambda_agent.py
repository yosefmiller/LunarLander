
from utils import plot_learning_curve
from lunar_lander_env import SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, Renderer
import pygame
import os
import time
from tqdm import tqdm
import numpy as np

class TileCoder:
    """
    Tile coder for continuous state spaces.
    This implementation creates multiple overlapping tilings to allow for better generalization.
    """
    def __init__(self, low, high, tilings=8, tiles=8):
        """
        Initialize the tile coder.

        :param low: list of lower bounds for each state dimension
        :param high: list of upper bounds for each state dimension
        :param tilings: number of overlapping tilings
        :param tiles: number of tiles per dimension in each tiling
        """
        self.low = np.array(low)
        self.high = np.array(high)
        self.tilings = tilings
        self.tiles = tiles
        
        self.tile_width = (self.high - self.low) / tiles
        
        # offsets for each tiling
        self.offsets = [
            (i / tilings) * self.tile_width
            for i in range(tilings)
        ]
    
    def get_tiles(self, state):
        """
        Get the active tiles for a given state.
        The state is first shifted by the tiling offsets and then discretized into tile indices.

        :param state: the continuous state to be encoded
        :return: a list of active tile indices for each tiling
        """
        state = np.array(state.to_array()[:6])  # ignore leg contacts for tiling
        tiles = []
        
        for tiling, offset in enumerate(self.offsets):
            shifted = state + offset
            indices = np.clip(((shifted - self.low) / self.tile_width).astype(int), 0, self.tiles - 1)
            tiles.append((tiling, *indices))
        
        return tiles
    
class SARSALambdaAgent:
    """ 
    SARSA(λ) agent with tile coding for function approximation to solve the Lunar Lander problem.
    This agent uses eligibility traces to allow for more efficient learning from sequences of actions and rewards.
    """
    def __init__(self, state_low, state_high, actions=4, alpha=0.1):
        self.actions = actions        
        self.tc = TileCoder(state_low, state_high)
        
        self.alpha = alpha / self.tc.tilings
        self.gamma = 0.99
        self.lam = 0.9
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Use dictionaries for sparse representation
        self.w = {}  # weights
        self.e = {}  # eligibility traces
    
    def get_q(self, tiles, action):
        """
        Calculate the Q-value for a given state (represented by active tiles) and action by summing 
        the weights of the active tiles for that action.

        :param tiles (List[Tuple[int]]): List of active tile indices for the current state.
        :param action (int): The action for which to calculate the Q-value.
        :return: The estimated Q-value for the given state and action.
        """
        q = 0
        for tile in tiles:
            key = (*tile, action)
            q += self.w.get(key, 0.0)
        return q
    
    def choose_action(self, tiles, evaluate=False):
        """
        Choose an action based on the current state and epsilon-greedy policy.

        :param tiles (List[Tuple[int]]): List of active tile indices for the current state.
        :return: The chosen action.
        """
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.actions)
        
        q_vals = np.array([self.get_q(tiles, a) for a in range(self.actions)])
        return np.argmax(q_vals)
    
    def update(self, tiles, action, reward, next_tiles, next_action):
        """
        Update the weights and eligibility traces based on the observed transition (state, action, reward, next state, next action).
        The update is performed using the SARSA(λ) algorithm, which incorporates eligibility traces to allow for more 
        efficient learning from sequences of actions and rewards.

        :param tiles: List of active tile indices for the current state.
        :param action: The action taken in the current state.
        :param reward: The reward received after taking the action.
        :param next_tiles: List of active tile indices for the next state.
        :param next_action: The action taken in the next state.
        """
        q = self.get_q(tiles, action)
        q_next = self.get_q(next_tiles, next_action)        
        delta = reward + self.gamma * q_next - q
        
        # Decay traces (only for non-zero traces)
        decay_factor = self.gamma * self.lam
        keys_to_remove = []
        for key in self.e:
            self.e[key] *= decay_factor
            # Prune traces below threshold
            if abs(self.e[key]) < 0.01:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.e[key]
        
        # Set traces for current state-action
        for tile in tiles:
            key = (*tile, action)
            self.e[key] = 1.0
        
        # Update weights (only for non-zero traces)
        for key, trace_value in self.e.items():
            self.w[key] = self.w.get(key, 0.0) + self.alpha * delta * trace_value
    
    def decay_epsilon(self):
        """
        Decay the exploration rate epsilon after each episode to reduce exploration as the agent learns more about the environment.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def train(self,
              env,
              episodes=5000,
              chkpt_path="SARSA_Lambda_Agent_checkpoints/agent1",
              debug=False,
              log_every_episodes=500) -> list:
        """
        Train the given agent in the specified environment for a certain number of episodes.
        The rewards received in each episode are recorded to analyze the learning progress of the agent.
        The training loop includes a mechanism to limit the number of steps per episode to prevent infinite loops.
        
        :param env: The environment in which the agent will be trained.
        :param agent: The agent to be trained.
        :param episodes: The number of episodes to train the agent for.
        :param max_steps: The maximum number of steps allowed per episode to prevent infinite loops.
        :return: A list of total rewards received in each episode, which can be used to analyze the learning progress of the agent.
        """
        rewards_history = []
        avg_rewards_per_interval = []
        result_history = {'crashed': 0, 'landed': 0, 'exceeded_max_steps': 0}
        episode_durations = []

        # Save Q table periodically
        os.makedirs(chkpt_path, exist_ok=True)
        ckpt_interval = max(1, episodes // 3)

        print(f"Training for {episodes} episodes...")

        # Training duration
        start = time.time()   

        for ep in range(episodes):
            obs = env.reset()
            tiles = self.tc.get_tiles(obs)
            action = self.choose_action(tiles)
            self.e = {}

            done = False
            total_reward = 0
            ep_duration = 0
            
            while not done:
                # Step the environment
                obs_next, reward, done, result, _ = env.step(action)

                # Use tiles to determine next action
                next_tiles = self.tc.get_tiles(obs_next)
                next_action = self.choose_action(next_tiles)

                # Update the weights
                self.update(tiles, action, reward, next_tiles, next_action)

                # Progress to the next step
                tiles = next_tiles
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
                print(f"Episode {ep+1:04d}/{episodes}: Epsilon: {self.epsilon:.3f} | Last {log_every_episodes} Avg Reward: {avg_reward:.1f} | Last {log_every_episodes} Avg Episode Duration: {avg_duration:.1f} steps")
                avg_rewards_per_interval.append(avg_reward)
                episode_durations = []

            # Save Q table checkpoints
            if (ep + 1) % ckpt_interval == 0:
                np.save(f"{chkpt_path}/SARSALambda_ep_{ep+1:06d}.npy", dict(self.w))

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Results:\n Max Steps Exceeded={result_history['exceeded_max_steps']}, Crashed={result_history['crashed']}, Landed={result_history['landed']}")

        # save final checkpoint and avg rewards
        np.save(f"{chkpt_path}/SARSALambda_trained.npy", dict(self.w))
        np.save(f"{chkpt_path}/SARSALambda_avg_rewards.npy", avg_rewards_per_interval)

        return avg_rewards_per_interval

    def evaluate(self, env, episodes=100):
            """Runs the trained agent and returns average reward."""
            total_rewards = []
            ep_durations = []
            results = {'exceeded_max_steps': 0, 'crashed': 0, 'landed': 0}
            
            for ep in tqdm(range(episodes)):
                obs = env.reset()
                tiles = self.tc.get_tiles(obs)
                done = False
                total_reward = 0
                ep_duration = 0

                while not done:
                    action = self.choose_action(tiles, evaluate=True)
                    obs, reward, done, result, _ = env.step(action)
                    tiles = self.tc.get_tiles(obs)
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

    def show_progress(self, env, episodes=5):
        """
        Evaluate the trained agent by running it in the environment for a specified number of episodes and rendering the results.
        
        :param env: The environment in which to evaluate the agent.
        :param agent: The trained agent to be evaluated.
        :param num_episodes: The number of episodes to run for evaluation.
        """
        renderer = Renderer(env)
        self.epsilon = 0

        for i in range(episodes):
            obs = env.reset()
            tiles = self.tc.get_tiles(obs)
            done = False
            total_reward = 0

            while not done:
                renderer.clock.tick(60) # Lock to 60 FPS for viewing
                
                # Keep pygame pumping so the window doesn't freeze
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    
                action = self.choose_action(tiles, evaluate=True)
                obs, reward, done, _, _ = env.step(action)
                tiles = self.tc.get_tiles(obs)
                total_reward += reward            
                renderer.render(action)

            print(f"Episode {i}: Total Reward={total_reward:.1f}")
            pygame.time.wait(1000) # Pause for a second before the next episode
            
        pygame.quit()


if __name__ == "__main__":
    # Environment
    num_actions = 4  # Main engine + left/right thrusters
    # lunar_env = SimpleLunarLanderEnv(num_actions=num_actions)
    # lunar_env = LLE_XOffset()
    # lunar_env = LLE_InitialVelocity()
    lunar_env = LunarLanderEnv()

    # Agent
    state_low = [-1.5, -0.5, -2, -2, -1.0, -5]
    state_high = [1.5, 1.5, 2, 2, 1.0, 5]

    agent = SARSALambdaAgent(state_low, state_high, actions=num_actions)

    # Train the agent
    logging_interval = 50
    rewards = agent.train(lunar_env, episodes=20000, debug=True, log_every_episodes=logging_interval)
    episode_intervals = logging_interval*np.ones(len(rewards))
    plot_learning_curve(rewards, episode_intervals=episode_intervals, agent_type='SARSA Lambda (with Tile Coding)')

    # Evaluate and render the trained agent
    agent.evaluate(lunar_env, episodes=1000)

    # Evaluate and render the trained agent
    agent.show_progress(lunar_env, episodes=5)