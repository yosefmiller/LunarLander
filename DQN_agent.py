from collections.abc import Collection
from typing import List, Type

import torch
import torch.nn as nn
import numpy as np
import pygame
import os
import time
from tqdm import tqdm
from utils import plot_learning_curve, save_as_gif, plot_outcomes
from lunar_lander_env import LunarLanderEnv, VectorizedEnv, Renderer, EpisodeResult, SimpleLunarLanderEnv
from base_agent import BaseAgent

class QNetwork(nn.Module):
    """
    A simple feedforward neural network for approximating Q-values.
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(state, torch.Tensor):
            x = torch.tensor(state, device=self.device, dtype=torch.float32)        
        return x
    
    def forward(self, x):
        x = self._format(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output_layer(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def reset(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences.
    """
    def __init__(self, capacity, state_dim, min_size=1000):
        self.capacity = capacity
        self.min_size = min_size
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        self.states[self.position] = s
        self.actions[self.position] = a
        self.rewards[self.position] = r
        self.next_states[self.position] = ns
        self.dones[self.position] = float(d)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def batch_push(self, states, actions, rewards, next_states, dones):
        n = len(actions)

        # If wrap-around, split write into two slices
        first = min(n, self.capacity - self.position)
        second = n - first
        if first > 0:
            self.states[self.position:self.position+first] = states[:first]
            self.actions[self.position:self.position+first] = actions[:first]
            self.rewards[self.position:self.position+first] = rewards[:first]
            self.next_states[self.position:self.position+first] = next_states[:first]
            self.dones[self.position:self.position+first] = dones[:first].astype(np.float32)
        if second > 0:
            self.states[0:second] = states[first:]
            self.actions[0:second] = actions[first:]
            self.rewards[0:second] = rewards[first:]
            self.next_states[0:second] = next_states[first:]
            self.dones[0:second] = dones[first:].astype(np.float32)

        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

class DQNAgent(BaseAgent):
    """
    A Deep Q-Network (DQN) agent for the Lunar Lander environment. This implementation includes experience replay,
    target network updates, and epsilon-greedy action selection with optional decay.
    """
    def __init__(self,
                num_state_features=8,  # Use all features including fuel!
                num_actions=4,
                alpha=1e-4,
                gamma=0.99,
                use_double_dqn=False,
                epsilon_config=None,
                update_freq=4,
                replay_buffer_config=None,
                target_update_freq=120,
                minibatch_size=256):
        if epsilon_config is None:
            epsilon_config = {'start': 1.0, 'end': 0.05, 'decay': True, 'decay_rate': 0.999995}
        if replay_buffer_config is None:
            replay_buffer_config = {'max_size': 500000, 'min_size': 10000}

        self.num_state_features = num_state_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.use_double_dqn = bool(use_double_dqn)

        # Initialize Q-Network and Target Network
        self.q_network = QNetwork(num_state_features, num_actions)
        self.target_network = QNetwork(num_state_features, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Start with identical weights
        self.target_network.eval()  # Target network is only used for inference, so set to eval mode

        # Q Network parameters
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), lr=alpha)
        self.batch_size = minibatch_size
        self.running_loss = 0.0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_config['max_size'], 
                                          state_dim=num_state_features, 
                                          min_size=replay_buffer_config['min_size'])
        
        # Epsilon-greedy parameters
        self.use_epsilon_decay = epsilon_config['decay']
        self.epsilon = epsilon_config['start']
        self.epsilon_min = epsilon_config['end']
        self.epsilon_decay = epsilon_config['decay_rate']

        # Keep track of the number of completed training steps
        self.steps_done = 0

    @property
    def name(self) -> str:
        return "DoubleDQN" if self.use_double_dqn else "DQN"

    def act(self, state, evaluate=False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.q_network.device)
                q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values).item())

    def act_batch(self, states, evaluate=False) -> Collection[int]:
        """Select actions for a batch of states with per-environment epsilon-greedy."""
        batch_size = len(states)
        
        if evaluate:
            # Pure exploitation during evaluation
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(self.q_network.device)
                q_values = self.q_network(states_tensor)
                return torch.argmax(q_values, dim=1).cpu().numpy()
        
        # Per-environment epsilon-greedy
        actions = np.zeros(batch_size, dtype=np.int64)
        explore_mask = np.random.rand(batch_size) < self.epsilon
        
        # Random actions for exploration
        actions[explore_mask] = np.random.randint(0, self.num_actions, size=np.sum(explore_mask))
        
        # Greedy actions for exploitation
        if not np.all(explore_mask):
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states[~explore_mask]).to(self.q_network.device)
                q_values = self.q_network(states_tensor)
                actions[~explore_mask] = torch.argmax(q_values, dim=1).cpu().numpy()
        
        return actions

    def optimize_q_network(self):
        if len(self.replay_buffer) < max(self.replay_buffer.min_size, self.batch_size):
            return
        
        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.q_network.device, non_blocking=True)
        actions = torch.from_numpy(actions).to(self.q_network.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).to(self.q_network.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).to(self.q_network.device, non_blocking=True)
        dones = torch.from_numpy(dones).to(self.q_network.device, non_blocking=True)

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(1)[0]

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and optimize the Q-network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
        self.optimizer.step()

        self.running_loss += loss.item()
    
    def decay_epsilon(self):
        if self.use_epsilon_decay:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        weights_path = path if path.endswith((".pth", ".pt")) else os.path.join(path, "best_weights.pth")
        state_dict = torch.load(weights_path, map_location=self.q_network.device)
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(self.q_network.state_dict())
        return self

    def train(self,
              env: Type[LunarLanderEnv] | LunarLanderEnv,
              episodes=3000,
              max_steps=1000,
              num_parallel_envs=8,
              chkpt_path="DQN_Agent_checkpoints/agent1",
              debug=False,
              logging_rate=5000,
              time_based_logging=False,
              log_every_seconds=15.0) -> List[EpisodeResult]:

        if debug:
            print(f"Using device: {self.q_network.device}")

        # Save episode history: one dict per completed episode
        episode_history: List[EpisodeResult] = []

        # Keep max_steps for API compatibility and rough progress/checkpoint planning.
        estimated_training_steps = max(1, episodes * max_steps // max(1, num_parallel_envs))

        # Save Q network weights periodically
        os.makedirs(chkpt_path, exist_ok=True)
        ckpt_episode_interval = max(1, episodes // 3)
        next_ckpt_episode = ckpt_episode_interval

        # Accept either an env class or an env instance.
        env_factory = env if isinstance(env, type) else type(env)

        # Create vectorized environments
        vec_env = VectorizedEnv(env_factory, num_envs=num_parallel_envs, debug=False)
        obs = vec_env.reset()  # shape (N, state_dim)

        # Per-env accumulators (vectorized)
        total_rewards = np.zeros(num_parallel_envs, dtype=np.float32)
        interval_history: List[EpisodeResult] = []
        interval_loss_sum = 0.0
        interval_loss_count = 0
        completed_episodes = 0
        next_log_episode = max(1, logging_rate)
        next_log_time = time.monotonic() + log_every_seconds
        min_replay_size = max(self.replay_buffer.min_size, self.batch_size)

        if debug:
            print(f"Training for {episodes} episodes (estimated {estimated_training_steps} vectorized steps)...")

        # Training duration
        start = time.time()
        progress_bar = tqdm(total=episodes, disable=debug, desc="Episodes", unit="ep")

        while completed_episodes < episodes:
            # Act for all envs
            actions = self.act_batch(obs)

            # Step envs
            next_obs, rewards, dones, infos = vec_env.step(actions)

            # Batch push transitions to replay buffer
            self.replay_buffer.batch_push(obs, actions, rewards, next_obs, dones)

            # Update per-env accumulators
            total_rewards += rewards

            # Collect completed episodes immediately; vectorized envs may finish several at once.
            if np.any(dones):
                done_mask = dones.astype(bool)
                idx = np.where(done_mask)[0]
                episodes_added = 0
                for i, val in zip(idx, total_rewards[done_mask].copy()):
                    if completed_episodes >= episodes:
                        break
                    # noinspection PyTypeChecker
                    episode_result: EpisodeResult = infos[i]
                    episode_result['reward'] = float(val)
                    episode_history.append(episode_result)
                    interval_history.append(episode_result)
                    completed_episodes += 1
                    episodes_added += 1

                progress_bar.update(episodes_added)

                # Reset only done env accumulators
                total_rewards[done_mask] = 0.0

            obs = next_obs

            # Periodic optimization
            if (self.steps_done + 1) % self.update_freq == 0:
                self.optimize_q_network()
                interval_loss_count += 1
                interval_loss_sum += getattr(self, 'running_loss', 0.0)
                self.running_loss = 0.0

            # Target network update
            if (self.steps_done + 1) % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            should_log = False
            if interval_history:
                if time_based_logging:
                    should_log = time.monotonic() >= next_log_time
                else:
                    should_log = completed_episodes >= next_log_episode

            if should_log:
                avg_loss = interval_loss_sum / max(1, interval_loss_count)
                stats = [
                    f"Step: {self.steps_done + 1}",
                    f"Episodes: {completed_episodes}/{episodes}",
                    f"Buffer: {len(self.replay_buffer)}",
                    f"Epsilon: {self.epsilon:.3f}",
                    f"Avg Q Loss: {avg_loss:.3f}",
                    *self._calculate_stats(interval_history),
                ]
                if debug:
                    print(" | ".join(stats))

                interval_history.clear()
                interval_loss_sum = 0.0
                interval_loss_count = 0
                if time_based_logging:
                    next_log_time = time.monotonic() + log_every_seconds
                else:
                    next_log_episode += max(1, logging_rate)

            # Also save periodic checkpoints for learning analysis
            if completed_episodes >= next_ckpt_episode:
                checkpoint_path = f"{chkpt_path}/ckpt_step_{self.steps_done + 1:07d}.pth"
                torch.save(self.q_network.state_dict(), checkpoint_path)
                next_ckpt_episode += ckpt_episode_interval

            # Epsilon decay once per vectorized step
            if len(self.replay_buffer) >= min_replay_size:
                self.decay_epsilon()

            self.steps_done += 1

        progress_bar.close()

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if episode_history:
            print(" | ".join(self._calculate_stats(episode_history)))

        torch.save(self.q_network.state_dict(), f"{chkpt_path}/best_weights.pth")
        np.save(f"{chkpt_path}/training_history.npy", np.array(episode_history, dtype=object))

        return episode_history

    def evaluate(self, env: LunarLanderEnv, episodes=100, debug=False) -> List[EpisodeResult]:
        """
        Runs the trained agent and returns episode history.
        """
        episode_history: List[EpisodeResult] = []

        for ep in tqdm(range(episodes), disable=(not debug)):
            obs = env.reset().to_array()
            done = False
            total_reward = 0.0

            while not done:
                action = self.act(obs, evaluate=True)  # Greedy action selection
                next_obs, reward, done, result = env.step(action)
                obs = next_obs.to_array()
                total_reward += reward

            # noinspection PyUnboundLocalVariable
            result['reward'] = total_reward
            episode_history.append(result)

        if debug:
            print("Evaluation complete.")
            print(" | ".join(self._calculate_stats(episode_history)))

        return episode_history

    def show_progress(self, env: LunarLanderEnv, episodes=5, save_gif=False, gif_path="dqn_agent_recordings"):
        """
        Runs the trained agent and renders the environment.
        """
        renderer = Renderer(env, self.name, save_gif, output_dir=gif_path)

        for ep in range(episodes):
            obs = env.reset().to_array()
            done = False
            total_reward = 0.0

            while not done:
                renderer.clock.tick(60) # Lock to 60 FPS for viewing
                
                # Keep pygame pumping so the window doesn't freeze
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Greedy action selection (epsilon=0)
                action = self.act(obs, evaluate=True)
                
                next_obs, reward, done, result = env.step(action)
                obs = next_obs.to_array()
                total_reward += reward

                renderer.render(action)

            if save_gif:
                save_as_gif(renderer=renderer, landing_result=result)
            print(f"Episode {ep + 1}: Total Reward: {total_reward:.1f}")
            pygame.time.wait(1000) # Pause for a second before the next episode

        pygame.quit()

class DoubleDQNAgent(DQNAgent):
    def __init__(self, **kwargs):
        super().__init__(use_double_dqn=True, **kwargs)

if __name__ == "__main__":
    # Environment
    lunar_env = LunarLanderEnv(debug=False)

    # Agent
    agent = DQNAgent(use_double_dqn=True)

    # Train the agent
    history = agent.train(LunarLanderEnv, episodes=5000, debug=True, logging_rate=100)
    # noinspection PyTypeChecker
    plot_learning_curve([h['reward'] for h in history], agent_type=agent.name)
    plot_outcomes(history, agent_type="DQN")

    # Evaluate and render the trained agent
    agent.evaluate(lunar_env, episodes=100)

    # Show a few episodes of the trained agent
    agent.show_progress(lunar_env, episodes=5)