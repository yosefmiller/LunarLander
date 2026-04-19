import torch
import torch.nn as nn
import numpy as np
import pygame
import os
import time
import threading
import queue
from tqdm import tqdm
from utils import plot_learning_curve, save_as_gif
from lunar_lander_env import SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, Renderer

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
        self.load_state_dict(torch.load(path))

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

class DQNAgent:
    """ 
    A Deep Q-Network (DQN) agent for the Lunar Lander environment. This implementation includes experience replay, 
    target network updates, and epsilon-greedy action selection with optional decay.
    """
    def __init__(self, 
                num_state_features=8,  # Use all features including fuel!
                num_actions=4,
                alpha=1e-4,
                gamma=0.99,
                epsilon_config={'start': 1.0, 'end': 0.05, 'decay': True, 'decay_rate': 0.99995},
                update_freq=4,
                replay_buffer_config={'max_size': 500000, 'min_size': 10000},
                target_update_freq=120,
                minibatch_size=256):
        self.num_state_features = num_state_features
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma

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

    def act(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.q_network.device)
                q_values = self.q_network(state)
                return torch.argmax(q_values).cpu().numpy()
            
    def act_batch(self, states, evaluate=False):
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
        if not explore_mask.all():
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states[~explore_mask]).to(self.q_network.device)
                q_values = self.q_network(states_tensor)
                actions[~explore_mask] = torch.argmax(q_values, dim=1).cpu().numpy()
        
        return actions

    def optimize_q_network(self):
        if len(self.replay_buffer) < self.replay_buffer.min_size:
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
            target_q_values = rewards + (1 - dones) * self.gamma * self.target_network(next_states).max(1)[0]
        
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
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))

    def train(self,
              env,
              episodes=3000,
              max_steps=1000,
              num_parallel_envs=8,
              chkpt_path="DQN_Agent_checkpoints/agent1",
              debug=False,
              logging_rate=5000,
              time_based_logging=False,
              log_every_seconds=15.0):
        
        if debug:
            print(f"Using device: {self.q_network.device}")

        # Save stats for post-training analysis
        completed_episodes = []
        avg_rewards_per_interval = []
        avg_losses = []
        result_history = {'exceeded_max_steps': 0, 'crashed': 0, 'landed': 0}
        max_avg_reward = 0.0

        # Precompute total training steps
        num_training_steps = episodes * max_steps // num_parallel_envs

        # Save Q network weights periodically
        os.makedirs(chkpt_path, exist_ok=True)
        ckpt_interval = max(1, num_training_steps // 3)
        self.ckpt_writer = AsyncCheckpointWriter(max_queue_size=4)

        # Create vectorized environments
        vec_env = VectorizedEnv(env, num_envs=num_parallel_envs, debug=debug)
        obs = vec_env.reset()  # shape (N, state_dim)

        # Per-env accumulators (vectorized)
        total_rewards = np.zeros(num_parallel_envs, dtype=np.float32)
        ep_durations = np.zeros(num_parallel_envs, dtype=np.int32)
        return_history = {i: [] for i in range(num_parallel_envs)}

        # logging Interval accumulators
        interval_episode_returns_sum = 0.0
        interval_episode_durations_sum = 0
        interval_episode_count = 0
        interval_loss_sum = 0.0
        interval_loss_count = 0

        # Optional wall-time logging
        last_log_time = time.monotonic()

        if debug:
            print(f"Training for {episodes} episodes...")

        # Training duration
        start = time.time()      

        for step in tqdm(range(num_training_steps), disable=debug):
            # Act for all envs
            actions = self.act_batch(obs)

            # Step envs
            next_obs, rewards, dones, infos, _ = vec_env.step(actions)

            # Batch push transitions to replay buffer
            self.replay_buffer.batch_push(obs, actions, rewards, next_obs, dones)

            # Update per-env accumulators
            total_rewards += rewards
            ep_durations += 1

            # Handle terminations via mask (vectorized)
            if np.any(dones):
                done_mask = dones.astype(bool)

                # Aggregate per-episode stats into interval sums
                interval_episode_returns_sum += float(total_rewards[done_mask].sum())
                interval_episode_durations_sum += int(ep_durations[done_mask].sum())
                interval_episode_count += int(done_mask.sum())

                # Per environment returns
                idx = np.where(done_mask)[0]
                for i, val in zip(idx, total_rewards[done_mask]):
                    return_history[i].append(float(val))

                # Sum results for done envs
                if len(infos) == num_parallel_envs:
                    # Vectorized-ish accumulation
                    crashed = sum(int(infos[i][0]) for i in range(num_parallel_envs) if done_mask[i])
                    landed = sum(int(infos[i][1]) for i in range(num_parallel_envs) if done_mask[i])
                    exceeded = sum(int(infos[i][2]) for i in range(num_parallel_envs) if done_mask[i])
                    result_history['crashed'] += crashed
                    result_history['landed'] += landed
                    result_history['exceeded_max_steps'] += exceeded

                # Reset only done env accumulators
                total_rewards[done_mask] = 0.0
                ep_durations[done_mask] = 0

            obs = next_obs

            # Periodic optimization
            if self.steps_done % self.update_freq == 0:
                self.optimize_q_network()
                interval_loss_count += 1
                interval_loss_sum += getattr(self, 'running_loss', 0.0)
                self.running_loss = 0.0

            # Target network update
            if self.steps_done % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Logging control: time-based
            do_log = False
            if time_based_logging:
                now = time.monotonic()
                if now - last_log_time >= log_every_seconds:
                    do_log = True
                    last_log_time = now
            else:
                if (step + 1) % logging_rate == 0:
                    do_log = True

            if do_log:
                avg_reward = (interval_episode_returns_sum / interval_episode_count) if interval_episode_count > 0 else 0.0
                avg_duration = (interval_episode_durations_sum / interval_episode_count) if interval_episode_count > 0 else 0.0
                avg_loss = (interval_loss_sum / max(1, interval_loss_count))

                completed_episodes.append(interval_episode_count)
                avg_rewards_per_interval.append(avg_reward)
                avg_losses.append(avg_loss)
                if debug:
                    print(f"Step {step+1}/{num_training_steps}: Epsilon: {self.epsilon:.3f} | Avg Ep Reward: {avg_reward:.1f} | Avg Q Loss: {avg_loss:.3f} | Avg Ep Len: {avg_duration:.1f} | Episodes Completed: {interval_episode_count}")

                # Reset interval accumulators
                interval_episode_returns_sum = 0.0
                interval_episode_durations_sum = 0
                interval_episode_count = 0
                interval_loss_sum = 0.0
                interval_loss_count = 0

                # Save the Q network weights that yield the highest avg rewards
                if avg_reward > max_avg_reward:
                    checkpoint_path = f"{chkpt_path}/best_weights.pth"
                    current_weights = self.q_network.state_dict()
                    self.ckpt_writer.enqueue(checkpoint_path, current_weights)
                    max_avg_reward = avg_reward

            # Also save periodic checkpoints for learning analysis
            if (step + 1) % ckpt_interval == 0:
                checkpoint_path = f"{chkpt_path}/ckpt_step_{step+1:07d}.pth"
                current_weights = self.q_network.state_dict()
                self.ckpt_writer.enqueue(checkpoint_path, current_weights)

            # Epsilon decay once per vectorized step
            if len(self.replay_buffer) > self.replay_buffer.min_size:
                self.decay_epsilon()

            self.steps_done += 1

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f"Results:\n Max Steps Exceeded={result_history['exceeded_max_steps']}, Crashed={result_history['crashed']}, Landed={result_history['landed']}")

        # Final save of Q network weights
        avg_reward = (interval_episode_returns_sum / interval_episode_count) if interval_episode_count > 0 else 0.0
        if avg_reward > max_avg_reward:
            final_weights_path = f"{chkpt_path}/best_weights.pth"
            torch.save(self.q_network.state_dict(), final_weights_path)
        
        # Final save of stats
        env_episodes = [len(returns) for returns in return_history.values()]
        min_episodes = min(env_episodes)  # truncate to the minimum episodes completed
        avg_returns_across_envs = np.array([returns[:min_episodes] for returns in return_history.values()]).mean(axis=0)

        np.save(f"{chkpt_path}/training_returns.npy", avg_returns_across_envs)
        np.save(f"{chkpt_path}/completed_episodes_per_env.npy", env_episodes)
        np.save(f"{chkpt_path}/training_landing_results.npy", result_history)
        np.save(f"{chkpt_path}/training_Q_network_losses.npy", np.array(avg_losses, dtype=np.float32))
        np.save(f"{chkpt_path}/episodes_per_training_interval.npy", np.array(completed_episodes, dtype=np.float32))

        # Stop the checkpoint writer
        self.ckpt_writer.stop(wait=True)

        return avg_returns_across_envs, time_elapsed
    
    def evaluate(self, env, episodes=100, debug=False):
        """ 
        Runs the trained agent and returns average reward.
        """
        total_rewards = []
        ep_durations = []
        results = {'exceeded_max_steps': 0, 'crashed': 0, 'landed': 0}
        
        for ep in tqdm(range(episodes), disable=(not debug)):
            obs = env.reset().to_array()
            done = False
            total_reward = 0
            ep_duration = 0
            
            while not done:
                action = self.act(obs, evaluate=True)  # Greedy action selection
                next_obs, reward, done, result, _ = env.step(action)
                obs = next_obs.to_array()
                total_reward += reward
                ep_duration += 1
                
            ep_durations.append(ep_duration)
            results['crashed'] += int(result[0])
            results['landed'] += int(result[1])
            results['exceeded_max_steps'] += int(result[2])
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        avg_duration = np.mean(ep_durations)
        if debug:
            print(f"Average Reward over {episodes} episodes: {avg_reward:.1f}")
            print(f"Average Episode Duration: {avg_duration:.1f} steps")
            print(f"Results:\n Max Steps Exceeded={results['exceeded_max_steps']}, Crashed={results['crashed']}, Landed={results['landed']}")      

        return total_rewards, avg_duration, results
    
    def show_progress(self, env, episodes=5, save_gif=False, outdir="dqn_agent_recordings"):
        """
        Runs the trained agent and renders the environment.
        """
        renderer = Renderer(env, save_gif=save_gif, output_dir=outdir)
        
        for ep in range(episodes):
            obs = env.reset().to_array()
            done = False
            total_reward = 0
            
            while not done:
                renderer.clock.tick(60) # Lock to 60 FPS for viewing
                
                # Keep pygame pumping so the window doesn't freeze
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Greedy action selection (epsilon=0)
                action = self.act(obs, evaluate=True)
                
                next_obs, reward, done, result, _ = env.step(action)
                obs = next_obs.to_array()
                total_reward += reward

                renderer.render(action)
            
            # Save GIF if enabled
            if save_gif:
                save_as_gif(renderer=renderer, landing_result=result)

            print(f"Episode {ep + 1}: Total Reward: {total_reward:.1f}")
            pygame.time.wait(1000) # Pause for a second before the next episode```

class VectorizedEnv:
    """
    Runs multiple environments in parallel.
    """
    def __init__(self, env_fn, num_envs=8, debug=False):
        self.num_envs = num_envs
        self.envs = [env_fn(debug=debug) for _ in range(num_envs)]
    
    def reset(self):
        return np.array([env.reset().to_array() for env in self.envs])
    
    def reset_env(self, env_index):
        return self.envs[env_index].reset().to_array()
    
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs = np.array([r[0].to_array() for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = np.array([r[3] for r in results])
        shaping_rewards = np.array([r[4] for r in results])
        
        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset().to_array()
        
        return obs, rewards, dones, infos, shaping_rewards
    
class AsyncCheckpointWriter:
    """
    A simple asynchronous checkpoint writer that uses a background thread and a queue to save model weights without 
    blocking the main training loop.
    """
    def __init__(self, max_queue_size=4, flush_interval=0.1):
        self.q = queue.Queue(maxsize=max_queue_size)
        self.flush_interval = flush_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                item = self.q.get(timeout=self.flush_interval)
            except queue.Empty:
                continue
            if item is None:
                # Poison pill to stop
                break
            path, state_dict = item
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                torch.save(state_dict, path)
            except Exception as e:
                print(f"[AsyncCheckpointWriter] Failed to save {path}: {e}")
            finally:
                self.q.task_done()

    def enqueue(self, path, state_dict):
        try:
            self.q.put_nowait((path, state_dict))
        except queue.Full:
            # If queue is full, you can drop the oldest by getting one item
            try:
                _ = self.q.get_nowait()
                self.q.task_done()
                self.q.put_nowait((path, state_dict))
            except Exception:
                # As a fallback, save synchronously
                torch.save(state_dict, path)

    def stop(self, wait=True):
        self._stop_event.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if wait:
            self._thread.join()

if __name__ == "__main__":
    # Environment
    num_actions = 4
    # lunar_env = SimpleLunarLanderEnv(num_actions=num_actions, debug=True)
    # lunar_env = LLE_XOffset(debug=True)
    # lunar_env = LLE_InitialVelocity(debug=True)
    lunar_env = LunarLanderEnv(debug=True)

    # agent
    agent = DQNAgent()

    # Vectorized training with multiple environments in parallel
    chkpt_path="DQN_Agent_checkpoints/agent1"
    rewards, _ = agent.train(LunarLanderEnv,
                             episodes=5000,
                             chkpt_path=chkpt_path,
                             debug=True,
                             time_based_logging=True
                             )

    # Training curves
    num_parallel_envs = 8
    step_factor = 100
    xticks = np.arange(0, num_parallel_envs * len(rewards), step=(num_parallel_envs * step_factor))
    plot_learning_curve(rewards, agent_type='DQN', xticks=xticks)

    logging_intervals = np.load(os.path.join(chkpt_path, 'training_episodes_per_interval.npy'))
    losses = np.load(os.path.join(chkpt_path, 'training_Q_network_losses.npy'))
    plot_learning_curve(losses, agent_type='DQN', xticks=xticks, ylabel='Avg Q Network Loss')

    # Evaluate the trained agent
    rewards, avg_duration, results = agent.evaluate(lunar_env, episodes=100, debug=True)

    # Show a few episodes of the trained agent
    agent.show_progress(lunar_env, episodes=5)