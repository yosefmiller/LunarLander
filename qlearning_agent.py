from typing import List, Tuple

from sarsa_agent import SARSAAgent
from lunar_lander_env import SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, Renderer, EpisodeResult
from utils import plot_learning_curve


class QLearningAgent(SARSAAgent):
    """
    Q-Learning agent for the Lunar Lander environment. 
    Inherits from SARSAAgent but implements the n-step Q-Learning update rule instead of SARSA.
    """

    @property
    def name(self) -> str:
        return "QLEARNING"

    def update(self, trajectory: List[Tuple], next_state_tuple=None, next_action=None, done=False):
        """
        Applies the n-step Q-Learning update rule.
        Computes the n-step return using the maximum Q-value for the next state (off-policy).

        :param trajectory: List of (state_tuple, action, reward) tuples for the n-step trajectory.
        :param next_state_tuple: The state after the trajectory.
        :param next_action: Ignored for Q-Learning (off-policy).
        :param done: Whether the episode ended.
        """
        if not trajectory:
            return
        
        # Compute n-step return
        G = 0
        for i, (s, a, r) in enumerate(trajectory):
            G += (self.gamma ** i) * r
        if not done and next_state_tuple is not None:
            G += (self.gamma ** self.n_step) * max(self.q_table[next_state_tuple])

        # Update the first state-action in trajectory
        s_update, a_update, _ = trajectory[0]
        current_q = self.q_table[s_update][a_update]
        new_q = current_q + self.alpha * (G - current_q)
        self.q_table[s_update][a_update] = new_q

    def load(self, path="QLearning_Agent_checkpoints/agent1/best_qtable_values.npy") -> SARSAAgent:
        return super().load(path)

    def train(self,
              env: LunarLanderEnv,
              episodes=1000,
              agent_type='QLearning',
              chkpt_path="QLearning_Agent_checkpoints/agent1",
              debug=False,
              logging_rate=500) -> List[EpisodeResult]:

        """ Call SARSA train method but with Q-Learning specific parameters. """
        return super().train(env=env,
                             episodes=episodes,
                             agent_type=agent_type,
                             chkpt_path=chkpt_path,
                             debug=debug,
                             logging_rate=logging_rate)

if __name__ == "__main__":
    # Environment
    num_actions = 4  # Main engine + left/right thrusters
    # lunar_env = SimpleLunarLanderEnv(num_actions=num_actions, debug=True)
    # lunar_env = LLE_XOffset(debug=True)
    # lunar_env = LLE_InitialVelocity(debug=True)
    lunar_env = LunarLanderEnv(debug=True, max_number_of_seconds=25)

    # Agent
    qlearning_agent = QLearningAgent(n_actions=num_actions, n_step=10, epsilon_decay=0.9995)  #.load()

    # Train the agent and save the Q-table
    history = qlearning_agent.train(lunar_env, episodes=50000, debug=True)
    plot_learning_curve([h['reward'] for h in history], agent_type='Q-Learning', ylim=(-150, 200))

    # Evaluate and render the trained agent
    qlearning_agent.evaluate(lunar_env, episodes=1000, debug=True)

    # Show a few episodes of the trained agent
    qlearning_agent.show_progress(lunar_env, episodes=5, save_gif=False, gif_path="QLearning_Agent_checkpoints/agent1", show_bins=True)