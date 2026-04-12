import numpy as np
from sarsa_agent import SARSAAgent
from utils import plot_learning_curve
from lunar_lander_env import SimpleLunarLanderEnv, LLE_XOffset, LLE_InitialVelocity, LunarLanderEnv, Renderer

class QLearningAgent(SARSAAgent):
    """
    Q-Learning agent for the Lunar Lander environment. 
    Inherets from SARSAAgent but implements the Q-Learning update rule instead of SARSA.
    """
    def update(self, state_tuple, action, reward, next_state_tuple, next_action, done):
        """
        Update the Q-table based on the observed transition (state, action, reward, next state).
        The update is performed using the Q-Learning algorithm, which uses the maximum Q-value of the next state 
        to update the current Q-value.
        
        :param state_tuple: The current state represented as a tuple.
        :param action: The action taken in the current state.
        :param reward: The reward received after taking the action.
        :param next_state_tuple: The next state represented as a tuple.
        :param next_action: The action taken in the next state.
        :param done: A flag indicating whether the episode is done.
        """
        # q value for current state
        current_q = self.q_table[state_tuple][action]

        # Q-Learning Formula: Q(S,A) = Q(S,A) + alpha * [R + gamma * argmax_a(Q(S', A')) - Q(S,A)]
        next_q = 0 if done else max(self.q_table[next_state_tuple])
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state_tuple][action] = new_q

    def train(self,
              env,
              episodes=1000,
              agent_type='QLearning',
              chkpt_path="QLearning_Agent_checkpoints/agent1",
              debug=False,
              logging_rate=500) -> list:
        
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
    lunar_env = LunarLanderEnv(debug=True)

    # Agent
    qlearning_agent = QLearningAgent(n_actions=num_actions)

    # Train the agent and save the Q-table
    rewards = qlearning_agent.train(lunar_env, 
                                    episodes=50000, 
                                    debug=True)
    plot_learning_curve(rewards, agent_type='Q-Learning', ylim=(-150, 200))

    # Evaluate and render the trained agent
    qlearning_agent.evaluate(lunar_env, episodes=1000)

    # Show a few episodes of the trained agent
    qlearning_agent.show_progress(lunar_env, episodes=5)