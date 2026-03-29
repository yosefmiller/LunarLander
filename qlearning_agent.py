from sarsa_agent import SARSAAgent, train_agent, plot_learning_curve, evaluate_agent
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

if __name__ == "__main__":
    # Environment
    num_actions = 4  # Main engine + left/right thrusters
    # lunar_env = SimpleLunarLanderEnv(num_actions=num_actions)
    # lunar_env = LLE_XOffset()
    lunar_env = LLE_InitialVelocity()
    # lunar_env = LunarLanderEnv()

    # Agent
    qlearning_agent = QLearningAgent(n_actions=num_actions, epsilon_decay=0.9995, alpha=0.4, epsilon_min=0.05)

    # Train the agent and save the Q-table
    rewards = train_agent(lunar_env, qlearning_agent, episodes=50000, agent_type='Q-Learning')
    plot_learning_curve(rewards, "Simple Lunar Lander")

    # Evaluate and render the trained agent
    evaluate_agent(lunar_env, qlearning_agent, num_episodes=5)