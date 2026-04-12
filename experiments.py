
from lunar_lander_env import LunarLanderEnv
import sarsa_agent
import qlearning_agent
import sarsa_lambda_agent
import DQN_agent
import numpy as np
import os

# Define the training configuration for each agent type
training_config = {
    "SARSA": {"num_tr_episodes": 50000},
    "QLearning": {"num_tr_episodes": 50000},
    "SARSA_Lambda": {"num_tr_episodes": 25000},
    "DQN": {"num_tr_episodes": 5000}
}

def train_and_collect_metrics(env,
                              agent_type:str = "SARSA",
                              num_agents:int = 5,
                              num_tr_episodes:int = 500,
                              outdir='training_results'):
    """
    Train multiple instances of a given agent type on the Lunar Lander environment, and collect the episodic returns during training.
    
    Args:
        env: The Lunar Lander environment.
        agent_type (str): The type of agent to train and evaluate (e.g., "SARSA", "Q-learning").
        num_agents (int): The number of agents to train and evaluate.
        num_tr_episodes (int): The number of training episodes for each agent.
        outdir (str): The directory to save the training results.
    Returns:
        None (saves the episodic returns to a CSV file for later analysis)
    """
    # Save the episodic returns for each training instance
    all_returns = []

    # DQN passes in the environment by class instead of instance
    env = env if agent_type == 'DQN' else env()

    if agent_type == "SARSA": agent_constructor = sarsa_agent.SARSAAgent
    elif agent_type == "QLearning": agent_constructor = qlearning_agent.QLearningAgent
    elif agent_type == "SARSA_Lambda": agent_constructor = sarsa_lambda_agent.SARSALambdaAgent
    elif agent_type == "DQN": agent_constructor = DQN_agent.DQNAgent
    else: raise TypeError(f"No constructor found for agent type '{agent_type}'")

    # Train the agents
    training_results_path = f"{outdir}/{agent_type}_Agents"
    for i in range(num_agents):
        print(f'\nagent {i}:')
        agent = agent_constructor()
        G_tr = agent.train(env=env,
                           episodes=num_tr_episodes,
                           chkpt_path=os.path.join(training_results_path, f"agent{i}"))
        # all_returns[i] = G_tr
        all_returns.append(G_tr)

    # DQN trains via steps, so # of episodes can vary across training instances
    if agent_type == "DQN":
        min_num_episodes = min([len(r) for r in all_returns])
        all_returns = np.array([r[:min_num_episodes] for r in all_returns])

    # Print average returns and standard deviations
    avg_G_i_tr = np.mean(all_returns, axis=0)
    print(f"\nAverage agent return during training: {np.mean(avg_G_i_tr)}, standard deviation: {np.std(avg_G_i_tr)}")
            
    # Save the returns to a CSV file
    filename = os.path.join(training_results_path, f"{agent_type}_agent_training_returns.csv")
    np.savetxt(filename, all_returns, delimiter=",")

def train_all_agents(num_agents:int = 20):
    """ 
    Train each agent types for several intances on the Lundar Lander environment. 
    
    Args:
        num_agents (int): The number of agents to train for each agent type.
    Returns:
        None (saves the episodic returns to CSV files for later analysis)
    """
    for agent_type in training_config.keys():
        print(f"\nTraining and evaluating {num_agents} {agent_type} agents...")
        num_tr_episodes = training_config[agent_type]["num_tr_episodes"]
        train_and_collect_metrics(env=LunarLanderEnv,
                                  agent_type=agent_type,
                                  num_agents=num_agents,
                                  num_tr_episodes=num_tr_episodes)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    train_all_agents()