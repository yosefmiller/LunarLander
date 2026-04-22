from lunar_lander_env import LunarLanderEnv
import sarsa_agent
import qlearning_agent
import sarsa_lambda_agent
import DQN_agent
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import cast

# Define the training configuration for each agent type
training_config = {
    "SARSA": {"num_tr_episodes": 50000, "max_parallel_workers": 10},
    "SARSA_10": {"num_tr_episodes": 50000, "max_parallel_workers": 10},
    "QLearning": {"num_tr_episodes": 50000, "max_parallel_workers": 10},
    "QLearning_10": {"num_tr_episodes": 50000, "max_parallel_workers": 10},
    "SARSA_Lambda": {"num_tr_episodes": 25000, "max_parallel_workers": 10},
    "DQN": {"num_tr_episodes": 5000, "max_parallel_workers": 2},
    "DoubleDQN": {"num_tr_episodes": 5000, "max_parallel_workers": 2}
}

AGENT_CONSTRUCTORS = {
    "SARSA": sarsa_agent.SARSAAgent,
    "SARSA_10": sarsa_agent.SARSAAgent10Step,
    "QLearning": qlearning_agent.QLearningAgent,
    "QLearning_10": qlearning_agent.QLearningAgent10Step,
    "SARSA_Lambda": sarsa_lambda_agent.SARSALambdaAgent,
    "DQN": DQN_agent.DQNAgent,
    "DoubleDQN": DQN_agent.DoubleDQNAgent,
}

def _train_single_agent(agent_type: str,
                        agent_idx: int,
                        num_tr_episodes: int,
                        training_results_path: str,
                        env_factory):
    """Top-level worker for Windows spawn: instantiate env/agent inside each process."""
    agent_constructor = AGENT_CONSTRUCTORS.get(agent_type)
    if agent_constructor is None:
        raise TypeError(f"No constructor found for agent type '{agent_type}'")

    # print(f"\nagent {agent_idx}:")
    agent = agent_constructor()

    # DQN agents accept an env class; tabular agents train on an env instance.
    env_obj = env_factory if agent_type in {"DQN", "DoubleDQN"} else env_factory()
    history_tr = agent.train(
        env=env_obj,
        episodes=num_tr_episodes,
        chkpt_path=os.path.join(training_results_path, f"agent{agent_idx}"),
    )

    return agent_idx, [ep["reward"] for ep in history_tr]

def train_and_collect_metrics(env,
                              agent_type:str = "SARSA",
                              num_agents:int = 5,
                              num_tr_episodes:int = 500,
                              max_parallel_workers: int | None = None,
                              outdir='training_results'):
    """
    Train multiple instances of a given agent type on the Lunar Lander environment, and collect the episodic returns during training.
    
    Args:
        env: The Lunar Lander environment.
        agent_type (str): The type of agent to train and evaluate (e.g., "SARSA", "Q-learning").
        num_agents (int): The number of agents to train and evaluate.
        num_tr_episodes (int): The number of training episodes for each agent.
        max_parallel_workers (int | None): Optional cap for process workers. If None,
            uses per-agent config and defaults to num_agents.
        outdir (str): The directory to save the training results.
    Returns:
        None (saves the episodic returns to a CSV file for later analysis)
    """
    # Save the episodic returns for each training instance
    all_returns = []
    env_factory = env if isinstance(env, type) else type(env)

    if agent_type not in AGENT_CONSTRUCTORS:
        raise TypeError(f"No constructor found for agent type '{agent_type}'")

    # Train the agents
    training_results_path = f"{outdir}/{agent_type}_Agents"
    os.makedirs(training_results_path, exist_ok=True)

    configured_max_workers = training_config.get(agent_type, {}).get("max_parallel_workers", num_agents)
    worker_cap = configured_max_workers if max_parallel_workers is None else max_parallel_workers
    max_workers = max(1, min(num_agents, worker_cap))
    print(f"Using {max_workers} parallel workers for {agent_type} ({num_agents} agents total).")

    results_by_agent: list[list[float] | None] = [None] * num_agents
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _train_single_agent,
                agent_type,
                i,
                num_tr_episodes,
                training_results_path,
                env_factory,
            )
            for i in range(num_agents)
        ]

        for future in as_completed(futures):
            agent_idx, episode_returns = future.result()
            results_by_agent[agent_idx] = episode_returns

    if any(r is None for r in results_by_agent):
        raise RuntimeError("One or more agents did not return training results.")
    all_returns = cast(list[list[float]], [r for r in results_by_agent if r is not None])

    # DQN trains via steps, so # of episodes can vary across training instances
    if agent_type in {"DQN", "DoubleDQN"}:
        min_num_episodes = min([len(r) for r in all_returns])
        all_returns = np.array([r[:min_num_episodes] for r in all_returns])

    # Print average returns and standard deviations
    avg_G_i_tr = np.mean(all_returns, axis=0)
    print(f"\nAverage agent return during training: {np.mean(avg_G_i_tr)}, standard deviation: {np.std(avg_G_i_tr)}")
            
    # Save the returns to a CSV file
    filename = os.path.join(training_results_path, "training_returns.csv")
    np.savetxt(filename, all_returns, delimiter=",")

def train_all_agents(num_agents:int = 20, outdir='training_results'):
    """ 
    Train each agent types for several instances on the Lundar Lander environment.
    
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
                                  num_tr_episodes=num_tr_episodes,
                                  outdir=outdir)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    train_all_agents()