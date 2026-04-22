import random
from collections import Counter
from typing import Dict, Iterable

import sarsa_agent
import qlearning_agent
import sarsa_lambda_agent
import DQN_agent
from lunar_lander_env import LunarLanderEnv
from utils import plot_learning_curve
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import torch
import numpy as np
import pandas as pd
import glob
import os
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configuration for evaluation of the trained agents (e.g., checkpoint names, plotting configurations, etc.)
EVAL_CONFIGS = {
    'SARSA': {
        'ckpt_name': 'best_qtable_values.npy',
        'num_convergence_eps': 5000,
        'ylabel': "Avg Return (20 agents)",
        'ylim': (-60, 100),
        'measure_latency_runs': 100000
    },
    'QLearning': {
        'ckpt_name': 'best_qtable_values.npy',
        'num_convergence_eps': 5000,
        'ylabel': "Avg Return (20 agents)",
        'ylim': (-60, 100),
        'measure_latency_runs': 100000
    },
    'SARSA_Lambda': {
        'ckpt_name': 'best_weights.npy',
        'num_convergence_eps': 2500,
        'ylabel': "Avg Return (20 agents)",
        'ylim': (-50, 220),
        'measure_latency_runs': 10000
    },
    'DQN': {
        'ckpt_name': 'best_weights.pth',
        'num_convergence_eps': 125,  # (125 data points * 8 envs = 1000 total episodes)
        'ylabel': "Avg Return (20 agents)",
        'ylim': (-30, 220),
        'measure_latency_runs': 1000
    }
}

AGENT_TYPES = EVAL_CONFIGS.keys()

def DELETME_measure_avg_training_duration(agent_type, training_results_path='training_results'):
    """
    # TODO: DELETEME
    Measure the average training durations using saved policy file timestamps. This is a workaround
    since the training durations weren't saved during training (which took about 28 hours!).
    Preferably, going forward we should save the training times to a file for later analysis
    immediately after a training run has completed.

    Args:
        agent_type (str): type of agent to measure training duration for (e.g., 'SARSA', 'QLearning', 'SARSA_Lambda', 'DQN')
        training_results_path (str): path to the directory containing the training results
    Returns:
        str: average training duration in "MM:SS" format
    """
    all_train_durations = []
    policy_dirs = glob.glob(f"{training_results_path}/{agent_type}_Agents/agent*", recursive=True)
    ckpt_name = EVAL_CONFIGS[agent_type]['ckpt_name']
    ckpt_paths = []
    for d in policy_dirs:
        ckpt_paths.extend(p for p in Path(d).rglob(ckpt_name) if p.is_file())

    if not ckpt_paths:
        raise(f"Didn't find any policy checkpoints for {agent_type} agents")

    # Resort by agent number since glob mangles the file listing order
    sorted_ckpt_paths = sorted(ckpt_paths, key=lambda x: int(str(x).split("\\")[2].partition('agent')[2]))

    for i in range(1, len(sorted_ckpt_paths)):
        current = os.path.getctime(sorted_ckpt_paths[i])
        previous = os.path.getctime(sorted_ckpt_paths[i-1])
        timestamp1 = time.ctime(previous)
        timestamp2 = time.ctime(current)
        start = datetime.strptime(timestamp1, "%a %b %d %H:%M:%S %Y")
        end = datetime.strptime(timestamp2, "%a %b %d %H:%M:%S %Y")
        delta = end - start
        total_seconds = delta.total_seconds()
        all_train_durations.append(total_seconds)

    avg_time = np.mean(all_train_durations)
    return datetime.fromtimestamp(avg_time).strftime("%M:%S")

def measure_avg_training_duration(agent_type, training_results_path='training_results'):
    """
    Measure the average training durations for this agent type using the duration times saved during training.

    Args:
        agent_type (str): type of agent to measure training duration for (e.g., 'SARSA', 'QLearning', 'SARSA_Lambda', 'DQN')
        training_results_path (str): path to the directory containing the training results
    Returns:
        str: average training duration in "MM:SS" format
    """
    filename = os.path.join(training_results_path, f"{agent_type}_Agents/training_durations.npy")
    durations = np.load(filename)
    avg_duration = np.mean(durations)
    return datetime.fromtimestamp(avg_duration).strftime("%M:%S")

def training_metrics(agent_types=AGENT_TYPES, training_results_path='training_results'):
    """ 
    Process the training results and produce learning curve plots for each agent type.
    
    Args:
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
        training_results_path (str): path to the directory containing the training results.
    Returns:
        None
    """
    for agent_type in agent_types:
        print(f"\n--- {agent_type} agents Training Metrics ---")

        # Read in training stats
        stats_path = f"{training_results_path}/{agent_type}_Agents"
        df = pd.read_csv(f"{stats_path}/training_returns.csv", header=None)

        # Compute average returns across each agent instance
        avg_G_i_tr = np.mean(df, axis=0)
        print(f"Avg return during training: {np.mean(avg_G_i_tr)}, standard deviation: {np.std(avg_G_i_tr)}")

        # Prove that the number of training episodes was sufficient
        # I.e., show that the last several data points resulted in minimal improvement
        convergence_eps = EVAL_CONFIGS[agent_type]['num_convergence_eps']
        last_several_pts = avg_G_i_tr[-convergence_eps:]
        avg_abs_change = last_several_pts.diff()
        s = pd.Series(last_several_pts)
        avg_percent_change = s.pct_change()
        convergence_eps = convergence_eps * 8 if agent_type in 'DQN' else convergence_eps
        print(f"Last {convergence_eps} training episodes: avg_abs_change={np.mean(avg_abs_change):.6f}, avg_percent_change={100*avg_percent_change.mean():.2f}%")

        # Compute average training duration for this agent type
        avg_train_time = measure_avg_training_duration(agent_type=agent_type)
        print(f"Avg training duration: {avg_train_time}")

        # Get learning curve plot configuration for this agent type
        ylim=EVAL_CONFIGS[agent_type].get('ylim', (-150, 200))
        ylabel = EVAL_CONFIGS[agent_type].get('ylabel', "Return (Total Reward)")

        # Update xticks for DQN since it was trained using multiple environments
        xticks=None
        if agent_type in 'DQN':
            num_parallel_envs = 8
            step_factor = 100
            xticks = np.arange(0, num_parallel_envs * len(avg_G_i_tr), step=(num_parallel_envs * step_factor))

        # Add hyphen to "QLearning" for better readability
        agent_type = 'Q-Learning' if agent_type in 'QLearning' else agent_type

        # Save a learning curve plot using the average returns across agent instances
        plot_name = f"{training_results_path}/{agent_type}_lc_plot.jpg"
        print(f"Saving learning curve plot to '{f"{training_results_path}/{agent_type}_lc_plot.jpg"}'...")
        plot_learning_curve(avg_G_i_tr,
                            agent_type=f"{agent_type}",
                            ylabel=ylabel,
                            ylim=ylim,
                            xticks=xticks,
                            save_path=plot_name,
                            save_only=True)

def evaluate_agents(agent_types: Iterable[str]=AGENT_TYPES, training_results_path="training_results", episodes=1000, outdir="test_results") -> Dict:
    """
    Evaluate the trained agents on Lunar Lander and save the test results (returns, episode durations, landing results) for each agent type.

    Args:
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
        training_results_path (str): path to the directory containing the training results
        episodes (int): number of episodes to evaluate each agent for
        outdir (str): path to the directory where test results will be saved
    Returns:
        None
    """
    all_history = {a: None for a in agent_types}

    print("--- Agent Evaluation ---")
    for agent_type in agent_types:
        results_path = os.path.join(outdir, agent_type)
        os.makedirs(results_path, exist_ok=True)

        if agent_type == "SARSA": agent_constructor = sarsa_agent.SARSAAgent
        elif agent_type == "QLearning": agent_constructor = qlearning_agent.QLearningAgent
        elif agent_type == "SARSA_Lambda": agent_constructor = sarsa_lambda_agent.SARSALambdaAgent
        elif agent_type == "DQN": agent_constructor = DQN_agent.DQNAgent
        else: raise TypeError(f"No constructor found for agent type '{agent_type}'")

        # Gather all trained agents for this agent type
        stats_path = f"{training_results_path}/{agent_type}_Agents"
        agent_dirs = glob.glob(f"{stats_path}/agent*", recursive=True)
        ckpt_dirs = [d for d in agent_dirs if os.path.isdir(d)]

        # Resort the ckpt directories since glob mangles the file listing order
        sorted_ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(str(x).split("\\")[2].partition('agent')[2]))

        number_of_agents = len(sorted_ckpt_dirs)

        # Evaluate all trained agents of this type
        agent = agent_constructor()
        env = LunarLanderEnv()
        episode_histories = []
        ckpt_filename = EVAL_CONFIGS[agent_type]['ckpt_name']

        print(f"\nEvaluating {number_of_agents} {agent_type} agents, each over {episodes} episodes...")

        for i in tqdm(range(number_of_agents)):
            agent.load(f"{ckpt_dirs[i]}/{ckpt_filename}")
            episode_histories.append(agent.evaluate(env=env, episodes=episodes))

        rewards = [h['reward'] for episode_history in episode_histories for h in episode_history]
        durations = [h['steps'] for episode_history in episode_histories for h in episode_history]

        crashes = [h['crashed'] for episode_history in episode_histories for h in episode_history]
        lands = [h['landed'] for episode_history in episode_histories for h in episode_history]
        timeout = [h['timeout'] for episode_history in episode_histories for h in episode_history]
        crash_counts = Counter(tuple(sorted(h['crash_reason'], key=lambda r: r.value))
                               for episode_history in episode_histories
                               for h in episode_history if not h.get('landed'))  # shape: (num_agents, episodes)

        # Compute average stats across each agent instance
        print(f"Avg return: {np.mean(rewards)}, standard deviation: {np.std(rewards)}")
        print(f"Avg episode duration: {np.mean(durations)}, standard deviation: {np.std(durations)}")
        percent_timeout = 100 * np.mean(timeout)
        percent_crashed = 100 * np.mean(crashes)
        percent_landed = 100 * np.mean(lands)
        print(f'Avg landing results: timeout={percent_timeout:.2f}%, crashed={percent_crashed:.2f}%, landed={percent_landed:.2f}%')

        # Save test results
        print(f"Saving test results to '{results_path}'")
        np.save(os.path.join(results_path, 'evaluation_history.npy'), episode_histories)

        all_history[agent_type] = episode_histories
    
    return all_history

def statistical_significance(test_results_path='test_results', agent_types=AGENT_TYPES, alpha=0.05):
    """
    Determine the statistical significance of the evaluation results by performing a one-way ANOVA test on the average returns of the 
    different agent types, followed by a Tukey HSD test to determine which agent types are significantly different from each other.

    Args:
        test_results_path (str): path to the directory containing the test results
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
        alpha (float): significance level for the statistical tests
    Returns:
        None
    """
    eval_returns = {a: np.load(f"{test_results_path}/{a}/returns.npy") for a in agent_types}

    print("\n" + "="*50 + "\n")
    print(f"Determining statistical significance of the agents' avg returns...")

    print(f"--- Pre-requisites: Shapiro-Wilk Test and Levene's Test ---")

    # Shapiro-Wilk Test (Testing for Normality)
    avg_returns = []
    for a in agent_types:
        avg_G_i_ts = np.mean(eval_returns[a], axis=0)
        shapiro_group = stats.shapiro(avg_G_i_ts)
        pval = shapiro_group.pvalue
        if pval < alpha:
            print(f"WARNING: Shapiro-Wilk Test failed: {a} agent returns: p-value = {shapiro_group.pvalue:.4f}")
        avg_returns.append(avg_G_i_ts)

    # Levene's Test (Testing for Homogeneity of Variance)
    levene_stat, levene_p = stats.levene(*avg_returns)
    if levene_p < alpha:
        print(f"WARNING: Levene's Test failed: Statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")

    # Perform One-Way ANOVA Test
    f_stat, p_value = f_oneway(*avg_returns)

    print("\n--- ANOVA Test (Statistical Significance) ---")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < alpha:
        print("Result: Statistically significant (Reject Null Hypothesis)")
    else:
        print("Result: Not statistically significant (Fail to Reject Null Hypothesis)")

    # Turkey HSD
    print("\n--- Turkey HSD (statistical difference between agent types) ---")
    num_agents = len(avg_returns)
    num_ts_episodes = len(avg_returns[0])
    group_names = [a for a in agent_types for _ in range(num_ts_episodes)]
    avg_returns_concat = np.array(avg_returns).reshape((num_agents * num_ts_episodes, -1))
    tukey = pairwise_tukeyhsd(endog=avg_returns_concat, groups=group_names, alpha=alpha)
    print(tukey)

    # # Convert to DataFrame
    # tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
    #                         columns=tukey._results_table.data[0])
    # print(tukey_df)

def find_largest_policy(agent_type, training_results_path="training_results"):
    """ Helper function to find the trained agent policy with the largest memory footprint. """
    policy_dirs = glob.glob(f"{training_results_path}/{agent_type}_Agents/agent*", recursive=True)
    ckpt_name = EVAL_CONFIGS[agent_type]['ckpt_name']
    ckpt_paths = []
    for d in policy_dirs:
        ckpt_paths.extend(p for p in Path(d).rglob(ckpt_name) if p.is_file())

    if not ckpt_paths:
        return None

    largest = max(ckpt_paths, key=lambda f: f.stat().st_size)
    filesize = largest.stat().st_size / (1024**2)  # Compute in MB
    return largest, filesize

def measure_memory_footprint(agent_types=AGENT_TYPES, training_results_path="training_results"):
    """ Helper function to measure average memory footprint for agent type's policy """
    avg_file_sizes = {}
    for a in agent_types:
        policy_dirs = glob.glob(f"{training_results_path}/{a}_Agents/agent*", recursive=True)
        ckpt_name = EVAL_CONFIGS[a]['ckpt_name']
        ckpt_paths = []
        for d in policy_dirs:
            ckpt_paths.extend(p for p in Path(d).rglob(ckpt_name) if p.is_file())

        if not ckpt_paths:
            raise(f"Didn't find any policy checkpoints for {a} agents")

        avg_size = np.mean([f.stat().st_size for f in ckpt_paths])
        avg_file_sizes[a] = avg_size / (1024**2)  # Measure in MB

    return avg_file_sizes

def measure_action_latency(agent_types=AGENT_TYPES):
    """
    Measure the average action latency for each agent type by loading the largest policy checkpoint (in terms of file size)
    for that agent type and measuring the time taken to select an action over multiple runs.

    Args:
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
    Returns:
        dict: average action latency (in milliseconds) for each agent type
    """
    env = LunarLanderEnv()
    obs = env.reset()
    avg_latencies = {a: 0.0 for a in agent_types}

    for a in agent_types:
        if a == "SARSA": agent_constructor = sarsa_agent.SARSAAgent
        elif a == "QLearning": agent_constructor = qlearning_agent.QLearningAgent
        elif a == "SARSA_Lambda": agent_constructor = sarsa_lambda_agent.SARSALambdaAgent
        elif a == "DQN": agent_constructor = DQN_agent.DQNAgent
        else: raise TypeError(f"No constructor found for agent type '{a}'")

        agent = agent_constructor()

        # Find the most costly policy based on file size (to measure worse case latency)
        largest_policy, size = find_largest_policy(agent_type=a)
        # print(f"Largest {a} agent policy: {size:.1f} MB")
        agent.load(largest_policy)
        n_runs = EVAL_CONFIGS[a].get('measure_latency_runs', 1000)
        device = 'cpu'
        start, end = 0.0, 0.0

        if a in "DQN":
            agent.q_network.eval()
            obs = obs.to_array()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Warm-up (important for fair timing)
            for _ in range(100):
                _ = agent.act(obs, evaluate=True)

            torch.cuda.synchronize() if device == "cuda" else None
            start = time.perf_counter()
            for _ in range(n_runs):
                _ = agent.act(obs, evaluate=True)
            torch.cuda.synchronize() if device == "cuda" else None
            end = time.perf_counter()

        elif a in ['SARSA', 'QLearning']:
            start = time.perf_counter()
            for _ in range(n_runs):
                obs_discrete = agent.discretize(obs)  # Include time for state discretization
                _ = agent.act(obs_discrete, evaluate=True)
            end = time.perf_counter()

        elif a in 'SARSA_Lambda':
            start = time.perf_counter()
            for _ in range(n_runs):
                tiles = agent.tc.get_tiles(obs)  # Include time for state tiling
                _ = agent.act(tiles, evaluate=True)
            end = time.perf_counter()

        else:
            raise(f"Cannot measure action latency for {a} agent")

        avg_latency = 1000 * (end - start) / n_runs  # miliseconds
        avg_latencies[a] = avg_latency

    return avg_latencies

def measure_env_latency(n_runs=100000):
    """ Helper function to compute average environment step latency """
    env = LunarLanderEnv()
    env.reset()
    start = time.perf_counter()
    for _ in range(n_runs):
        a = random.randrange(len(env.action_space))
        env.step(a)
    end = time.perf_counter()
    return 1000 * (end - start) / n_runs  # miliseconds

def computational_efficiency(agent_types=AGENT_TYPES):
    """
    Measure average action latency, steps per second, and memory footprint for each agent type.
    The results can be used to compare the tradeoffs vs. performance within Lunar Lander.

    Args:
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
    Returns:
        None
    """
    # Measure average action policy latency for each agent type
    print('\n--- Action Policy Latency ---')
    latencies = {a: [] for a in agent_types}
    for i in tqdm(range(5)):
        latency = measure_action_latency(agent_types=agent_types)
        for a in agent_types:
            latencies[a].append(latency[a])

    avg_latencies = {}
    for k, v in latencies.items():
        avg_latencies[k] = np.mean(v)
        print(f"Avg {k} agent latency: {avg_latencies[k]:.3f} ms")

    # Measure avg steps per second (i.e., 1 / [avg_action_latency + time(env_step)])
    print(f"\n--- Avg Steps per second ---")
    avg_env_step_latency = measure_env_latency()
    print(f"Avg env step latency: {avg_env_step_latency:.6f} ms")
    for k, v in avg_latencies.items():
        steps_per_second = 1 / (v + avg_env_step_latency)  # steps / ms
        steps_per_second *= 1000  # steps / s
        print(f"{k}: {int(steps_per_second)} steps / s")

    # Measure avg memory footprint for each agent policy
    print(f"\n --- Avg Memory Footprint ---")
    avg_footprints = measure_memory_footprint()
    for k, v in avg_footprints.items():
        print(f"{k}: {v:.1f} MB")

def collect_qualitative_examples(agent_types=AGENT_TYPES,
                                 train_results_path='training_results',
                                 test_results_path='test_results',
                                 num_of_examples=5,
                                 outdir='qualitative_examples'):
    """ 
    Collect qualitative examples (e.g., GIFs) of the agents executing their policies in Lunar Lander.

    Args:
        agent_types (list): list of agent types (e.g., ['SARSA', 'QLearning', 'SARSA_Lambda', 'DQN'])
        train_results_path (str): path to the directory containing the training results
        test_results_path (str): path to the directory containing the test results
        num_of_examples (int): number of episode examples to collect for each agent type
        outdir (str): path to the directory where the collected examples will be saved
    Returns:
        None
    """
    env = LunarLanderEnv()

    # Collect GIFs of the best agent instance for each agent type running in the Lunar Lander environment
    for a in agent_types:
        print(f"\n--- Best {a} agent ---")
        if a == "SARSA": agent_constructor = sarsa_agent.SARSAAgent
        elif a == "QLearning": agent_constructor = qlearning_agent.QLearningAgent
        elif a == "SARSA_Lambda": agent_constructor = sarsa_lambda_agent.SARSALambdaAgent
        elif a == "DQN": agent_constructor = DQN_agent.DQNAgent
        else: raise TypeError(f"No constructor found for agent type '{a}'")

        # Determine the best performing agent instance based on the evaluation results
        landing_results = np.load(os.path.join(test_results_path, a, "landing_results.npy"), allow_pickle=True)
        best_agent_idx = max(range(len(landing_results)), key=lambda i: landing_results[i]['landed'])
        num_lands_during_eval = landing_results[best_agent_idx]['landed']
        print(f"Choosing {a} agent number {best_agent_idx} since it had {num_lands_during_eval} lands during evaluation...")

        # Load the best checkpoint for this agent
        agent = agent_constructor()
        ckpt = os.path.join(train_results_path, f"{a}_Agents/agent{best_agent_idx}", EVAL_CONFIGS[a]['ckpt_name'])
        agent.load(ckpt)

        # Save the GIFs to outdir
        agent.show_progress(env=env, episodes=num_of_examples, save_gif=True, gif_path=os.path.join(outdir, a))

    # Show learning progression of the best DQN agent
    print(f"\n--- Learning Progression of the Best DQN agent ---")
    landing_results = np.load(os.path.join(test_results_path, "DQN", "landing_results.npy"), allow_pickle=True)
    best_agent_idx = max(range(len(landing_results)), key=lambda i: landing_results[i]['landed'])

    agent = DQN_agent.DQNAgent()
    best_agent_ckpts_path = f"{train_results_path}/DQN_Agents/agent{best_agent_idx}"
    ckpt_dirs = glob.glob(f"{best_agent_ckpts_path}/ckpt*.pth", recursive=True)

    for ckpt in ckpt_dirs:
        print(f"Loading checkpoint '{ckpt}'...")
        agent.load(ckpt)

        # Save the GIFs to outdir
        agent.show_progress(env=env, episodes=1, save_gif=True, gif_path=os.path.join(outdir, "DQN_learning_progression"))


if __name__=='__main__':
    # Process the training results and produce learning curve plots
    training_metrics()

    # Evaluate the trained agents on Lunar Lander
    evaluate_agents()

    # Determine the statistical significance of the evaluation results
    statistical_significance()

    # Measure computational efficiency of the agents (e.g., inference time)
    computational_efficiency()

    # Record videos of the agents executing their policies in Lunar Lander
    collect_qualitative_examples()