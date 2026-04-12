
import sarsa_agent
import qlearning_agent
import sarsa_lambda_agent
import DQN_agent
# from lunar_lander_env import LunarLanderEnv, Renderer
from upgraded_lunar_lander_env import LunarLanderEnv, Renderer
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
from tqdm import tqdm

# Configuration for evaluation of the trained agents (e.g., checkpoint names, plotting configurations, etc.)
EVAL_CONFIGS = {
    'SARSA': {
        'ckpt_name': 'best_qtable_values.npy',
        'num_convergence_eps': 5000,
        'ylim': (-60, 100)
    },
    'QLearning': {
        'ckpt_name': 'best_qtable_values.npy',
        'num_convergence_eps': 5000,
        'ylim': (-60, 100)
    },
    'SARSA_Lambda': {
        'ckpt_name': 'best_weights.npy',
        'num_convergence_eps': 2500,
        'ylim': (-50, 220)
    },
    'DQN': {
        'ckpt_name': 'best_weights.pth',
        'num_convergence_eps': 125,  # (125 data points * 8 envs = 1000 total episodes)
        'ylim': (-30, 220)
    }
}

AGENT_TYPES = EVAL_CONFIGS.keys()

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
        print(f"\n{agent_type} agents training metrics:")

        # Read in training stats
        stats_path = f"{training_results_path}/{agent_type}_Agents"
        df = pd.read_csv(f"{stats_path}/{agent_type}_agent_training_returns.csv", header=None)

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

        # Get learning curve plot configuration for this agent type
        ylim=EVAL_CONFIGS[agent_type].get('ylim', (-150, 200))

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
                            ylim=ylim,
                            xticks=xticks,
                            save_path=plot_name,
                            save_only=True)

def evaluate_agents(agent_types=AGENT_TYPES, training_results_path="training_results", episodes=1000, outdir="test_results"):
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
    all_returns = {a: None for a in agent_types}

    for agent_type in AGENT_TYPES:
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
        number_of_agents = len(ckpt_dirs)

        # Evaluate all trained agents of this type
        agent = agent_constructor()
        env = LunarLanderEnv()
        returns = np.zeros((number_of_agents, episodes))
        avg_eps_durations = np.zeros(number_of_agents)
        landing_results = []
        ckpt_filename = EVAL_CONFIGS[agent_type]['ckpt_name']

        print(f"\nEvaluating {number_of_agents} {agent_type} agents, each over {episodes} episodes...")

        for i in tqdm(range(number_of_agents)):
            agent.load(f"{ckpt_dirs[i]}/{ckpt_filename}")
            total_rewards, avg_duration, results = agent.evaluate(env=env, episodes=episodes)
            returns[i] = np.array(total_rewards)
            avg_eps_durations[i] = avg_duration
            landing_results.append(results)

        # Compute average stats across each agent instance
        avg_G_i_ts = np.mean(returns, axis=0)
        print(f"Avg return: {np.mean(avg_G_i_ts)}, standard deviation: {np.std(avg_G_i_ts)}")
        print(f"Avg episode duration: {np.mean(avg_eps_durations)}, standard deviation: {np.std(avg_eps_durations)}")
        num_max_steps_exceeded = np.array([i['exceeded_max_steps'] for i in landing_results])
        crashes = np.array([i['crashed'] for i in landing_results])
        lands = np.array([i['landed'] for i in landing_results])
        percent_max_steps_exceeded = 100 * np.mean(num_max_steps_exceeded) / episodes
        percent_crashed = 100 * np.mean(crashes) / episodes
        percent_landed = 100 * np.mean(lands) / episodes
        print(f'Avg landing results: max_steps_exceeded={percent_max_steps_exceeded:.2f}%, crashed={percent_crashed:.2f}%, landed={percent_landed:.2f}%')

        # save test results
        print(f"Saving test results to '{results_path}'")
        np.save(os.path.join(results_path, 'returns.npy'), returns)
        np.save(os.path.join(results_path, 'avg_eps_durations.npy'), avg_eps_durations)
        np.save(os.path.join(results_path, 'landing_results.npy'), landing_results)

        all_returns[agent_type] = returns
    
    return all_returns

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


# # inference time test
# def benchmark(_model, _x, warmup=10, runs=50):
#     _model.eval()
#     with torch.no_grad():
#         for _ in range(warmup):  # warm up
#             _ = model(_x)
#         start = time.time()
#         for _ in range(runs):
#             _ = model(_x)
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         end = time.time()
#     return (end-start)/runs*1000  # ms per inference

def computational_efficiency():
    """
    TODO: Measure Average action latency, memory footprint for each agent type. 
    Compare the tradeoffs vs. performance within Lunar Lander.
    """
    print("Inference Time")
    # for name, model in [('R0.0', m0), ('R1.0', m1)]:
    #     t = benchmark(model, x)
    #     print(f"{name:<8}: {t:.2f} ms / inference")

def collect_qualitative_examples(agent_types=AGENT_TYPES,
                                 train_results_path='training_results',
                                 test_results_path='test_results',
                                 num_of_examples=5,
                                 outdir='qualitative_examples'):
    """ 
    TODO: Collect qualitative examples (e.g., videos of episodes) of the agents executing their policies in Lunar Lander.
    """
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
        env = LunarLanderEnv()
        agent = agent_constructor()
        ckpt = os.path.join(train_results_path, f"{a}_Agents/agent{best_agent_idx}", EVAL_CONFIGS[a]['ckpt_name'])
        agent.load(ckpt)

        # TODO: save the GIFs to outdir
        # agent.show_progress(env=env, episodes=num_of_examples, save_gif=True, gif_path=os.path.join(outdir, a))
        agent.show_progress(env=env, episodes=num_of_examples)

    # Show learning progression of the best DQN agent
    print(f"\n--- Learning Progression of the Best DQN agent ---")
    landing_results = np.load(os.path.join(test_results_path, "DQN", "landing_results.npy"), allow_pickle=True)
    best_agent_idx = max(range(len(landing_results)), key=lambda i: landing_results[i]['landed'])

    env = LunarLanderEnv()
    agent = DQN_agent.DQNAgent()
    best_agent_ckpts_path = f"{train_results_path}/DQN_Agents/agent{best_agent_idx}"
    ckpt_dirs = glob.glob(f"{best_agent_ckpts_path}/ckpt*.pth", recursive=True)

    for ckpt in ckpt_dirs:
        print(f"Loading checkpoint '{ckpt}'...")
        agent.load(ckpt)

        # TODO: save the GIFs to outdir
        # agent.show_progress(env=env, episodes=num_of_examples, save_gif=True, gif_path=os.path.join(outdir, a))
        agent.show_progress(env=env, episodes=num_of_examples)



if __name__=='__main__':
    # Process the training results and produce learning curve plots
    training_metrics()

    # Evaluate the trained agents on Lunar Lander
    evaluate_agents()

    # Determine the statistical significance of the evaluation results
    statistical_significance()

    # Measure computational efficiency of the agents (e.g., inference time)
    # computational_efficiency()

    # Record videos of the agents executing their policies in Lunar Lander
    collect_qualitative_examples()