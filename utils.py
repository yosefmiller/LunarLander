import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(rewards_history: list, episode_intervals = None, agent_type='SARSA', title="Lunar Lander", ylabel='Total Reward'):
    # Plotting the learning curve
    plt.figure(figsize=(10, 5))
    if episode_intervals is None:
        plt.plot(rewards_history, alpha=0.5, color='gray', label='Raw Reward')
    else:
        episodes = np.cumsum(episode_intervals)
        plt.plot(episodes, rewards_history, alpha=0.5, color='gray', label='Raw Reward')

    # Calculate a moving average
    window = 20
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards_history)), moving_avg, color='blue', label='Moving Average (20 ep)')

    plt.title(f"{agent_type} Agent Learning Curve - {title}")
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()