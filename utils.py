import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(rewards_history, **config):
    # Parse plotting configuration
    agent_type = config.get('agent_type', "Awesome_RL_agent")
    environment = config.get('environment', "Lunar Lander")
    xlabel = config.get('xlabel', "Episodes")
    ylabel = config.get('ylabel', "Return (Total Reward)")
    xticks = config.get('xticks', None)
    xticks_step_factor = config.get('xticks_step_factor', 100)
    ylim = config.get('ylim', -50.0)
    save_path = config.get('save_path', f"{agent_type}_lc_plot.jpg")
    save_only = config.get('save_only', False)

    # Plot the learning curve
    plt.figure(figsize=(14, 5))
    plt.plot(rewards_history, alpha=0.5, color='gray', label=ylabel)

    # Calculate a moving average
    window = 20
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards_history)), moving_avg, color='blue', label='Moving Average (20 ep)')

    plt.title(f"{agent_type} Agent Learning Curve - {environment}", fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    if xticks is not None:
        plt.xticks(np.arange(0, len(rewards_history), step=xticks_step_factor), xticks, fontsize=14)
    plt.ylabel(ylabel, fontsize=16)
    plt.yticks(fontsize=14)
    plt.ylim(ylim)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)

    # Save the plot for later analysis
    plt.savefig(save_path)
    
    if save_only:
        plt.close()
        return
    
    plt.show()