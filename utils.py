from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from lunar_lander_env import FPS, EpisodeResult, CrashReason


def plot_learning_curve(rewards_history: List[float], **config):
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
    windows = config.get('windows', [
        (len(rewards_history)//1000, 'gray'),
        (len(rewards_history)//250, 'blue'),
        # (len(rewards_history)//100, 'green')
    ])

    # Plot the learning curve
    plt.figure(figsize=(14, 5))
    # plt.plot(rewards_history, alpha=0.5, color='gray', label=ylabel)

    # Calculate a moving average
    for window, color in windows:
        if len(rewards_history) >= window:
            moving_avg = np.convolve(rewards_history, np.full(window, 1/window), mode='valid')
            plt.plot(np.arange(window-1, len(rewards_history)), moving_avg, color=color, linewidth=1.0, label=f'Moving Average ({window} ep)')

    plt.title(f"{agent_type} Agent Learning Curve - {environment}", fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    if xticks is not None:
        plt.xticks(np.arange(0, len(rewards_history), step=xticks_step_factor), xticks)
    plt.xticks(fontsize=14)
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


def plot_outcomes(history: List[EpisodeResult], **config):
    """Plot running averages for landed/timeout and each observed crash-reason combination."""
    if not history:
        raise ValueError("history must contain at least one episode")

    agent_type = config.get('agent_type', 'SARSA')
    environment = config.get('environment', 'Lunar Lander')
    xlabel = config.get('xlabel', 'Episodes')
    ylabel = config.get('ylabel', 'Running Average Outcome Rate')
    save_path = config.get('save_path', f"{agent_type}_results_plot.jpg")
    save_only = config.get('save_only', False)

    n = len(history)
    episode_idx = np.arange(1, n + 1)

    outcomes: Dict[str, np.ndarray] = {
        'Landed': np.array([1.0 if h.get('landed') else 0.0 for h in history]),
        'Timeout': np.array([1.0 if h.get('timeout') else 0.0 for h in history]),
    }

    crash_combos = sorted(
        {
            tuple(sorted(h.get('crash_reason', CrashReason(0)), key=lambda r: r.value))
            for h in history
            if h.get('crashed')
        },
        key=lambda combo: sum(reason.value for reason in combo),
        reverse=True,
    )

    for combo in crash_combos:
        if not combo:
            continue
        label = f"Crash: {'+'.join(reason.name for reason in combo)}"
        outcomes[label] = np.array([
            1.0 if h.get('crashed') and tuple(sorted(h.get('crash_reason', CrashReason(0)), key=lambda r: r.value)) == combo else 0.0
            for h in history
        ])

    plt.figure(figsize=(14, 8))
    for label, values in outcomes.items():
        running_avg = np.cumsum(values) / episode_idx
        plt.plot(episode_idx, running_avg, linewidth=2.0, label=label)

    plt.title(f"{agent_type} Outcome Progress - {environment}", fontsize=16)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)

    if save_only:
        plt.close()
        return

    plt.show()

def rand_argmax(b, **kw):
    """A random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == np.amax(b,**kw, keepdims=True)), **kw)

def save_as_gif(renderer, landing_result: EpisodeResult):
    """ Helper function to save the current episode as a GIF. """
    # Render extra frames to make the landing result is captured
    for _ in range(2 * FPS):
        renderer.clock.tick(FPS)
        renderer.render()
    status = "landed" if landing_result['landed'] else ("crashed" if landing_result['crashed'] else "timeout")
    filename = f"episode_{renderer.episode_count}_{status}"
    renderer.save_recording(filename)