import numpy as np
import matplotlib.pyplot as plt
import src.agent as agent
import src.environment as env
import time

def run_episode_and_collect_data(model):
    racing_env = agent.RacingEnv()
    state, _ = racing_env.reset() 
    episode_data = {
        'rewards': [],
        'speeds': [],
        'angles': [],
        'steps': []
    }
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _, _ = racing_env.step(action)
        episode_data['rewards'].append(reward)
        episode_data['speeds'].append(state[5])
        episode_data['angles'].append(state[6])  
        episode_data['steps'].append(step_count)
        
        total_reward += reward
        step_count += 1
    crashed = env.crashed
    finished = env.finished
    
    return episode_data, total_reward, step_count, crashed, finished


def plot_episode_performance(episode_data, episode_num, crashed, finished):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    outcome = "FINISHED" if finished else ("CRASHED" if crashed else "TIMEOUT")
    fig.suptitle(f'Episode {episode_num} Performance - {outcome}', fontsize=16)
    #Rewards over steps
    axes[0, 0].plot(episode_data['steps'], episode_data['rewards'], label='Reward per step', color='green')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Over Time')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    #Speed over steps
    axes[0, 1].plot(episode_data['steps'], episode_data['speeds'], label='Speed', color='blue')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Speed (normalized)')
    axes[0, 1].set_title('Speed Over Time')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    # Cumulative reward
    cumulative_reward = np.cumsum(episode_data['rewards'])
    axes[1, 0].plot(episode_data['steps'], cumulative_reward, label='Cumulative Reward', color='orange')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].set_title('Cumulative Reward Over Time')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    #  Agent angle
    axes[1, 1].plot(episode_data['steps'], episode_data['angles'], label='Angle', color='purple')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Angle (normalized)')
    axes[1, 1].set_title('Agent Direction Over Time')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/analysis/episode_performance.png', dpi=100)
    print("Saved: results/analysis/episode_performance.png")
    plt.show()


def plot_multiple_episodes(model, num_episodes=10):
    rewards_per_episode = []
    steps_per_episode = []
    crashed_episodes = []
    finished_episodes = []
    
    print(f"\nRunning {num_episodes} episodes for analysis:")
    for i in range(num_episodes):
        episode_data, total_reward, step_count, crashed, finished = run_episode_and_collect_data(model)
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)
        crashed_episodes.append(crashed)
        finished_episodes.append(finished)
        
        outcome = "FINISHED" if finished else ("CRASHED" if crashed else "TIMEOUT")
        print(f"Episode {i+1}: Reward={total_reward:.2f}, Steps={step_count}, Outcome={outcome}")
    
    # Calculate rates
    crash_rate = (sum(crashed_episodes) / num_episodes) * 100
    finish_rate = (sum(finished_episodes) / num_episodes) * 100
    timeout_rate = 100 - finish_rate - crash_rate
    success_rate = finish_rate
    
    print(f"\n Statistics:")
    print(f"Success Rate (Finished): {success_rate:.1f}%")
    print(f"Crash Rate: {crash_rate:.1f}%")
    print(f"Timeout Rate: {timeout_rate:.1f}%")
    print(f"Average Reward: {np.mean(rewards_per_episode):.2f}")
    print(f"Average Steps: {np.mean(steps_per_episode):.1f}")
    print(f"Max Reward: {np.max(rewards_per_episode):.2f}")
    print(f"Min Reward: {np.min(rewards_per_episode):.2f}")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    #Rewards per episode
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, num_episodes+1), rewards_per_episode, marker='o', color='green', linewidth=2, markersize=8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Reward per Episode')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(rewards_per_episode), color='r', linestyle='--', label='Average')
    ax1.legend()
    # Steps per episode
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, num_episodes+1), steps_per_episode, marker='o', color='blue', linewidth=2, markersize=8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.set_title('Steps per Episode')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(steps_per_episode), color='r', linestyle='--', label='Average')
    ax2.legend()
    #Episode outcomes (pie chart)
    ax3 = fig.add_subplot(gs[0, 2])
    outcomes = [sum(finished_episodes), sum(crashed_episodes), num_episodes - sum(finished_episodes) - sum(crashed_episodes)]
    labels = [f'Finished\n({finish_rate:.1f}%)', f'Crashed\n({crash_rate:.1f}%)', f'Timeout\n({timeout_rate:.1f}%)']
    colors = ['green', 'red', 'yellow']
    ax3.pie(outcomes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax3.set_title('Episode Outcomes')
    #Success metrics (bar chart)
    ax4 = fig.add_subplot(gs[1, 0])
    metrics = [success_rate, crash_rate, timeout_rate]
    metric_labels = ['Success\n(Finished)', 'Crash', 'Timeout']
    colors_bar = ['green', 'red', 'yellow']
    bars = ax4.bar(metric_labels, metrics, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Success Metrics')
    ax4.set_ylim([0, 100])
    ax4.grid(True, axis='y', alpha=0.3)
    
    for bar, v in zip(bars, metrics):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Reward distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(rewards_per_episode, bins=num_episodes//2, color='purple', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(rewards_per_episode), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards_per_episode):.1f}')
    ax5.set_xlabel('Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Reward Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    #Steps distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(steps_per_episode, bins=num_episodes//2, color='cyan', alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean(steps_per_episode), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps_per_episode):.0f}')
    ax6.set_xlabel('Steps')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Steps Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    #Reward trend
    ax7 = fig.add_subplot(gs[2, :])
    ax7.fill_between(range(1, num_episodes+1), rewards_per_episode, alpha=0.3, color='green')
    ax7.plot(range(1, num_episodes+1), rewards_per_episode, marker='o', color='green', linewidth=2, markersize=6, label='Reward')
    # Add moving average
    if num_episodes > 3:
        moving_avg = np.convolve(rewards_per_episode, np.ones(3)/3, mode='valid')
        ax7.plot(range(2, num_episodes), moving_avg, color='orange', linewidth=2, linestyle='--', label='3-Episode Moving Average')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Reward')
    ax7.set_title('Reward Trend Over Episodes')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.savefig('results/analysis/multiple_episodes.png', dpi=100, bbox_inches='tight')
    print("\nSaved: results/analysis/multiple_episodes.png")
    plt.show()
    
    return {
        'success_rate': success_rate,
        'crash_rate': crash_rate,
        'finish_rate': finish_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': np.mean(rewards_per_episode),
        'avg_steps': np.mean(steps_per_episode),
        'max_reward': np.max(rewards_per_episode),
        'min_reward': np.min(rewards_per_episode)
    }


def load_and_visualize(model_path="models/trained_racing_agent", episodes=10):
    from stable_baselines3 import PPO
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded.")
    print(f"\nRunning {episodes} episodes and collecting data...")
    stats = plot_multiple_episodes(model, num_episodes=episodes)
    print("\nRunning single detailed episode:")
    episode_data, total_reward, step_count, crashed, finished = run_episode_and_collect_data(model)
    outcome = "FINISHED" if finished else ("CRASHED" if crashed else "TIMEOUT")
    print(f"Episode complete. Reward: {total_reward:.2f}, Steps: {step_count}, Outcome: {outcome}")
    plot_episode_performance(episode_data, episode_num=1, crashed=crashed, finished=finished)
    return stats
if __name__ == "__main__":
    stats = load_and_visualize(model_path="models/trained_racing_agent", episodes=10)