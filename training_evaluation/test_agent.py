import src.agent as agent
import src.environment as env
def load_model(path="models/trained_racing_agent"):
    from stable_baselines3 import PPO
    model = PPO.load(path)
    print(f"Model loaded from {path}")
    return model
def test_agent(model, episodes=5):
    """Test trained agent with visual feedback"""
    import time
    for episode in range(episodes):
        racing_env = agent.RacingEnv()
        state, _ = racing_env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\nEpisode {episode+1}:")
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _, _ = racing_env.step(action)
            racing_env.render()
            # Slow down to 20 FPS (add 50ms delay)
            time.sleep(0.05)
            total_reward += reward
            steps += 1
            # Print progress every 100 steps
            #if steps % 100 == 0:
                #print(f"  Step {steps}, Reward so far: {total_reward:.2f}")
        crashed = env.crashed
        finished = env.finished
        outcome = "FINISHED" if finished else ("CRASHED" if crashed else "TIMEOUT")
        
        print(f"Episode {episode+1} Complete: Reward={total_reward:.2f}, Steps={steps}, Outcome={outcome}")

if __name__ == "__main__":
    print("Loading trained agent:")
    model = load_model()
    print("\nTesting agent for 3 episodes:")
    test_agent(model, episodes=3)