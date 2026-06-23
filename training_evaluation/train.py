import numpy as np
from stable_baselines3 import PPO
import src.agent as agent
learning_rate=0.0003
n_steps=2048
batch_size=64
epochs=10
gamma=0.99
total_timestamps=10000
max_steps_per_episode=1000

def train_agent():
    racing_env=agent.RacingEnv()
    model=PPO("MlpPolicy",
        racing_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=epochs,
        gamma=gamma,
        verbose=1
    )
    print(f"Training agent for {total_timestamps} timesteps:")
    model.learn(total_timesteps=total_timestamps)
    print("Training complete.")
    return model

    