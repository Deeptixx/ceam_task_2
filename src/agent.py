"""This is a wrapper class because environment.py has functions like step reset and render but
state-baselines3 expects a class with gym.Env structure
so this file wraps the functions in environment.py into a class that inherits from gym.Env"""
import numpy as np
from stable_baselines3 import PPO #proximal policy optimization
import gymnasium as gym
from gymnasium import spaces
from src import environment as env

class RacingEnv(gym.Env):
    def __init__(self):
        self.observation_space=spaces.Box(low=0,high=1,shape=(7,),dtype=np.float32)
        self.action_space=spaces.Discrete(4)
    def reset(self,seed=None):
        state=env.reset()
        return state,{}
    def step(self,action):
        state, reward, done=env.step(action)
        return state, reward, done, False, {}
    def render(self):
        env.render()




