import numpy as np
import gymnasium as gym
import math
class CustomMountainCarRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        custom_reward = self.custom_reward_function(observation, reward, done)

        return observation, custom_reward, terminated, truncated, info
    def custom_reward_function(self, state, original_reward, done):
        pos_x = state[0]
        velocity = state[1]
        progress_bonus = max(0, pos_x - 0.45)
        penalty = 0
        if pos_x < 0 and velocity < 0:
            penalty = -0.1
        return original_reward + progress_bonus + penalty