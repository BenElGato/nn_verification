from typing import Optional

import gymnasium as gym
import numpy as np


class PendulumEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        custom_reward = self.custom_reward_function(observation, reward, done)

        return observation, custom_reward, terminated, truncated, info

    def custom_reward_function(self, state, original_reward, done):

        return original_reward