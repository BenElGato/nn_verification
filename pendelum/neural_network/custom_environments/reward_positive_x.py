import numpy as np
import gym
import math
class CustomMountainCarRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_speed = 0
        self.last_x = 0
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        custom_reward = self.custom_reward_function(observation, reward, done)

        return observation, custom_reward, terminated, truncated, info
    def custom_reward_function(self, state, original_reward, done):
        x = state[0]
        y = state[1]
        speed = state[2]

        reward = original_reward
        reward += (20*y)**3
        return reward
