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
        pos_x = state[0]
        speed = abs(state[1])
        height = np.sin(3 * pos_x) * 0.45 + 0.55
        reward = 100 * ((math.sin(3 * pos_x) * 0.0025 + 0.5 * speed * speed) - (
                    math.sin(3 * self.last_x) * 0.0025 + 0.5 * self.last_speed * self.last_speed))
        if pos_x >= 0.35:
            reward += 10
        if pos_x >= 0.45:
            reward += 100000
        self.last_speed = speed
        self.last_x = pos_x
        return reward
