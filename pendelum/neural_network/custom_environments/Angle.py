import numpy as np
import gymnasium as gym
import math
class AngleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs = self.getAngle(observation)
        if obs[0] % math.pi < 0.1 and abs(obs[1] < 0.01):
            reward += 20
        return self.getAngle(observation), reward, terminated, truncated, info
    def getAngle(self, state):
       return np.array([math.atan2(state[1], state[0]),state[2]], dtype=np.float32)
    def reset(self):
        obs, info = self.env.reset()
        return self.getAngle(obs), info