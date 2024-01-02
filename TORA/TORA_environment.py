import math
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import spaces


class TORAEnv(gym.Env):

    def __init__(self):
        '''
        Observation space initialization
        '''

        self.min_action = -1.0
        self.max_action = 1.0

        self.min_x1 = -1.0
        self.max_x1 = 1.0

        self.min_x2 = -1.0
        self.max_x2 = 1.0

        self.min_x3 = -1.0
        self.max_x3 = 1.0

        self.min_x4 = -1.0
        self.max_x4 = 1.0

        self.dt = 0.05

        self.low_state = np.array(
            [self.min_x1, self.min_x2, self.min_x3, self.min_x4], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_x1, self.max_x2, self.max_x3, self.max_x4], dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32)

    def step(self, action: np.ndarray):

        x1 = self.state[0]
        x2 = self.state[1]
        x3 = self.state[2]
        x4 = self.state[3]
        u = 11 * action[0]
        dt = self.dt


        x1 += x2 * dt
        '''if x1 > self.max_x1:
            x1 = self.max_x1
        if x1 < self.min_x1:
            x1 = self.min_x1'''

        x2 += -x1 + 0.1 * np.sin(x3) * dt
        '''if x2 > self.max_x2:
            x2 = self.max_x2
        if x2 < self.min_x2:
            x2 = self.min_x2'''

        x3 += x4 * dt
        '''if x3 > self.max_x3:
            x3 = self.max_x3
        if x3 < self.min_x3:
            x3 = self.min_x3'''

        x4 += u * dt
        '''if x4 > self.max_x4:
            x4 = self.max_x4
        if x4 < self.min_x4:
            x4 = self.min_x4'''

        terminated = bool(x1 == 0 and x2 == 0 and x3 == 0 and x4 == 0)
        target_state = np.zeros(4)
        reward = 0
        if terminated:
            reward += 100.0

        distance = -np.linalg.norm(self.state - target_state)
        reward = distance

        self.state = np.array([x1, x2, x3, x4], dtype=np.float32)
        return self.state, reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.

        self.state = np.array([self.np_random.uniform(low=-1.0, high=1.0), self.np_random.uniform(low=-1.0, high=1.0),
                               self.np_random.uniform(low=-1.0, high=1.0), self.np_random.uniform(low=-1.0, high=1.0)])

        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode='human'):
        pass
