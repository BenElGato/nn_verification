import random

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import spaces


class CustomEnv(gym.Env):
    '''
    Observation space: [x_lead, v_lead, a_lead, x_ego, v_ego, a_ego]
    '''
    def __init__(self):
        '''
        Observation space initialization
        '''
        self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0

        self.x_ego = random.uniform(10, 11)
        self.v_ego = random.uniform(30, 30.2)
        self.a_ego = 0.0

        '''
        Internal parameters
        '''
        self.a_c_lead = -2.0
        self.dt = 0.01


        '''
        Parameters for calulating the rewards/costs
        '''
        self.D_Default = 10.0
        self.T_Gap = 1.4
        self.D_rel = self.x_lead - self.x_ego
        self.D_safe = self.D_Default + self.T_Gap * self.v_ego
        '''
        Definition of action and observation space
        '''
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        low = np.array([-np.inf, -np.inf, - np.inf, -np.inf, -np.inf, - np.inf], dtype=np.float32)


        self.action_space = spaces.Box(
            low=-4.0, high=4.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.x_lead += (self.v_lead)*self.dt
        self.x_ego += (self.x_ego)*self.dt

        self.v_lead += (self.a_lead)*self.dt
        self.v_ego += (self.a_ego)*self.dt

        self.a_lead += (-2*self.a_lead + 2*self.a_c_lead - 0.0001*(self.v_lead)**2)*self.dt
        self.a_ego += (-2*self.a_ego + 2*action[0] - 0.0001*(self.v_ego)**2)*self.dt

        self.D_rel = self.x_lead - self.x_ego
        self.D_safe = self.D_Default + self.T_Gap * self.v_ego

        reward = (self.D_rel - self.D_safe)
        return np.array([self.x_lead, self.v_lead, self.a_lead, self.x_ego, self.v_ego, self.a_ego], dtype=np.float32), reward, False, False, {}
    def reset(self):
        super().reset()
        self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0

        self.x_ego = random.uniform(10, 11)
        self.v_ego = random.uniform(30, 30.2)
        self.a_ego = 0.0

        self.a_c_lead = -2.0

        self.D_Default = 10.0
        self.T_Gap = 1.4
        self.D_rel = self.x_lead - self.x_ego
        self.D_safe = self.D_Default + self.T_Gap * self.v_ego

        return np.array([self.x_lead, self.v_lead, self.a_lead, self.x_ego, self.v_ego, self.a_ego], dtype=np.float32), {}

    def render(self, mode='human'):
        pass
