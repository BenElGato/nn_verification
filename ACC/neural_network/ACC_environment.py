import random

import gymnasium as gym
import matplotlib.pyplot
import numpy as np
from gymnasium.vector.utils import spaces


class CustomEnv(gym.Env):
    '''
    Observation space: [v_ego, T_gap, D_rel, D_safe]
    '''
    def __init__(self):
        '''
        Observation space initialization
        '''
        self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0
        self.x_ego = random.uniform(28, 38)
        self.v_ego = random.uniform(25, 35)
        self.a_ego = 0.0
        self.v_set = self.v_ego
        self.a_c_lead = random.uniform(-3.0, -1.0)
        self.dt = 0.1
        self.t = 0

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
        high = np.array([np.inf, np.inf, np.inf, np.inf,np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        low = np.array([-np.inf, -np.inf, - np.inf, -np.inf,-np.inf, -np.inf, - np.inf, -np.inf], dtype=np.float32)


        self.action_space = spaces.Box(
            low=-4.0, high=4.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self.t += self.dt

        self.x_lead += (self.v_lead)*self.dt
        self.x_ego += (self.v_ego)*self.dt

        self.v_lead += (self.a_lead)*self.dt
        self.v_ego += (self.a_ego)*self.dt

        self.a_lead += (-2*self.a_lead + 2*self.a_c_lead - 0.0001*(self.v_lead)**2)*self.dt
        self.a_ego += (-2*self.a_ego + 2*action[0] - 0.0001*(self.v_ego)**2)*self.dt

        self.D_rel = self.x_lead - self.x_ego
        self.D_safe = self.D_Default + self.T_Gap * self.v_ego

        reward = -(self.D_rel-self.D_safe - 20)**2

        terminated = bool(
            self.x_ego >= self.x_lead
        )
        if terminated:
            reward -= 10000
            #print(self.a_ego)
        return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.a_lead, self.a_ego,self.T_Gap, self.D_Default], dtype=np.float32), reward, terminated, False, {}
    def reset(self):
        super().reset()
        self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0
        self.x_ego = random.uniform(28, 38)
        self.v_ego = random.uniform(25, 35)
        self.a_ego = 0.0
        self.v_set = self.v_ego
        self.a_c_lead = random.uniform(-3.0, -1.0)
        self.dt = 0.1
        self.t = 0

        '''
        Parameters for calulating the rewards/costs
        '''
        self.D_Default = 10.0
        self.T_Gap = 1.4
        self.D_rel = self.x_lead - self.x_ego
        self.D_safe = self.D_Default + self.T_Gap * self.v_ego

        return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.a_lead, self.a_ego,self.T_Gap, self.D_Default], dtype=np.float32), {}

    def render(self, mode='human'):
        pass
