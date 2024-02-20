from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
DEFAULT_X = np.pi
DEFAULT_Y = 1.0
class PendulumEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.01
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        rewards = -costs
        if abs(angle_normalize(th)) < 0.1 and abs(thdot < 0.01):
            rewards += 20
        self.state = np.array([newth, newthdot])
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), rewards, False, False, {}

    def reset(self, *, seed: Optional[int] = None,  options: Optional[dict] = None):
        super().reset(seed=seed)
        th = np.random.uniform(-np.pi, np.pi)
        thdot = np.random.uniform(-self.max_speed, self.max_speed)
        self.state = np.array([th, thdot])
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([angle_normalize(theta), thetadot], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi