from typing import Optional

import gymnasium as gym
import numpy as np


class PendulumEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.starting_states = starting_states = {
		"x_init": -0.5,
		"y_init": -0.5
	}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        normal_reset, info = self.env.reset()
        return np.array([-0.5, -0.5, 0.0]), info

