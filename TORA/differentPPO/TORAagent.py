from typing import List, Callable

import gym
import numpy as np
from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# Define the TORA system dynamics
def tora_system(x, u):
    x1_dot = x[1]
    x2_dot = -x[0] + 0.1 * np.sin(x[2])
    x3_dot = x[3]
    x4_dot = u

    return np.stack([x1_dot, x2_dot, x3_dot, x4_dot])


# Custom Gym environment for TORA
class TORAEnv(gym.Env):
    def __init__(self):
        super(TORAEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        # Initialize the state
        self.state = np.random.rand(4)
        return self.state

    def step(self, action):
        # Apply the action to the TORA system
        self.state = tora_system(self.state, action)

        # Calculate the reward (you may need to define a custom reward function based on the task)
        reward = -np.sum(np.abs(self.state))

        # Check if the episode is done (you may need to define your termination condition)
        done = False

        return self.state, reward, done, {}


# Wrap the environment in a DummyVecEnv
env_fn = lambda: TORAEnv()
env_list: List[Callable[[], gym.Env]] = [env_fn] * 1  # Replace 1 with the number of environments you want
env = DummyVecEnv(env_list)

# PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Training the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_tora_model")
