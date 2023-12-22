import gym
import tf2onnx
from gymnasium import register
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import numpy as np
import tensorflow as tf
import keras


# Define the TORA system dynamics
def tora_system(x, u):
    x1_dot = x[1]
    x2_dot = -x[0] + 0.1 * tf.sin(x[2])
    x3_dot = x[3]
    x4_dot = u

    return tf.stack([x1_dot, x2_dot, x3_dot, x4_dot])


# Custom Gym environment for TORA
class TORAEnv(gym.Env):
    def __init__(self):
        super(TORAEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        # Initialize the state
        self.state = np.random.rand(4)
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        # Apply the action to the TORA system
        self.state = tora_system(self.state, action)

        # Calculate the reward (you may need to define a custom reward function based on the task)
        reward = -np.sum(np.abs(self.state))

        # Check if the episode is done (you may need to define your termination condition)
        terminated = bool(self.state[0] == 0 and self.state[1] == 0 and self.state[2] == 0 and self.state[3] == 0)


        return self.state, reward, terminated, False, {}

