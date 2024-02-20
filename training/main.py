import argparse
import os
import sys

from gymnasium.envs.registration import register
import gymnasium as gym
from ppo import PPO
import torch
from matplotlib import pyplot as plt
from network import tanhNN, RELUNN, tinytanhNN, hugetanhNN
from gym import envs


class YourClass:
    def __init__(self):
        pass

    def export_onnx(self, actor_model, neurons, name, nn, obs_dim, act_dim):
        policy = nn(obs_dim, act_dim, neurons=neurons)
        policy.load_state_dict(torch.load(actor_model))

        dummy_input = torch.randn(1, obs_dim)
        onnx_filename = f'{actor_model}/network{name}.onnx'
        torch.onnx.export(policy, dummy_input, onnx_filename, verbose=True, opset_version=10)

    def train_network(self, params_file, environment, nn_name):
        # Load params from the text file
        params = []
        with open(params_file, 'r') as file:
            for line in file:
                params.append(eval(line.strip()))  # Safely evaluate each line as a dictionary

        # Set up environment
        if environment.lower() == 'acc':
            env = gym.make("ACCEnv")
            name = "ACCEnv"
        elif environment.lower() == 'pendulum':
            env = gym.make("PendulumEnv")
            name = "PendulumEnv"
        else:
            raise ValueError("Invalid environment name. Must be 'ACC' or 'pendulum'.")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Compare settings
        current_directory = './'  # Current directory
        self.compare_settings(env, name, params, current_directory, obs_dim, act_dim, nn_name)

        # Export ONNX
        actor_model = current_directory  # Path to the current directory
        neurons = params[0].get('neurons')  # Neurons from the first set of parameters
        self.export_onnx(actor_model, neurons, nn_name, params[0]["neural_network"], obs_dim, act_dim)

    def compare_settings(self, env, name, settings, path, obs_dim, act_dim, nn_name):
        counter = 1
        for i in settings:
            model = PPO(policy_class=i["neural_network"], env=env, name=name, params=i, path=path, counter=counter, obs_dim=obs_dim, act_dim=act_dim)
            average_rewards = model.learn()
            print("")

            plt.plot(average_rewards, label=f"{counter}. settings")
            plt.legend()
            plt.title('Average Episodic Returns Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Average Return')
            plt.savefig(f'{path}/graph{counter}.png')
            plt.close()
            with open(f'{path}/report.md', 'a') as file:
                file.write(f"## Results for Setting {counter}\n")
                file.write(f"![Average Rewards Plot]({path}/graph{counter}.png)\n\n")
                file.write(f"{str(i)}\n\n")
            counter += 1
'''
Registration of the custom environments........................
'''

def main():
    sys.path.append("..")
    register(
        id='ACCEnv',
        entry_point='ACC.neural_network.ACC_environment:CustomEnv',
    )
    register(
        id='PendulumEnv',
        entry_point='pendulum.neural_network.custom_environments.Custom_Pendulum:PendulumEnv',
    )
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description="Train and export neural network.")
    parser.add_argument("params_file", help="Path to the txt file containing training parameters.")
    parser.add_argument("environment", choices=["ACC", "pendulum"], help="Environment for training.")
    parser.add_argument("nn_name", help="Name of the neural network.")
    args = parser.parse_args()

    your_instance = YourClass()
    your_instance.train_network(args.params_file, args.environment, args.nn_name)

if __name__ == "__main__":
    main()