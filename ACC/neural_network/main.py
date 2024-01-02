import sys

from gymnasium.envs.registration import register
from matplotlib import pyplot as plt

from ACC.neural_network.network import tanhNN, RELU_NN, HeavyBrakes_RELU_NN
#from ACC.neural_network.ppo import PPO
from ACC_environment import CustomEnv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

import gymnasium as gym
def test(env, actor_model, neurons, nn, obs_dim, act_dim):
	print(f"Testing {actor_model}", flush=True)
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)
	policy = nn(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))
	observation, info = env.reset()
	for i in range(10000):
		action = policy(observation).detach().numpy() # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()
			print(observation)

	env.close()
def export_onnx(env, actor_model, path, neurons, name, nn, obs_dim, act_dim):
	policy = nn(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))

	dummy_input = torch.randn(1, obs_dim)
	onnx_filename = f'{path}/{name}.onnx'
	torch.onnx.export(policy, dummy_input, onnx_filename, verbose=True, opset_version=10)
def compare_settings(env, name, settings, path, obs_dim, act_dim):
	counter = 100
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
		counter+=1

register(
        id='CustomEnv',
        entry_point='ACC_environment:CustomEnv',
    )

'''
Hyperparamters ###########################
'''
timesteps_per_batch =  2048
max_timesteps_per_episode = 200
gamma = 0.99
n_updates_per_iteration = 10
lr = 3e-3
clip = 0.2
entropy_coef = 0.0
max_grad_norm = 0.9
'''
####################################
'''

# TODO make episode lenght longer
params = {'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7_000_000, 'neural_network': tanhNN}


env = gym.make("CustomEnv")
actor_model = ""
critic_model = ""
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

settings = [params]
name = "CustomEnv"

compare_settings(env,name, settings,"/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network", obs_dim=obs_dim,act_dim=act_dim)

actor_model = f"/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/ppo_actor11.pth"

#test(env=env, actor_model=actor_model, neurons=params["neurons"], neural_network=params["neural_network"], obs_dim=obs_dim, act_dim=act_dim)
#export_onnx(env,actor_model=actor_model, path="/home/benedikt/PycharmProjects/nn_verification/ACC/cora", neurons=params["neurons"], name="network11", nn=params["neural_network"], obs_dim=obs_dim,act_dim=act_dim)



