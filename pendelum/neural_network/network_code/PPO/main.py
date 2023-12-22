

import gymnasium as gym
import sys

import numpy as np
import torch
from ppo import PPO
from network import FeedForwardNN

import matplotlib.pyplot as plt
import optuna
from network import sigmoidNN, tanhNN
from network import SimpleFeedForwardNN
from network import SimpleSigmoidNN
from skopt import gp_minimize
from pendelum.neural_network.custom_environments.Angle import AngleWrapper


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
	counter = 9
	for i in settings:
		model = PPO(policy_class=i["nn"], env=env, name=name, params=i, path=path, counter=counter, obs_dim=obs_dim, act_dim=act_dim)
		average_rewards = model.learn()
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




if __name__ == '__main__':
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

	# TODO adjust reset function so that it always starts between y=-0.48 and y=0.48
	params = {'neurons': 30, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 1000000, 'nn': tanhNN}

	#best_hyperparams = optimize_hyperparameters()
	#print("Best hyperparameters (optimized): ", best_hyperparams)

	name = 'Pendulum-v1'
	env = gym.make(name)
	env = AngleWrapper(env)
	actor_model = ""
	critic_model = ""
	obs_dim = env.observation_space.shape[0]
	obs_dim = 2
	act_dim = env.action_space.shape[0]

	settings = [params]
	#compare_settings(env,name, settings,"/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/network_code/PPO/training_docs/networks", obs_dim=obs_dim,act_dim=act_dim)

	actor_model = f"/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/network_code/PPO/training_docs/entropy/angularNetwork.pth"

	env = gym.make(name, render_mode="human")
	env = AngleWrapper(env)

	#test(env=env, actor_model=actor_model, neurons=params["neurons"], nn=params["nn"], obs_dim=obs_dim, act_dim=act_dim)
	export_onnx(env,actor_model=actor_model, path="/home/benedikt/PycharmProjects/nn_verification/pendelum/cora", neurons=params["neurons"], name="network", nn=params["nn"], obs_dim=obs_dim,act_dim=act_dim)

	#study = optuna.create_study(direction="minimize")
	#study.optimize(lambda trial: objective(trial, env=env, params=params,name=name), n_trials=100)

	# Get the best hyperparameters
	#best_hyperparams = study.best_params
	#print("Best hyperparameters: ", best_hyperparams)
