

import gymnasium as gym
import sys

import numpy as np
import torch
from ppo import PPO
from network import FeedForwardNN
from network import tanhNN
from mountaincar.neural_network.custom_environments import custum_wrapper_energy_function as wrapper
import matplotlib.pyplot as plt
import optuna
from network import sigmoidNN
from network import SimpleFeedForwardNN
from network import SimpleSigmoidNN
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from network import FeedForwardInitialisationTanhNN
from pendelum.neural_network.custom_environments.spawn import PendulumEnvWrapper

def objective(trial, env, params, name):
	config = {
		"lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
		"gamma": trial.suggest_float("gamma", 0.9, 0.9999),
		"clip": trial.suggest_float("clip", 0.1, 0.3),
		"entropy_coef": trial.suggest_float("entropy_coef", 1e-4, 1e-1)
		# include other parameters as needed
	}
	params["lr"] = config["lr"]
	params["gamma"] = config["gamma"]
	params["clip"] = config["clip"]
	params["entropy_coef"] = config["entropy_coef"]

	model = PPO(policy_class=FeedForwardNN, env=env, name=name, params=params)
	model.learn()
	return -np.mean(model.avg_ep_rews_history)


def train(env, actor_model, critic_model,name,params, path, counter):
	model = PPO(policy_class=params["nn"], env=env, name=name, params=params, path=path, counter=counter)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)
	return model.learn()

def test(env, actor_model, neurons, nn):
	print(f"Testing {actor_model}", flush=True)
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
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
def export_onnx(env, actor_model, path, neurons, name, nn):
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	policy = nn(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))

	dummy_input = torch.randn(1, obs_dim)
	onnx_filename = f'{path}/{name}.onnx'
	torch.onnx.export(policy, dummy_input, onnx_filename, verbose=True, opset_version=10)
def compare_settings(env, name, settings, path):
	counter = 9
	for i in settings:
		model = PPO(policy_class=i["nn"], env=env, name=name, params=i, path=path, counter=counter)
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
"------------------------------------Bayesian optimization--------------------------------------------"
# Define the hyperparameter space
space = [
    Real(1e-5, 1e-3, name='lr', prior='log-uniform'),
    Real(0.9, 0.9999, name='gamma'),
    Real(0.1, 0.3, name='clip'),
	Real(0.1, 0.5, name='max_grad_norm'),
]
# Define the objective function
@use_named_args(space)
def objective(lr, gamma, clip,max_grad_norm):
    params = {
        "lr": lr,
        "gamma": gamma,
        "clip": clip,
        "entropy_coef": 0.0,
		"neurons": 9,
		"timesteps_per_batch": 2048,
		"max_timesteps_per_episode": 200,
		"n_updates_per_iteration": 18,
		"dynamic_lr": True,
		"gradient_clipping": True,
		"max_grad_norm": max_grad_norm,
		"total_timesteps": 1_000_000,
		"nn": SimpleSigmoidNN
    }

    env = gym.make('Pendulum-v1')
    model = PPO(policy_class=SimpleSigmoidNN, env=env, params=params, counter=1,path="/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/network_code/PPO/training_docs/bayesian_optimization", name="bay")
    model.learn()
    env.close()

    # Negative mean reward for minimization
    return -np.mean(model.avg_ep_rews_history)

def optimize_hyperparameters():
    # Run Bayesian optimization
    result = gp_minimize(objective, space, n_calls=100, random_state=0)

    # Best hyperparameters
    return result.x
"-----------------------------------------------------------------------------------------------------"

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
	params = {'neurons': 9, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 500000, 'nn': SimpleSigmoidNN}

	#best_hyperparams = optimize_hyperparameters()
	#print("Best hyperparameters (optimized): ", best_hyperparams)

	name = 'Pendulum-v1'
	env = gym.make(name)
	env = PendulumEnvWrapper(env)
	actor_model = ""
	critic_model = ""

	settings = [params]
	compare_settings(env,name, settings,"/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/network_code/PPO/training_docs/networks")
	#train(env=env, params=params, actor_model=actor_model, critic_model=critic_model, name=name, path="/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/network_code/PPO/training_docs/networks", counter=9)


	actor_model = f"/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/network_code/PPO/training_docs/networks/ppo_actor9.pth"

	env = gym.make(name, render_mode="human")
	env = PendulumEnvWrapper(env)

	#test(env=env, actor_model=actor_model, neurons=params["neurons"], nn=params["nn"])
	export_onnx(env,actor_model=actor_model, path="/home/benedikt/PycharmProjects/nn_verification/pendelum/cora", neurons=params["neurons"], name="network", nn=params["nn"])

	#study = optuna.create_study(direction="minimize")
	#study.optimize(lambda trial: objective(trial, env=env, params=params,name=name), n_trials=100)

	# Get the best hyperparameters
	#best_hyperparams = study.best_params
	#print("Best hyperparameters: ", best_hyperparams)
