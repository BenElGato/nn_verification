

import gymnasium as gym
import sys

import numpy as np
import torch
from ppo import PPO
from network import FeedForwardNN
from mountaincar.neural_network.custom_environments import custum_wrapper_energy_function as wrapper
import matplotlib.pyplot as plt
import optuna
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os


def objective(trial,env, timesteps_per_batch, max_timesteps_per_episode,n_updates_per_iteration,name, timesteps):
    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
		"clip": trial.suggest_float("trial", 0.01, 0.3),
		"entropy_coef": trial.suggest_float("entropy_coef", 1e-4, 1e-1)
        # include other parameters as needed
    }

    # Create a PPO instance with the current set of hyperparameters
    model = PPO(policy_class=FeedForwardNN, env=env, gamma=config['gamma'], lr=config['lr'], clip=config['clip'], timesteps_per_batch=timesteps_per_batch, max_timesteps_per_episode=max_timesteps_per_episode,n_updates_per_iteration=n_updates_per_iteration,entropy_coef=config["entropy_coef"],name=name,total_timesteps=timesteps)

    # Run training
    model.learn()

    # Return the negative average reward
    return -np.mean(model.avg_ep_rews_history)




def train(env, actor_model, critic_model,name,params):
	model = PPO(policy_class=FeedForwardNN, env=env, name=name, params=params)

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

def test(env, actor_model, neurons):
	print(f"Testing {actor_model}", flush=True)
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	policy = FeedForwardNN(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))
	observation, info = env.reset()
	for i in range(10000):
		action = policy(observation).detach().numpy() # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()

	env.close()
def export_onnx(env, actor_model, batchsize, path, neurons):
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	policy = FeedForwardNN(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))

	#dummy_input = torch.randn(batchsize, obs_dim)
	dummy_input = torch.randn(1, obs_dim)
	print(dummy_input.size())
	onnx_filename = f'{path}/actor_model_pendulum.onnx'
	torch.onnx.export(policy, dummy_input, onnx_filename, verbose=True, opset_version=10)
def compare_settings(env, name, settings, pdfName):
	c = canvas.Canvas(f"{pdfName}.pdf", pagesize=letter)
	width, height = letter

	for i in settings:
		model = PPO(policy_class=FeedForwardNN, env=env, name=name, params=params)
		average_rewards = train(env,'', '',name, params)
		plt.plot(average_rewards, label=f"{i}. settings")
		plt.legend()
		plt.title('Average Episodic Returns Over Iterations')
		plt.xlabel('Iteration')
		plt.ylabel('Average Return')
		plt.savefig(f'{pdfName}.png')
		plt.close()

		c.drawImage(f'{pdfName}.png', 50, height - 250)
		text_object = c.beginText(50, height - 300)
		text_object.textLines(i)
		c.drawText(text_object)
		os.remove(f'{pdfName}.png')
		height -= 300
	c.save()
	print(f"PDF saved as {pdfName}")

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
	'''
	####################################
	'''

	params = {
		"neurons": 64,
		"timesteps_per_batch": 2048,
		"max_timesteps_per_episode": 200,
		"gamma": 0.99,
		"n_updates_per_iteration": 10,
		"dynamic_lr": True,
		"lr":3e-3,
		"clip": 0.2,
		"entropy_coef": 0.0,
		"gradient_clipping": False,
		"max_grad_norm": 0.5,
		"total_timesteps": 100
	}
	params2 = {
		"neurons": 32,
		"timesteps_per_batch": 2048,
		"max_timesteps_per_episode": 200,
		"gamma": 0.99,
		"n_updates_per_iteration": 10,
		"dynamic_lr": True,
		"lr": 3e-3,
		"clip": 0.2,
		"entropy_coef": 0.0,
		"gradient_clipping": False,
		"max_grad_norm": 0.5,
		"total_timesteps": 100
	}
	name = 'Pendulum-v1'
	env = gym.make(name)
	actor_model = ""
	critic_model = ""

	settings = [params, params2]
	compare_settings(env,name, params,"/home/elgato/nn_verification/pendelum/neural_network/network_code/PPO/training_graphs/neuronsize")
	#train(env=env, params=params, actor_model=actor_model, critic_model=critic_model, timesteps=1_000_000, name=name)


	#actor_model = f"{name}ppo_actor.pth"
	#env = gym.make(name, render_mode="human")
	#test(env=env, actor_model=actor_model)
	#export_onnx(env,actor_model=actor_model,batchsize=timesteps_per_batch, path="/home/elgato/nn_verification/pendelum/cora")

	#study = optuna.create_study(direction="minimize")
	#study.optimize(lambda trial: objective(trial, env=env,timesteps_per_batch=timesteps_per_batch, max_timesteps_per_episode=max_timesteps_per_episode,n_updates_per_iteration=n_updates_per_iteration,name=name,timesteps=1_000_000), n_trials=100)

	# Get the best hyperparameters
	#best_hyperparams = study.best_params
	#print("Best hyperparameters: ", best_hyperparams)
