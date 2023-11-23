"""
	This file is the executable for running PPO. It is based on this medium article:
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch
from ppo import PPO
from network import FeedForwardNN

def train(env, timesteps_per_batch, max_timesteps_per_episode, gamma,n_updates_per_iteration, lr, clip, render, render_every_i, actor_model, critic_model, timesteps=200_000_000):
	model = PPO(policy_class=FeedForwardNN, env=env, timesteps_per_batch=timesteps_per_batch, max_timesteps_per_episode=max_timesteps_per_episode,
	gamma=gamma,n_updates_per_iteration=n_updates_per_iteration,lr=lr,clip=clip,render=render,render_every_i=render_every_i)

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
	model.learn(timesteps)

def test(env, actor_model):
	print(f"Testing {actor_model}", flush=True)
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	policy = FeedForwardNN(obs_dim, act_dim)
	policy.load_state_dict(torch.load(actor_model))
	observation, info = env.reset()
	for i in range(10000):
		action = policy(observation).detach().numpy() # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()

	env.close()


if __name__ == '__main__':
	'''
	Hyperparamters ###########################
	'''
	timesteps_per_batch =  2048
	max_timesteps_per_episode = 200
	gamma = 0.99
	n_updates_per_iteration = 10
	lr = 3e-4
	clip = 0.2
	render = True
	render_every_i = 10
	'''
	####################################
	'''
	env = gym.make('Pendulum-v1')
	actor_model = ""
	critic_model = ""
	'''
	train(env=env, timesteps_per_batch=timesteps_per_batch, max_timesteps_per_episode=max_timesteps_per_episode,
	gamma=gamma,n_updates_per_iteration=n_updates_per_iteration,lr=lr,clip=clip,render=render,render_every_i=render_every_i
		  , actor_model=actor_model, critic_model=critic_model, timesteps=10_000_000)
	'''
	actor_model = "ppo_actor.pth"
	env = gym.make('Pendulum-v1', render_mode="human")
	test(env=env, actor_model=actor_model)