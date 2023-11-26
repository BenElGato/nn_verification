"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from network import FeedForwardNN

class PPO:
	def __init__(self, policy_class, env, timesteps_per_batch, max_timesteps_per_episode, gamma,n_updates_per_iteration, lr, clip, name):
		'''Hyperparameters ###############################'''
		self.timesteps_per_batch = timesteps_per_batch  # Number of timesteps to run per batch
		self.max_timesteps_per_episode = max_timesteps_per_episode  # Max number of timesteps per episode
		self.n_updates_per_iteration = n_updates_per_iteration  # Number of times to update actor/critic per iteration
		self.lr = lr  # Learning rate of actor optimizer
		self.gamma = gamma  # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = clip  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.name = name

		sin(atan2(x(1), x(2)) + clip(x(3) + (3 * g / (2 * l) * sin(atan2(x(1), x(2)) + 3.0 / (m * l ^ 2) * clip(u(1), min_speed, max_speed)) * dt), min_speed, max_speed)) - x(2);

		self.save_freq = 10  # How often we save in number of iterations
		self.seed = 50  # Sets the seed of our program, used for reproducibility of results
		'''##########################################################################'''
		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.collectData()
			t_so_far += np.sum(batch_lens)
			i_so_far += 1
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize advantages ---> This is an optimization!

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), f'./{self.name}ppo_actor.pth')
				torch.save(self.critic.state_dict(), f'./{self.name}ppo_critic.pth')

	def collectData(self):
		"""
			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch
		while t < self.timesteps_per_batch:
			ep_rews = []
			obs, info = self.env.reset()
			done = False
			for ep_t in range(self.max_timesteps_per_episode):
				t += 1
				batch_obs.append(obs)
				action, log_prob = self.get_action(obs)
				obs, rew, terminated, truncated, info = self.env.step(action)
				done = terminated or truncated
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)
		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
		return batch_rtgs

	def get_action(self, obs):
		mean = self.actor(obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		V = self.critic(batch_obs).squeeze()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)
		return V, log_probs

	def _log_summary(self):
		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------\n")
		print(f"Average Episodic Return: {avg_ep_rews}\n")
		print(f"Average actor Loss: {avg_actor_loss}\n")
		print(f"Timesteps So Far: {t_so_far}\n")
		print(f"------------------------------------------------------\n")

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
