import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class PPO:
	def __init__(self, policy_class, env, name, params, path, counter, obs_dim, act_dim):
		'''Hyperparameters ###############################'''
		self.timesteps_per_batch = params["timesteps_per_batch"]  # Number of timesteps to run per batch
		self.max_timesteps_per_episode = params["max_timesteps_per_episode"]  # Max number of timesteps per episode
		self.n_updates_per_iteration = params["n_updates_per_iteration"]  # Number of times to update actor/critic per iteration
		self.lr = params["lr"]  # Learning rate of actor optimizer
		self.gamma = params["gamma"]  # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = params["clip"]  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.name = name
		self.total_timesteps = params["total_timesteps"]
		self.entropy_coef = params["entropy_coef"]
		self.max_grad_norm = params["max_grad_norm"]
		self.dynamic_lr = params["dynamic_lr"]
		self.gradient_clipping = params["gradient_clipping"]
		self.seed = 50  # Sets the seed of our program, used for reproducibility of results
		'''##########################################################################'''
		self.env = env
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.path = path
		self.counter = counter

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim, params["neurons"])                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1, params["neurons"])

		#self.actor.load_state_dict(torch.load("/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/ppo_actor4.pth"))
		#self.critic.load_state_dict(torch.load("/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/ppo_critic4.pth"))

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		self.avg_ep_rews_history = []  # List to store average episodic returns per iteration

		self.delta_t = time.time_ns()
		self.t_so_far = 0
		self.i_so_far = 0
		self.batch_lens = [] # episodic lengths in batch
		self.batch_rews = [] # episodic returns in batch
		self.actor_losses = [] # losses of actor network in current iteration
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


	def learn(self):
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < self.total_timesteps:
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.collectData()
			t_so_far += np.sum(batch_lens)
			i_so_far += 1
			self.t_so_far = t_so_far
			self.i_so_far = i_so_far
			V, _, entropy= self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize advantages ---> This is an optimization!

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):
				new_lr = self.get_lr()
				self.actor_optim.param_groups[0]["lr"] = new_lr
				self.critic_optim.param_groups[0]["lr"] = new_lr
				V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				actor_loss = (-torch.min(surr1, surr2)).mean()
				actor_loss = actor_loss - self.entropy_coef * entropy.mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				if self.gradient_clipping:
					nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				if self.gradient_clipping:
					nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				self.critic_optim.step()

				# Log actor loss
				self.actor_losses.append(actor_loss.detach())


			self._log_summary()
			torch.save(self.actor.state_dict(), f'{self.path}/ppo_actor_{self.counter}.pth')
			torch.save(self.critic.state_dict(), f'{self.path}/ppo_critic_{self.counter}.pth')
		return self.avg_ep_rews_history

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
		self.batch_rews = batch_rews
		self.batch_lens = batch_lens

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
		return V, log_probs, dist.entropy()

	def _log_summary(self):
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.batch_rews])
		self.avg_ep_rews_history.append(avg_ep_rews)
		avg_ep_rews = str(round(avg_ep_rews, 2))
		print(f"-------------------- Iteration #{self.i_so_far} --------------------\nAverage Episodic Return: {avg_ep_rews}\nTimesteps So Far: {self.t_so_far}")
		self.batch_lens = []
		self.batch_rews = []
		self.logger = []

	def get_lr(self):
		'''
		Stabilizes the search for better convergence --> The learning rates decrease with time
		:return:
		'''
		if self.dynamic_lr:
			return max(self.lr * (1.0 - (self.t_so_far - 1.0) / self.total_timesteps), 0.0)
		else:
			return self.lr