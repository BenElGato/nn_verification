import math

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_output = nn.Linear(hidden_size, action_dim)
        self.log_std_output = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_output(x)
        log_std = self.log_std_output(x)
        return mean, log_std


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.output(x)
        return value


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).view(-1, 1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).view(-1, 1)
        )


# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_size=128, lr_actor=0.0005, lr_critic=0.0005, lr_alpha=0.0003,
                 gamma=0.99999, tau=0.1, alpha_init=0.2, alpha_target=0.2, buffer_capacity=50000, batch_size=256):
        self.alpha = alpha_init
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.critic_1 = Critic(state_dim, action_dim, hidden_size)
        self.critic_2 = Critic(state_dim, action_dim, hidden_size)
        self.target_critic_1 = Critic(state_dim, action_dim, hidden_size)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_size)
        self.log_alpha = torch.nn.Parameter(torch.log(torch.FloatTensor([alpha_init])))
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

        # Initialize target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=lr_critic)
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=lr_alpha)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state)
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z).detach().numpy()
        return action if deterministic else np.clip(action, -1.0, 1.0)

    def update_networks(self, target_entropy):
        # Sample a mini-batch from the replay buffer
        batch = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch

        # Compute targets for Q functions
        with torch.no_grad():
            next_action, next_log_prob = self.sample_action_and_log_prob(next_state)
            target_q_value_1 = self.target_critic_1(next_state, next_action)
            target_q_value_2 = self.target_critic_2(next_state, next_action)
            target_q_value = torch.min(target_q_value_1, target_q_value_2) - self.alpha_target * next_log_prob
            target_q_value = reward + self.gamma * (1.0 - done) * target_q_value

        # Update Critic networks
        q_value_1 = self.critic_1(state, action)
        q_value_2 = self.critic_2(state, action)
        critic_loss_1 = nn.MSELoss()(q_value_1, target_q_value.detach())
        critic_loss_2 = nn.MSELoss()(q_value_2, target_q_value.detach())
        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()
        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()

        # Update Actor network
        sampled_action, log_prob = self.sample_action_and_log_prob(state)
        min_q_value = torch.min(self.critic_1(state, sampled_action), self.critic_2(state, sampled_action))
        actor_loss = (self.alpha * log_prob - min_q_value).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()
        self.alpha = self.log_alpha.exp().item()

        # Update target networks with Polyak averaging
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)

    def sample_action_and_log_prob(self, state):
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)


# Training loop
def train_sac_agent(agent, env, max_episodes=10000, max_steps_per_episode=50000):
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()

    for episode in range(1, max_episodes + 1):
        state,_ = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, terminated,truncated, _ = env.step(action)
            agent.buffer.add((state, action, reward, next_state, terminated or truncated))
            episode_reward += reward

            if len(agent.buffer.buffer) >= agent.batch_size:
                agent.update_networks(target_entropy)

            state = next_state

            if terminated or truncated:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")

    env.close()


# Create the SAC agent and train it
env = gym.make('Pendulum-v1')


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim)
train_sac_agent(agent, env)
