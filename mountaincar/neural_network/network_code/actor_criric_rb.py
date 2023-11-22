import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc_action = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        return torch.tanh(self.fc_action(x))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.fc_action = nn.Linear(action_dim, hidden_dim)
        self.fc_combined = nn.Linear(hidden_dim * 2, 1)

    def forward(self, state, action):
        x_state = torch.relu(self.fc_state(state))
        x_action = torch.relu(self.fc_action(action))
        x_combined = torch.cat([x_state, x_action], dim=1)
        return self.fc_combined(x_combined).squeeze()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).view(-1, 1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).view(-1, 1)
        )

class ActorCriticAgentContinuousWithReplay:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr_actor=0.001, lr_critic=0.001, buffer_capacity=10000, batch_size=128):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state, exploration_noise=0.1):
        state = torch.FloatTensor(state)
        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, exploration_noise * torch.ones_like(mean_action))
        return torch.clamp(sampled_action, -1.0, 1.0).item()

    def update_critic(self, state, action, reward, next_state, done, gamma=0.99):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor([action])
        next_state = torch.FloatTensor(next_state)

        next_value = self.critic(next_state, self.actor(next_state)).detach().numpy() if not done else 0
        td_target = reward + gamma * next_value

        value_prediction = self.critic(state, action)
        critic_loss = nn.MSELoss()(value_prediction, torch.FloatTensor(np.array([td_target])))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

    def update_actor(self, state, action, advantage):
        state = torch.FloatTensor(state)

        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, torch.exp(torch.zeros_like(mean_action) * 0.1))
        actor_loss = -self.critic(state, sampled_action) * advantage
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def update_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train_with_experience_replay(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return  # Wait until we have enough samples in the replay buffer

        # Sample a mini-batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        state, action, reward, next_state, terminated = batch

        # Update critic
        next_value = self.critic(next_state, self.actor(next_state)).detach().numpy()
        td_target = reward + 0.99 * next_value
        value_prediction = self.critic(state, action)
        critic_loss = nn.MSELoss()(value_prediction, td_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update actor
        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, torch.exp(torch.zeros_like(mean_action) * 0.1))
        actor_loss = -self.critic(state, sampled_action) * (td_target - value_prediction)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

def train_actor_critic_continuous_visualization_with_replay(agent, env, num_episodes=1000, exploration_decay=0.995, min_exploration_noise=0.01):
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        exploration_noise = 2.0  # initial exploration noise
        while True:
            action = agent.select_action(state, exploration_noise=exploration_noise)
            next_state, reward, done, _, _ = env.step([action])
            agent.update_replay_buffer(state, action, reward, next_state, done)
            agent.train_with_experience_replay()

            state_value = agent.critic(torch.FloatTensor(state), torch.FloatTensor([action])).item()
            next_state_value = agent.critic(torch.FloatTensor(next_state),
                                            agent.actor(torch.FloatTensor(next_state))).item() if not done else 0
            advantage = reward + 0.99 * next_state_value - state_value
            agent.update_actor(state, action, advantage)
            episode_reward += reward
            env.render()
            if done:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        # decay exploration noise
        exploration_noise = max(min_exploration_noise, exploration_noise * exploration_decay)

    env.close()

# Example usage:
# state_dim_continuous = 3
# action_dim_continuous = 1
# env_continuous = gym.make("Pendulum-v0")
# agent_continuous = ActorCriticAgentContinuousWithReplay(state_dim_continuous, action_dim_continuous)
# train_actor_critic_continuous_visualization_with_replay(agent_continuous, env_continuous)
