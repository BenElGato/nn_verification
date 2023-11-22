from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.act_limit = 1.0

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return self.act_limit * action


# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


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


class TD3Agent:

    def __init__(self, state_dim, action_dim, hidden_dim=256, discount=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, lr_actor=1e-3, lr_critic=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state = torch.FloatTensor(state).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy()

    def update(self, replay_buffer, batch_size=64):
        self.total_it += 1

        # Sample a batch from the replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        # Update Critic networks
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.discount * target_Q).detach()

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        critic_loss_1 = nn.MSELoss()(current_Q1, target_Q)
        critic_loss_2 = nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)


# Create the MountainCarContinuous environment
env = gym.make("MountainCarContinuous-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Initialize the TD3 agent
agent = TD3Agent(state_dim, action_dim)

# Training parameters
max_episodes = 500
max_steps = 500
batch_size = 64
replay_buffer_size = 10000
exploration_noise = 1

# Replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)

# Training loop
for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Select action with exploration noise
        action = agent.select_action(state)
        action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(
            -1.0, 1.0)  # Add exploration noise

        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Store the transition in the replay buffer
        replay_buffer.add((state, action, reward, next_state, float(not (terminated or truncated))))

        state = next_state
        episode_reward += reward

        # Update the agent
        if len(replay_buffer.buffer) > batch_size:
            agent.update(replay_buffer, batch_size)

        if terminated or truncated:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

env.close()
