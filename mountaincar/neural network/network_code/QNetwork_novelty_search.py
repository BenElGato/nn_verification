import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# Define the neural network architecture
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# Calculate the car's total energy
def calculate_total_energy(state):
    position, velocity = state[0], state[1]
    gravity = 9.8
    mass = 1.0

    potential_energy = mass * gravity * position
    kinetic_energy = 0.5 * mass * velocity ** 2

    total_energy = potential_energy + kinetic_energy
    return total_energy


# Reshape the reward function based on changes in total energy and novelty search
def reshape_reward(state, next_state, reward, done, novelty_scores):
    total_energy_before = calculate_total_energy(state)
    total_energy_after = calculate_total_energy(next_state)
    energy_change = total_energy_after - total_energy_before

    # Convert the one-dimensional state to a tuple of floats
    state_index = tuple(map(float, state))

    # Calculate the novelty score based on state visitation frequency
    novelty_score = novelty_scores.get(state_index, 0)
    novelty_bonus = novelty_score * 0.1  # Scaling factor for novelty bonus

    # Combine energy change and novelty bonus as the reshaped reward
    reshaped_reward = energy_change + novelty_bonus

    return reshaped_reward


# Update the novelty scores based on state visitation frequency
def update_novelty_scores(states, novelty_scores):
    for state in states:
        # Convert the one-dimensional state to a tuple of floats
        state_index = tuple(map(float, state))
        if state_index not in novelty_scores:
            novelty_scores[state_index] = 0
        else:
            novelty_scores[state_index] += 1


# Hyperparameters
discount_factor = 0.9
learning_rate = 0.5
num_episodes_per_stage = 50
replay_buffer_capacity = 10000
batch_size = 256
epsilon_start = 10.0
epsilon_end = 0.01
epsilon_decay = 0.895

# Initialize the environment and the Q-network
env = gym.make('MountainCarContinuous-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
q_network = QNetwork(state_size, action_size)
optimizer = optim.SGD(q_network.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(replay_buffer_capacity)
novelty_scores = {}  # Dictionary to store novelty scores for each state


# Epsilon-greedy exploration
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.uniform(-1, 1, action_size)
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = q_network(state)
        return action_values.numpy()[0]


# Training loop
epsilon = epsilon_start
for stage in range(1, 4):  # Three stages with increasing difficulty
    # Adjust the environment parameters based on the stage
    env.hill_difficulty = stage * 0.1  # Example adjustment, modify based on your environment

    for episode in range(num_episodes_per_stage):
        state,_ = env.reset()
        episode_reward = 0
        episode_states = []

        while True:
            # Select action using epsilon-greedy policy
            action = epsilon_greedy_policy(state, epsilon)

            # Take action and observe next state and reward
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reshape_reward(state, next_state, 0, done, novelty_scores)  # Use reshape_reward here

            # Store transition in replay buffer with reshaped reward
            transition = (state, action, reshape_reward(state, next_state, 0, done, novelty_scores), next_state, done)
            replay_buffer.push(transition)
            episode_states.append(state)

            # Sample a random batch from the replay buffer
            if len(replay_buffer.buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)

                # Compute Q targets
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                with torch.no_grad():
                    target_values = rewards + discount_factor * (1 - dones) * torch.max(q_network(next_states), dim=1)[
                        0]

                # Compute Q values and update the network
                q_values = q_network(states)
                selected_q_values = torch.sum(q_values * actions, dim=1)
                loss = nn.MSELoss()(selected_q_values, target_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update novelty scores based on state visitation frequency
            update_novelty_scores(episode_states, novelty_scores)

            # Render the environment
            

            state = next_state

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        print(f"Stage: {stage}, Episode: {episode + 1}, Reward: {episode_reward}")

# Close the environment after training
env.close()
