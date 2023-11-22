import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imageio

env = gym.make("MountainCarContinuous-v0")
env.reset()

LEARNING_RATE = 0.001
DISCOUNT = 0.99
EPISODES = 1000

SHOW_EVERY = 100

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = QNetwork(state_size, 24, action_size)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def get_scaled_state(state):
    if isinstance(state, tuple):
        return torch.from_numpy(state[0]).float().unsqueeze(0)
    else:
        return torch.from_numpy(state).float().unsqueeze(0)

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    state,_ = env.reset()
    frames = []
    done = False
    total_reward = 0

    while not done:
        action = model(state).detach().numpy()
        new_state_tuple, reward, terminated,truncated, _ = env.step(action.reshape(-1))
        done = terminated or truncated
        new_state = get_scaled_state(new_state_tuple)

        if render:
            frames.append(env.render())

        total_reward += reward

        if not done:
            max_future_q = np.max(model(new_state).detach().numpy())
            current_q = model((state)).detach().numpy()

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            loss = nn.MSELoss()(current_q, new_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = new_state

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

    if render:
        print(frames[0].shape)
        imageio.mimsave(f'./{episode}.gif', frames, fps=40)

env.close()



