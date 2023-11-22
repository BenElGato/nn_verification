import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from pendelum.neural_network.custom_environments import reward_positive_x as wrapper
import math
class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,layer1_size=64, layer2_size=64, n_outputs=1, epsilon=0.1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=1)
        self.epsilon = epsilon

    def choose_actions(self, observation):
        if np.random.uniform(0, 1) < self.epsilon:
            #action = np.random.uniform(-1.0, 1.0, self.n_outputs)
            action_probs = T.distributions.Uniform(-1.0,1.0)
            #TODO random choice
        else:
            mu, sigma = self.actor.forward(observation)
            sigma = T.exp(sigma)
            action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)
        return action.item()
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
def run_trained_model(alpha, beta, input_dims, gamma, layer1_size, layer2_size, folder, environment, n_actions=2, num_episodes=100):
    env = wrapper.CustomMountainCarRewardWrapper(gym.make(environment, render_mode="human"))
    agent = Agent(alpha=alpha, beta = beta, input_dims=input_dims, gamma=gamma, layer1_size=layer1_size, layer2_size=layer2_size, epsilon=0.1)
    agent.actor.load_state_dict(T.load(f'{folder}/actor_weights.pth'))
    agent.critic.load_state_dict(T.load(f'{folder}/critic_weights.pth'))
    score_history = []
    for i in range(num_episodes):
        done = False
        score = 0
        observation, _ = env.reset()
        while not done:
            action = np.array(agent.choose_actions(observation)).reshape((1,))
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode ', i, ' score %.2f' % score)

    env.close()
def train(alpha, beta, input_dims, gamma, layer1_size, layer2_size, folder, environment, num_episodes=100):
    agent = Agent(alpha=alpha, beta = beta, input_dims=input_dims, gamma=gamma, layer1_size=layer1_size, layer2_size=layer2_size, epsilon=0.5)
    #agent.actor.load_state_dict(T.load(f'{folder}/actor_weights.pth'))
    #agent.critic.load_state_dict(T.load(f'{folder}/critic_weights.pth'))
    #env = gym.make('MountainCarContinuous-v0', render_mode="human")
    env = wrapper.CustomMountainCarRewardWrapper(gym.make(environment))
    score_history = []
    for i in range(num_episodes):
        done = False
        score = 0
        observation, _ = env.reset()
        while not done:
            action = np.array(agent.choose_actions(observation)).reshape((1,))
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        score_history.append(score)
        print('episode ', i, ' score %.2f' % score)
    T.save(agent.actor.state_dict(), f'{folder}/actor_weights.pth')
    T.save(agent.critic.state_dict(), f'{folder}/critic_weights.pth')
    state_dim = env.observation_space.shape[0]
    dummy_input = T.randn(1, state_dim)
    onnx_filename = 'actor_model.onnx'
    T.onnx.export(agent.actor, dummy_input, onnx_filename, verbose=True)
    x_values = []
    for i in range(num_episodes):
        x_values.append(i)
    plt.plot(x_values, score_history)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Line Plot of Floats')
    plt.savefig('my_plot.png')
environment = 'Pendulum-v1'
folder = "/home/elgato/nn_verification/pendelum/neural_network/network_code/SimpleNN"
input_dims = gym.make(environment).observation_space.shape
train(alpha=1e-5, beta = 1e-5, input_dims=input_dims, gamma=0.99, layer1_size=256, layer2_size=256, num_episodes=2000, folder = folder, environment=environment)
run_trained_model(alpha=0.005, beta = 0.001, input_dims=input_dims, gamma=0.99, layer1_size=256, layer2_size=256, num_episodes=200, folder = folder, environment=environment)
