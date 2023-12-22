import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from gym import register


# Define the Actor network for continuous action space
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mean_action = torch.tanh(self.fc2(x))  # Using tanh to ensure the action is in the range [-1, 1]
        return mean_action


# Define the Critic network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.fc_action = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, state, action):
        x_state = torch.relu(self.fc_state(state))
        x_action = torch.relu(self.fc_action(action))
        x = torch.cat((x_state, x_action), dim=-1)
        value = self.fc2(x)
        return value


# Actor-Critic Agent for MountainCarContinuous-v0
'''class ActorCriticAgentContinuous:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr_actor=0.0001, lr_critic=0.0001):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, torch.exp(torch.zeros_like(mean_action) * 0.1))
        return torch.clamp(sampled_action, -1.0, 1.0).item()  # Ensure action is within the valid range

    def update_critic(self, state, action, reward, next_state, terminated, truncated, gamma=0.99, clip_double_q=True):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor([action])
        next_state = torch.FloatTensor(next_state)

        # Compute TD target
        next_value_1 = self.critic(next_state, self.actor(next_state)).detach().numpy() if not (
                terminated or truncated) else 0
        next_value_2 = self.critic(next_state, self.actor(next_state)).detach().numpy() if not (
                terminated or truncated) else 0
        next_value = min(next_value_1, next_value_2) if clip_double_q else next_value_1  # Clipped double Q-learning
        td_target = reward + gamma * next_value
        # Compute critic loss and update
        value_prediction = self.critic(state, action)
        critic_loss = nn.MSELoss()(value_prediction, torch.FloatTensor(np.array([td_target])))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

    def update_actor(self, state, action, advantage, entropy_weight=0.001):
        state = torch.FloatTensor(state)

        # Compute actor loss and update
        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, torch.exp(torch.zeros_like(mean_action) * 0.1))

        critic_value = self.critic(state, sampled_action).squeeze()
        entropy = -torch.sum(torch.log(torch.distributions.Normal(mean_action, 0.1).entropy()))

        actor_loss = -critic_value * advantage + entropy_weight * entropy

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()


def train_actor_critic_continuous_visualization(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # Select action from the actor
            action = agent.select_action(state)

            # Take the selected action
            next_state, reward, terminated, truncated, _ = env.step([action])

            # Update the critic
            agent.update_critic(state, action, reward, next_state, terminated, truncated)

            # Compute advantage for updating the actor
            state_value = agent.critic(torch.FloatTensor(state), torch.FloatTensor([action])).item()
            next_state_value = agent.critic(torch.FloatTensor(next_state),
                                            agent.actor(torch.FloatTensor(next_state))).item() if not (
                        terminated or truncated) else 0
            advantage = reward + 0.99 * next_state_value - state_value

            # Update the actor
            agent.update_actor(state, action, advantage)

            episode_reward += reward

            # Visualize the agent's behavior
            env.render()

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()  # Close the environment after training


# Create the environment
env_continuous = gym.make('MountainCarContinuous-v0', render_mode="human")

# Get the state and action dimensions
state_dim_continuous = env_continuous.observation_space.shape[0]
action_dim_continuous = env_continuous.action_space.shape[0]

# Create the Actor-Critic agent for MountainCarContinuous-v0
agent_continuous = ActorCriticAgentContinuous(state_dim_continuous, action_dim_continuous)

# Train the agent with continuous visualization
train_actor_critic_continuous_visualization(agent_continuous, env_continuous, num_episodes=2)
torch.save(agent_continuous.actor.state_dict(), 'actor_weights.pth')

# If applicable, save the weights of the trained critic network


state_dim = state_dim_continuous  # Replace with the actual state dimension of your environment
action_dim = action_dim_continuous
hidden_dim = 128
actor = ActorNetwork(state_dim, action_dim, hidden_dim)


# Load the trained weights (replace 'actor_weights.pth' and 'critic_weights.pth' with the actual paths to your saved weights)
actor.load_state_dict(torch.load('actor_weights.pth'))


# Set the models to evaluation mode
actor.eval()


# Create dummy input tensors with the appropriate sizes and data types
dummy_input_actor = torch.randn(1, state_dim)
dummy_input_critic = torch.randn(1, state_dim)

# Export the Actor model to ONNX format
onnx_filename_actor = 'actor_model.onnx'
torch.onnx.export(actor, dummy_input_actor, onnx_filename_actor, verbose=True, opset_version=10)
print(f"The ONNX actor model has been exported to {onnx_filename_actor}")'''

register(
    id='TORAEnv-v0',
    entry_point='TORA_environment:TORAEnv',  # Replace with the actual path to your TORAEnv class
)


class ActorCriticAgentContinuous:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr_actor=0.00005, lr_critic=0.00005):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, torch.ones_like(mean_action))
        return torch.clamp(sampled_action, -1.0, 1.0).item()

    def update_critic(self, state, action, reward, next_state, terminated, truncated, gamma=0.99):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor([action])
        next_state = torch.FloatTensor(next_state)

        next_value = self.critic(next_state, self.actor(next_state)).detach().numpy() if not (
                terminated or truncated) else 0
        td_target = reward + gamma * next_value
        # Increase the reward when the agent gets closer to the target position

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


def train_actor_critic_continuous_visualization(agent, env, num_episodes=1000):
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step([action])
            agent.update_critic(state, action, reward, next_state, terminated, truncated)
            state_value = agent.critic(torch.FloatTensor(state), torch.FloatTensor([action])).item()
            next_state_value = agent.critic(torch.FloatTensor(next_state),
                                            agent.actor(torch.FloatTensor(next_state))).item() if not (
                    terminated or truncated) else 0
            advantage = reward + 0.99 * next_state_value - state_value
            agent.update_actor(state, action, advantage)
            episode_reward += reward
            env.render()
            if terminated or truncated:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
        # decay exploration noise

    env.close()


def training_and_exporting(agent_continuous, env_continuous, state_dim_continuous, action_dim_continuous):
    train_actor_critic_continuous_visualization(agent_continuous, env_continuous)
    torch.save(agent_continuous.actor.state_dict(), 'actor_weights.pth')
    torch.save(agent_continuous.critic.state_dict(), 'critic_weights.pth')

    state_dim = state_dim_continuous
    action_dim = action_dim_continuous
    hidden_dim = 128
    actor = ActorNetwork(state_dim, action_dim, hidden_dim)
    actor.load_state_dict(torch.load('actor_weights.pth'))

    actor.eval()

    # Exporting to onnx
    dummy_input = torch.randn(1, state_dim)
    onnx_filename = 'actor_model12.onnx'
    torch.onnx.export(actor, dummy_input, onnx_filename, verbose=True)


# TODO: Is it a problem that the exploration noise is still added??
def run_trained_model(state_dim, action_dim, hidden_dim, env):
    actor = ActorNetwork(state_dim, action_dim, hidden_dim)
    actor.load_state_dict(torch.load('actor_weights.pth'))
    actor.eval()
    for episode in range(1000):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            state = torch.FloatTensor(state)
            action = actor(state)
            action = torch.normal(action, torch.ones_like(action))
            action = torch.clamp(action, -1.0, 1.0).item()
            next_state, reward, terminated, truncated, _ = env.step([action])

            episode_reward += reward
            env.render()
            if terminated or truncated:
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()


env_continuous = gym.make('TORA_environment:TORAEnv-v0')
# env_continuous = gym.make('MountainCarContinuous-v0', render_mode="human")


state_dim_continuous = env_continuous.observation_space.shape[0]
action_dim_continuous = env_continuous.action_space.shape[0]
agent_continuous = ActorCriticAgentContinuous(state_dim_continuous, action_dim_continuous)

training_and_exporting(agent_continuous, env_continuous, state_dim_continuous, action_dim_continuous)
