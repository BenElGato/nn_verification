import gymnasium as gym
import sys

import numpy as np
import torch
from gymnasium import register

from ppo import PPO
from network import FeedForwardNN
import matplotlib.pyplot as plt


def train(env, timesteps_per_batch, max_timesteps_per_episode, gamma, n_updates_per_iteration, lr, clip, actor_model,
          critic_model, entropy_coef, name, timesteps=1_000_000):
    model = PPO(policy_class=FeedForwardNN, env=env, timesteps_per_batch=timesteps_per_batch,
                max_timesteps_per_episode=max_timesteps_per_episode,
                gamma=gamma, n_updates_per_iteration=n_updates_per_iteration, lr=lr, clip=clip, name=name,
                total_timesteps=timesteps, entropy_coef=entropy_coef)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)
    average_rewards = model.learn()
    plt.plot(average_rewards, label='')

    plt.legend()
    plt.title('Average Episodic Returns Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.savefig('training_process_second_network_fixed_state16.png')

    batch_obs, _, _, _, _ = model.collectData()
    batch_obs = np.array(batch_obs)
    num_timesteps = batch_obs.shape[0]

    # Plotting x1 and x2 over time in separate plots
    plot_state_variable(batch_obs, num_timesteps, variable_index=0, variable_name='x1',
                        save_path='plots/tora_system_state_variable_x1_plot_second_network_secondreward_fixed_state16.png')

    plot_state_variable(batch_obs, num_timesteps, variable_index=1, variable_name='x2',
                        save_path='plots/tora_system_state_variable_x2_plot_second_network_secondreward_fixed_state16.png')

    plot_state_variable(batch_obs, num_timesteps, variable_index=2, variable_name='x3',
                        save_path='plots/tora_system_state_variable_x3_plot_second_network_secondreward_fixed_state16.png')

    plot_state_variable(batch_obs, num_timesteps, variable_index=3, variable_name='x4',
                        save_path='plots/tora_system_state_variable_x4_plot_second_network_secondreward_fixed_state16.png')

    # Show the plots (optional)
    plt.show()

    # Reshape the data for plotting


def plot_state_variable(batch_obs, num_timesteps, variable_index, variable_name, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(num_timesteps), batch_obs[:, variable_index], label=variable_name)

    plt.title(f'TORA System State Variable {variable_name} over Time')
    plt.xlabel('Timestep')
    plt.ylabel(f'State Variable {variable_name}')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)


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
        action = policy(observation).detach().numpy()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def export_onnx(env, actor_model, batchsize, path):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = FeedForwardNN(obs_dim, act_dim)
    policy.load_state_dict(torch.load(actor_model))

    dummy_input = torch.randn(1, obs_dim)
    print(dummy_input.size())
    onnx_filename = f'{path}/actor_model_tora_second_network_second_reward_fixed_state16.onnx'
    torch.onnx.export(policy, dummy_input, onnx_filename, verbose=True, opset_version=10)

register(
    id='TORAEnv-v0',
    entry_point='TORA_environment:TORAEnv',  # Replace with the actual path to your TORAEnv class
)

if __name__ == '__main__':
    '''
	Hyperparamters ###########################
	'''
    timesteps_per_batch = 2048
    max_timesteps_per_episode = 400
    gamma = 0.99
    n_updates_per_iteration = 10
    lr = 3e-4
    clip = 0.1
    '''
	####################################
	'''
    name = 'TORAEnv-v0'
    env = gym.make(name)
    actor_model = ""
    critic_model = ""

    train(env=env, timesteps_per_batch=timesteps_per_batch, max_timesteps_per_episode=max_timesteps_per_episode,
          gamma=gamma, n_updates_per_iteration=n_updates_per_iteration, lr=lr, clip=clip,
          actor_model=actor_model, critic_model=critic_model, timesteps=1_000_000, name=name, entropy_coef=0.0)

    actor_model = f"{name}ppo_actor.pth"
    env = gym.make(name)
    test(env=env, actor_model=actor_model)
    export_onnx(env, actor_model=actor_model, batchsize=timesteps_per_batch,
                path="/Users/rayenmhadhbi/PycharmProjects/nn_verification/TORA/cora")
