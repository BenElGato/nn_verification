import sys

import onnx
from gymnasium.envs.registration import register
from matplotlib import pyplot as plt
from onnx import numpy_helper

from ACC.neural_network.network import tanhNN, RELU_NN, HeavyBrakes_RELU_NN
from pendelum.neural_network.custom_environments.Angle import AngleWrapper
from training.ppo import PPO
import torch
import gymnasium as gym

'''
This function can be used to visualize the behaviour of the neural network with the pygame visualization of gymnasium
inputs
---------------------------------------
env: The already made gym environment
actor_model: Path to the actor.pth file where the configuration of the neural network is stored
neurons: Number of neurons per layer --> MUST BE THE SAME AS IN TRAINING 
nn: Network class --> MUST BE THE SAME AS IN TRAINING 
obs_dim: Dimension of the observation space
act_dim: Dimension of the output space, actually always 1 in our cases
'''
def test(env, actor_model, neurons, nn, obs_dim, act_dim):
	print(f"Testing {actor_model}", flush=True)
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)
	policy = nn(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))
	observation, info = env.reset()
	for i in range(10000):
		action = policy(observation).detach().numpy() # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()
			print(observation)

	env.close()
'''
This function does the same as test(), but takes the network directly from an onnx file
inputs
--------------------------------------------------------------------------------------
env: The already made gym environment
path: Absolute path to the onnx file
neurons: Number of neurons per layer --> MUST BE THE SAME AS IN TRAINING 
nn: Network class --> MUST BE THE SAME AS IN TRAINING 
obs_dim: Dimension of the observation space
act_dim: Dimension of the output space, actually always 1 in our cases
'''
def test_onnx(env, path, nn, neurons, obs_dim, act_dim):
	onnx_model = onnx.load(path)
	weights = {}
	for initializer in onnx_model.graph.initializer:
		weights[initializer.name] = torch.tensor(numpy_helper.to_array(initializer))
	policy = nn(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(weights)
	observation, info = env.reset()
	for i in range(10000):
		action = policy(observation).detach().numpy()  # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			observation, info = env.reset()
			print(observation)

	env.close()

'''
This function exports a neural network to an onnx file, so that the network can be imported in MATLAB
inputs
------------------------------------------------------------------------
actor_model: Path to the actor.pth file where the configuration of the neural network is stored
path: Destination where the .onnx file should be saved
neurons: Number of neurons per layer --> MUST BE THE SAME AS IN TRAINING 
name: Name of the .onnx file
nn: Network class --> MUST BE THE SAME AS IN TRAINING 
obs_dim: Dimension of the observation space
act_dim: Dimension of the output space, actually always 1 in our cases
'''
def export_onnx(actor_model, path, neurons, name, nn, obs_dim, act_dim):
	policy = nn(obs_dim, act_dim, neurons=neurons)
	policy.load_state_dict(torch.load(actor_model))

	dummy_input = torch.randn(1, obs_dim)
	onnx_filename = f'{path}/{name}.onnx'
	torch.onnx.export(policy, dummy_input, onnx_filename, verbose=True, opset_version=10)
'''
Trains a neural network and saves it in path/ppo_actor{counter}.pth as well as path/ppo_critic{counter}.pth
inputs
---------------------------------------------------------------------------------------------------------
env: The already made gym environment
name: Name of the custom environment --> Must be the one which was registered
settings: List of hyperparameter dicts. Every hyperparameter dict will be trained, the first dict will have the number {counter}.
		  the numbers of the upcomming dicts will be incremented
path:	Destination where the .pth files should be stored
obs_dim: Dimension of the observation space
act_dim: Dimension of the output space, actually always 1 in our cases
counter: the first dict will have the number {counter}. the numbers of the upcomming dicts will be incremented
'''
def compare_settings(env, name, settings, path, obs_dim, act_dim, counter):
	for i in settings:
		model = PPO(policy_class=i["neural_network"], env=env, name=name, params=i, path=path, counter=counter, obs_dim=obs_dim, act_dim=act_dim)
		average_rewards = model.learn()
		print("")


		plt.plot(average_rewards, label=f"{counter}. settings")
		plt.legend()
		plt.title('Average Episodic Returns Over Iterations')
		plt.xlabel('Iteration')
		plt.ylabel('Average Return')
		plt.savefig(f'{path}/graph{counter}.png')
		plt.close()
		with open(f'{path}/report.md', 'a') as file:
			file.write(f"## Results for Setting {counter}\n")
			file.write(f"![Average Rewards Plot]({path}/graph{counter}.png)\n\n")
			file.write(f"{str(i)}\n\n")
		counter+=1

'''
Registration of the custom environments........................
'''
register(
        id='ACCEnv',
        entry_point='ACC.neural_network.ACC_environment:CustomEnv',
    )

'''
Declaration of the hyperparameters
'''
params = {'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99,
		  'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0,
		  'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7_000_000, 'neural_network': tanhNN}
scenario = 'pendulum'
render = True
if scenario == 'acc':
	env = gym.make("ACCEnv")
	name = "ACCEnv"
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
elif scenario == 'pendulum':
	name = 'Pendulum-v1'
	if render:
		env = gym.make(name, render_mode="human")
	else:
		env = gym.make(name)
	env = AngleWrapper(env)
	obs_dim = 2
	act_dim = env.action_space.shape[0]


actor_model = ""
critic_model = ""

settings = [params]

test_onnx(env,"/home/benedikt/PycharmProjects/nn_verification/pendelum/cora/network.onnx",tanhNN,30,obs_dim,act_dim)
#compare_settings(env,name, settings,"/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network", obs_dim=obs_dim,act_dim=act_dim)

#actor_model = f"/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/ppo_actor11.pth"

#test(env=env, actor_model=actor_model, neurons=params["neurons"], neural_network=params["neural_network"], obs_dim=obs_dim, act_dim=act_dim)
#export_onnx(actor_model=actor_model, path="/home/benedikt/PycharmProjects/nn_verification/ACC/cora", neurons=params["neurons"], name="network11", nn=params["neural_network"], obs_dim=obs_dim,act_dim=act_dim)


