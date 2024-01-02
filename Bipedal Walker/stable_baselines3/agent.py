import gym
import torch

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Retrieve the model from the hub
## repo_id =  id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(repo_id="mrm8488/ppo-BipedalWalker-v3", filename="bipedalwalker-v3.zip")
model = PPO.load(checkpoint)

# Evaluate the agent
env = gym.make('BipedalWalker-v3',render_mode='human')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Export the model to ONNX
dummy_input = torch.randn(1, env.observation_space.shape[0])
onnx_filename = 'ppo_bipedal_walker.onnx'
torch.onnx.export(model.policy, dummy_input, onnx_filename, verbose=True)


# Watch the agent play
obs,_ = env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, terminated,truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs,_ = env.reset()
env.close()

