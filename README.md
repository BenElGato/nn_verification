# Practical Course Submission
## Training new neural networks
New neural networks can be trained by using the following command:
  - python3 main.py parameters environment neural_network_name
  - parameters: Path to file in which the parameters are stored like this:
    - {'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7_000_000, 'neural_network': hugetanhNN}
    - For demo purposes, just use the parameters.txt file in the training folder
  - environment: ACC or pendulum
  - neural_network_name: All neural network related files will consist this name
### Outputs
    - An onnx file that is used for exporting the neural network to matlab
    - A graph that visualizes the training process
    - The actor and critic weights
## Verifying neural networks
    - Go to ./ACC/cora or ./pendulum/cora
    - Open the matlab code
    - Specify the initial parameters as well as the neural network as described in the comments of the matlab code