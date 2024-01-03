## Results for Setting 5
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph5.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'ACC.neural_network.network.tanhNN'>}

## Results for Setting 4
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph4.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'ACC.neural_network.network.tanhNN'>}

# Network 4 and 5 are interesting
--> Verify
Both verified, but network 4 has problems with exploding sets --> Less neurons --> Leared beahviour worse
## Results for Setting 6
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph6.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.reluNN'>}

## Results for Setting 6
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph6.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.reluNN'>}

# RELU networks with problems to learn the behaviour --> logical because no smooth transition in activation function
## Results for Setting 7
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph7.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.reluNN'>}

