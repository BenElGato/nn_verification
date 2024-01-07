
# Tanh 32 networks

## Results for Setting 22
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph22.png)

{'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.tanhNN'>}
Learned behaviour but will have big problems with exploding sets, maybe a good demonstration



------------------------------------------------
# RELU 16 networks
## Results for Setting 26
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph26.png)

{'neurons': 16, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.RELUNN'>}
--> Perfect behavior ;)



-------------------------------------------------------------------
# RELU 32 networks
## Results for Setting 30
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph30.png)

{'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.RELUNN'>}
- works ;)
## Results for Setting 32
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph32.png)

{'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.RELUNN'>}
- Musterverhalten
## Results for Setting 8
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph8.png)

{'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': RELUNN}
--------------------------------------------------------------
# Tanh 16 networks
## Results for Setting 36
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph36.png)

{'neurons': 16, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.tanhNN'>}
- Perfect behaviour

--------------------------------------------------------------------------------
# TinyTanh 16 networks

## Results for Setting 61
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph61.png)

{'neurons': 16, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.tinytanhNN'>}
- perfectooo :)


------------------------------------------------------------------------
# TinyTanh 32 networks

## Results for Setting 64
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph64.png)

{'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.tinytanhNN'>}

# Hugetanhh 16


## Results for Setting 68
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph68.png)

{'neurons': 16, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.hugetanhNN'>}

- perfect

# Hugetanh 32
## Results for Setting 70
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/ACC/neural_network/graph70.png)

{'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.hugetanhNN'>}

- perfect