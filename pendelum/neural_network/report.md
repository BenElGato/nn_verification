## Results for Setting 1
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph5.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'ACC.neural_network.network.tanhNN'>}

## Results for Setting 2
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph4.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'ACC.neural_network.network.tanhNN'>}

--> Verify
Both verified, but network 4 has problems with exploding sets --> Less neurons --> Leared beahviour worse

## Results for Setting 3
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph6.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.RELUNN'>}
--> Sim looked good

## Results for Setting 5
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph8.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 10000000, 'neural_network': <class 'training.network.tinytanhNN'>}
--> Looks good in sim, but didn't learn it for every starting position and will sometimes have problems with exp,loding sets
--> Keep for demonstrating need for long training, similar actions for similar starting states,.........

## Results for Setting 7
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph10.png)

{'neurons': 32, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 10000000, 'neural_network': <class 'training.network.hugetanhNN'>}
--> Well learned, will have some problems as Network 8 with exploding sets and is sometimes too slow to get to the angle
## Results for Setting 8
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph11.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 10000000, 'neural_network': <class 'training.network.hugetanhNN'>}
--> Perfect behaviour! Nice! -_> learned faster than with 32 neurons ;)

## Results for Setting 6
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph17.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 10000000, 'neural_network': <class 'training.network.tinytanhNN'>}
Yebbbbbbbbbbbbbbbbbbb
-------------------------------------------------
## Results for Setting 4
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph16.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.RELUNN'>}
worksssssss


## Results for Setting 19
![Average Rewards Plot](/home/benedikt/PycharmProjects/nn_verification/pendelum/neural_network/graph19.png)

{'neurons': 16, 'timesteps_per_batch': 5000, 'max_timesteps_per_episode': 500, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7000000, 'neural_network': <class 'training.network.RELUNN'>}

19 has some things dont work, maybe a good example how it looks when it didnt learn it for all starting states


