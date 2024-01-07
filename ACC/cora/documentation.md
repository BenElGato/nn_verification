
# network 8
    - It used a 3 layer (each relu unless the last layer which is tanh) network. The output was between -3 and 3
    - Output is between -7 and 7
    - Each layer had 32 neurons
    . Reward: r_a = -(self.a_ego)**2
        w_1 = 0.01
        '''
        Vehicle is in dangerous zone
        '''
        if self.D_rel <  1.25 * self.D_safe:
            w_2 = 2
        elif self.D_rel < 1.5 * self.D_safe:
            # TODO: here a linear function
            '''
            Desired distance
            '''
            w_2 = 0.5
        else:
            '''
            Pursuit
            '''
            w_2 = 2
        r_d = -w_2 * abs((self.D_rel / (1.375 * self.D_safe)) - 1)

        reward = w_1 * r_a + w_2 * r_d
    - Initial states:
    - self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(0, 32.2) # TODO: v lead variable machen
        self.a_lead = 0.0
        self.x_ego = random.uniform(0, 58)
        self.v_ego = random.uniform(self.v_lead - 5, self.v_lead + 5)
        self.a_ego = 0.0
        self.v_set = self.v_ego
        self.a_c_lead = -2.0
        self.dt = 0.1
        self.t = 0

        '''
        Parameters for calulating the rewards/costs
        '''
        self.D_Default = 10.0
        self.T_Gap = random.uniform(0.0, 3.0)
        self.D_rel = self.x_lead - self.x_ego
        self.D_safe = self.D_Default + self.T_Gap * self.v_ego
    - Training params: params = {'neurons': 32, 'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 50, 'gamma': 0.99, 'n_updates_per_iteration': 18, 'dynamic_lr': True, 'lr': 0.003, 'clip': 0.3, 'entropy_coef': 0.0, 'gradient_clipping': True, 'max_grad_norm': 0.1, 'total_timesteps': 7_000_000, 'neural_network': HeavyBrakes_RELU_NN}
## Problem a (Ego car starts between 50 and 58)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8a.jpg)
## Problem b (Ego car start between 42 and 50)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8b.jpg)
## Problem c (Ego car start between 34 and 42)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8c.jpg)
## Problem d (Ego car start between 26 and 34)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8d.jpg)
## Problem e (Ego car start between 18 and 26)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8e.jpg)
## Problem f (Ego car start between 10 and 18)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8f.jpg)
## Problem g (Ego car start between 0 and 10)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8g.jpg)
## All problems in one verification:
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network8all.jpg)
--> Splitting of starting space is necessary!!!

## Training of network 8
![Training](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/graph8.png)

# Network 9
Same as Network 8, but with 16 neurons a layer
## Problem a (Ego car starts between 50 and 58)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9a.jpg)
## Problem b (Ego car start between 42 and 50)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9b.jpg)
## Problem c (Ego car start between 34 and 42)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9c.jpg)
## Problem d (Ego car start between 26 and 34)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9d.jpg)
## Problem e (Ego car start between 18 and 26)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9e.jpg)
## Problem f (Ego car start between 10 and 18)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9f.jpg)
## Problem g (Ego car start between 0 and 10)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9g.jpg)
## All problems in one verification:
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network9all.jpg)
--> Less easy to verify, maybe because it is trained too short? Or less defined behaviour because of less neurons?
## Training of network 8
![Training](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/graph9.png)

# Network 11
3 tanh layers, 32 neurons
## Problem a (Ego car starts between 50 and 58)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11a.jpg)
## Problem b (Ego car start between 42 and 50)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11b.jpg)
## Problem c (Ego car start between 34 and 42)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11c.jpg)
## Problem d (Ego car start between 26 and 34)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11d.jpg)
## Problem e (Ego car start between 18 and 26)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11e.jpg)
## Problem f (Ego car start between 10 and 18)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11f.jpg)
## Problem g (Ego car start between 0 and 10)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11g.jpg)
## All problems in one verification:
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11all.jpg)
--> Reachable sets much bigger even though they use tanh....................why???
## Training of network 11
![Training](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/graph11.png)
    