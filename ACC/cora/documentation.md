#  network 0
    - reward = -(self.D_rel-self.D_safe)**2
    - It used a 3 layer (each tanh) network. The output was between -3 and 3
    - Each layer had 30 neurons
    - Training:
        -self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0
        self.x_ego = random.uniform(38, 58)
        self.v_ego = random.uniform(30, 30.2)
        self.a_ego = 0.0
        self.v_set = self.v_ego
        self.a_c_lead = random.uniform(-3.0, -1.0)
    
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network0.jpg)

![Training](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/graph0.png)

# network 1
    - reward = -(self.D_rel-self.D_safe)**2
    - It used a 3 layer (each RELU unless the last layer which is tanh) network. The output was between -3 and 3
    - Each layer had 30 neurons
    -Training
        self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0
        self.x_ego = random.uniform(38, 58)
        self.v_ego = random.uniform(30, 30.2)
        self.a_ego = 0.0
        self.v_set = self.v_ego
        self.a_c_lead = random.uniform(-3.0, -1.0)
![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network1.jpg)
![Training](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/graph1.png)
# Network 2: Adjusting the reward --> Training it to stay 5 meters above the safe distance instead of exactly at the distance!
    - reward = -(self.D_rel-self.D_safe - 10)**2
    - It used a 3 layer (each RELU unless the last layer which is tanh) network. The output was between -3 and 3
    - Each layer had 30 neurons
    -Training
        self.x_lead = random.uniform(90, 110)
        self.v_lead = random.uniform(32, 32.2)
        self.a_lead = 0.0
        self.x_ego = random.uniform(28, 48)
        self.v_ego = random.uniform(30, 30.2)
        self.a_ego = 0.0
        self.v_set = self.v_ego
        self.a_c_lead = random.uniform(-3.0, -1.0)

![Verification](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network2.jpg)
![Training](/home/benedikt/PycharmProjects/nn_verification/ACC/cora/graph2.png)



    