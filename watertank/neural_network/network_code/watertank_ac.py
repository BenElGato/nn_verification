import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dm_control.rl import control
from dm_control.rl.control import Environment
from dm_env import specs
import matplotlib.pyplot as plt



class Physics(control.Physics):
    """Water tank environment built on the dm_control.Environment class."""
    def __init__(
        self,
        alpha: float,
        dt_sim: float,
        hmax: float,
        init_state: [float],
        ):
        """Initializes water tank

        Attributes:
            alpha: nozzle outflow coefficient
            dt_sim:  [s] Discretization time interval for sim
            hmax: [m] max water height in tank
            init_state: [m] initial water height
        """
        self._alpha = alpha
        self._dt_sim = dt_sim
        self._h_max = hmax
        self._init_state = init_state
        self._state = self._init_state
        self._time = 0.
        self._action =  np.asarray([0.])

    def reset(self):
        """Resets environment physics"""
        self._state = self._init_state
        self._time = 0.
        self._action = np.asarray([0.])

    def after_reset(self):
       pass

    def step(self, n_sub_steps = 1):
        """Updates the environment according to the action"""

        # Euler explicit time step
        self._state = self._dt_sim*self._F() + self._state

        # Update sim time
        self._time += self._dt_sim

        # Keep h min at 0
        if self._state[0] <= 0.: self._state[0] = 0.

    def _F(self):
        """ Returns Physical RHS for ODE d state / dt = F(state, action) """
        return -self._alpha*np.sqrt(self._state) + self._action

    def time(self):
        """Returns total elapsed simulation time"""
        return self._time

    def timestep(self):
        """Returns dt simulation step"""
        return self._dt_sim

    def check_divergence(self):
        """ Checks physical terminations:
         - water reached maximum level
         - physical states not finite
         """
        if self._state[0] >= self._h_max:
            raise control.PhysicsError(
                f'h > max value = {self._h_max} [m]'
            )

        if not all(np.isfinite(self._state)):
            raise control.PhysicsError('System state not finite')

    def set_control(self, action):
        """Sets control actions"""
        self._action = action

    def get_state(self):
        """Returns physical states"""
        return self._state
class Step(control.Task):
    """ Step task:
    Keep constant value and step to different constant value at t_step
    """

    def __init__(self,
                 maxinflow: float,
                 h_goal: float,
                 precision: float,
                 ):
        """Initialize Step task

        Parameters:
            maxinflow: max control inflow
            h_goal: [m] target height 1st time interval
            precision: [m] desired precision on h target

        """
        self._maxinflow = maxinflow
        self._h_goal = h_goal
        self._precision = precision


    def initialize_episode(self, physics):
        """ Reset physics for the task """
        physics.reset()

    def get_reference(self):
        """Returns target reference"""
        return self._h_goal


    def get_observation(self, physics):
        """Returns specific observation for the task"""
        # Let the actor observe the reference and the state
        return np.concatenate((
            [self.get_reference()],
            physics.get_state()
        ))

    def get_reward(self, physics):
        """Returns the reward given the physical state """
        sigma = self._precision
        mean = self.get_reference()
        # Gaussian like rewards on target water level h
        return np.exp(
            -np.power(physics.get_state()[0] - mean, 2.) / (2 * np.power(sigma, 2.))
        )

    def before_step(self, action, physics):
        physics.set_control(action)

    def observation_spec(self, physics):
        """Returns the observation specifications"""
        return specs.Array(
            shape=(2,),
            dtype=np.float32,
            name='observation')

    def action_spec(self, physics):
        """Returns the action specifications"""
        return specs.BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=0.,
            maximum=self._maxinflow,
            name='action')





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

class ActorCriticAgentContinuous:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr_actor=0.0001, lr_critic=0.0001):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean_action = self.actor(state)
        sampled_action = torch.normal(mean_action, torch.ones_like(mean_action))
        return torch.clamp(sampled_action, 0.0, 5.0).item()  # Assuming the action range is [0.0, 5.0]

    def update_critic(self, state, action, reward, next_state, gamma=0.99):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor([action])
        next_state = torch.FloatTensor(next_state)

        next_value = self.critic(next_state, self.actor(next_state)).detach().numpy()
        td_target = reward + gamma * next_value

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


    def compute_advantage(self, rewards, values, next_value, gamma=0.99):
      returns = []
      advantages = []
      R = next_value

      for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        advantage = R - values[t]
        returns.insert(0, R)
        advantages.insert(0, advantage)

      return returns, advantages
# Create the environment, physics, and task
physics = Physics(
    alpha=1.0,
    dt_sim=0.05,
    hmax=5,
    init_state=[1.],
)
task = Step(
    maxinflow=5.,
    h_goal=1.,
    precision=0.05,
)
environment = Environment(
    physics,
    task,
    time_limit=2.5,
)

# Get observation and action dimensions
observation_spec = task.observation_spec(physics)
state_dimensions = observation_spec.shape[0]
action_spec = task.action_spec(physics)
action_dimensions = action_spec.shape[0]

# Create and initialize the agent
agent_continuous = ActorCriticAgentContinuous(state_dimensions, action_dimensions)

# Training loop
num_episodes = 1000
max_steps_per_episode = 200

controls_over_time = [] # To store control actions for visualization
precision_over_time = []
observations_over_time = []  # To store observations for visualization
rewards_over_time = []


for episode in range(num_episodes):
    physics.reset()
    task.initialize_episode(physics)

    episode_states, episode_actions, episode_rewards, episode_values = [], [], [], []

    for step in range(max_steps_per_episode):
        state = task.get_observation(physics)
        observations_over_time.append(state[0])
        action = agent_continuous.select_action(state)
        task.before_step(action, physics)


        physics.step()

        next_state = task.get_observation(physics)
        reward = task.get_reward(physics)

        # Store state, action, reward
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        controls_over_time.append(action)# Store control actions for visualization
        precision_over_time.append(task._precision)
        rewards_over_time.append(reward)


        if step == max_steps_per_episode - 1:
            # If we reached the maximum number of steps, consider the episode terminated
            break

    # Ensure that there are values to compute advantage
    if episode_values:
        # Compute returns and advantages
        next_value = agent_continuous.critic(torch.FloatTensor([next_state]),
                                             agent_continuous.actor(torch.FloatTensor([next_state]))).detach().numpy()

        returns, advantages = agent_continuous.compute_advantage(episode_rewards, episode_values, next_value)

        # Update Critic
        agent_continuous.update_critic(np.vstack(episode_states), np.array(episode_actions), returns)

        # Update Actor
        agent_continuous.update_actor(np.vstack(episode_states), np.array(episode_actions), advantages)

# Visualize controls over time

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time step')
ax1.set_ylabel('Control Action', color=color)
ax1.plot(controls_over_time, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Precision', color=color)  # we already handled the x-label with ax1
ax2.plot(precision_over_time, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # ensure that the two y-axis labels don't overlap

plt.savefig('controls&precision_over_timev4.png')
plt.show()

# Save the third plot (Rewards over Time)
fig, ax = plt.subplots()
ax.plot(rewards_over_time, color='purple')
ax.set_xlabel('Time step')
ax.set_ylabel('Rewards', color='purple')
ax.set_title('Rewards over Time')
fig.tight_layout()
plt.savefig('rewards_over_timev4.png')
plt.show()

torch.save(agent_continuous.actor.state_dict(), 'watertankactor_weightsv4.pth')

state_dim = state_dimensions
print(state_dim)
action_dim = action_dimensions
print(action_dim)
hidden_dim = 128
actor = ActorNetwork(state_dim, action_dim, hidden_dim)


# Load the trained weights
actor.load_state_dict(torch.load('watertankactor_weightsv4.pth'))


# Set the models to evaluation mode
actor.eval()

# Create dummy input tensors with the appropriate sizes and data types
dummy_input_actor = torch.randn(1, state_dim)
dummy_input_critic = torch.randn(1, state_dim)

# Export the Actor model to ONNX format
onnx_filename_actor = 'watertankactor_modelv4.onnx'
torch.onnx.export(actor, dummy_input_actor, onnx_filename_actor, verbose=True, opset_version=10)
print(f"The ONNX actor model has been exported to {onnx_filename_actor}")


