import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class tanhNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(tanhNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer2 = nn.Linear(neurons, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.tanh(self.layer1(obs))
		activation2 = F.tanh(self.layer2(activation1))
		output = 7.0 * F.tanh(self.layer3(activation2))

		return output
class RELUNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(RELUNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer2 = nn.Linear(neurons, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = 7.0 * F.tanh(self.layer3(activation2))

		return output

class tinytanhNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(tinytanhNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.tanh(self.layer1(obs))
		output = 7.0 * F.tanh(self.layer3(activation1))

		return output
class hugetanhNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(hugetanhNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer2 = nn.Linear(neurons, neurons)
		self.layer3 = nn.Linear(neurons, neurons)
		self.layer4 = nn.Linear(neurons, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.tanh(self.layer1(obs))
		activation2 = F.tanh(self.layer2(activation1))
		activation3 = F.tanh(self.layer3(activation2))
		output = 7.0 * F.tanh(self.layer4(activation3))

		return output
