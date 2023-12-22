"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim, neurons):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer2 = nn.Linear(neurons, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):

		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
class FeedForwardInitialisationTanhNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(FeedForwardInitialisationTanhNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer2 = nn.Linear(neurons, out_dim)
		# -------------------------------Using He initialization for layers with RELU for better convergence-----------------------------
		'''
		@inproceedings{he2015delving,
  		title={Delving deep into rectifiers: Surpassing human-level performance on imagenet classification},
  		author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  		booktitle={Proceedings of the IEEE international conference on computer vision},
  		pages={1026--1034},
  		year={2015}
		}
		'''

		#neural_network.init.normal_(self.layer1.weight, mean=0.0, std=np.sqrt(2. / in_dim))
		#neural_network.init.constant_(self.layer1.bias, 0)

		'''
		Use Glorot initialization for tanh layer
		@inproceedings{glorot2010understanding,
  		title={Understanding the difficulty of training deep feedforward neural networks},
  		author={Glorot, Xavier and Bengio, Yoshua},
  		booktitle={Proceedings of the thirteenth international conference on artificial intelligence and statistics},
  		pages={249--256},
  		year={2010},
  		organization={JMLR Workshop and Conference Proceedings}
		}

		'''
		#neural_network.init.normal_(self.layer2.weight, mean=0.0, std=np.sqrt(2. / in_dim))
		#neural_network.init.constant_(self.layer2.bias, 0)
		# ----------------------------------------------------------------------------------------------------------

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.tanh(self.layer1(obs))
		output = 2.0 * F.tanh(self.layer2(activation1))
		return output
class SimpleFeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim, neurons):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(SimpleFeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):

		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		output = 2.0 * F.tanh(self.layer3(activation1))

		return output
class SimpleSigmoidNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(SimpleSigmoidNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)
		activation1 = F.sigmoid(self.layer1(obs))
		output = 2.0 * F.tanh(self.layer3(activation1))
		return output
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
		output = 2.0 * F.tanh(self.layer3(activation2))

		return output
class sigmoidNN(nn.Module):
	def __init__(self, in_dim, out_dim, neurons):
		super(sigmoidNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, neurons)
		self.layer2 = nn.Linear(neurons, neurons)
		self.layer3 = nn.Linear(neurons, out_dim)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.sigmoid(self.layer1(obs))
		activation2 = F.sigmoid(self.layer2(activation1))
		output = 2.0 * F.tanh(self.layer3(activation2))

		return output

	# TODO: Test model in matlab, try with relus, try to adapt batch size, include momentum, try SELU