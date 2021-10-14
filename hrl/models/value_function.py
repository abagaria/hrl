import torch
from torch import nn


class ValueFunctionNetwork:
	"""
	a linear sequential model as the value function
	"""
	def __init__(self, obs_size):
		self.model = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
