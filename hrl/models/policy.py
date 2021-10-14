import torch
import pfrl
from torch import nn


class PolicyNetwork:
	"""
	a linear sequential network as the policy
	"""
	def __init__(self, obs_size, action_size):
		self.model = torch.nn.Sequential(
			nn.Linear(obs_size, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, action_size),
			pfrl.policies.GaussianHeadWithStateIndependentCovariance(
				action_size=action_size,
				var_type="diagonal",
				var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
				var_param_init=0,  # log std = 0 => std = 1
        ),
    )
