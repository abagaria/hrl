import pfrl
import numpy as np
from torch import nn


def lecun_init(layer, gain=1):
	if isinstance(layer, (nn.Conv2d, nn.Linear)):
		pfrl.initializers.init_lecun_normal(layer.weight, gain)
		nn.init.zeros_(layer.bias)
	else:
		pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
		pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
		nn.init.zeros_(layer.bias_ih_l0)
		nn.init.zeros_(layer.bias_hh_l0)
	return layer


def phi(x):
	# Feature extractor
	return np.asarray(x, dtype=np.float32) / 255


# While the original paper initialized weights by normal distribution,
# we use orthogonal initialization as the latest openai/baselines does.
def apply_ortho_init(policy, vf):
	"""
	used for init for Actor-Critic
	args:
		policy: a policy network (not a layer)
		vf: a value function network (not a layer)
	"""
	def ortho_init(layer, gain):

		nn.init.orthogonal_(layer.weight, gain=gain)
		nn.init.zeros_(layer.bias)

	ortho_init(policy[0], gain=1)
	ortho_init(policy[2], gain=1)
	ortho_init(policy[4], gain=1e-2)
	ortho_init(vf[0], gain=1)
	ortho_init(vf[2], gain=1)
	ortho_init(vf[4], gain=1)
