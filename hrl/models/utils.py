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
