import pfrl
from pfrl.policies import SoftmaxCategoricalHead
from torch import nn

from hrl.models.utils import lecun_init


class RecurrentModel:
	def __init__(self, obs_n_channels, n_actions):
		self.model = pfrl.nn.RecurrentSequential(
			lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
			nn.ReLU(),
			lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
			nn.ReLU(),
			lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
			nn.ReLU(),
			nn.Flatten(),
			lecun_init(nn.Linear(3136, 512)),
			nn.ReLU(),
			lecun_init(nn.GRU(num_layers=1, input_size=512, hidden_size=512)),
			pfrl.nn.Branched(
				nn.Sequential(
					lecun_init(nn.Linear(512, n_actions), 1e-2),
					SoftmaxCategoricalHead(),
				),
				lecun_init(nn.Linear(512, 1)),
			),
		)

