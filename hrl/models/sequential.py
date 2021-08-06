import pfrl
from pfrl.policies import SoftmaxCategoricalHead
from torch import nn

from hrl.models.utils import lecun_init


class SequentialModel:
	def __init__(self, obs_n_channels, n_actions):
		self.model = nn.Sequential(
			lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
			nn.ReLU(),
			lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
			nn.ReLU(),
			lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
			nn.ReLU(),
			nn.Flatten(),
			lecun_init(nn.Linear(3136, 512)),
			nn.ReLU(),
			pfrl.nn.Branched(
				nn.Sequential(
					lecun_init(nn.Linear(512, n_actions), 1e-2),
					SoftmaxCategoricalHead(),
				),
				lecun_init(nn.Linear(512, 1)),
			),
		)