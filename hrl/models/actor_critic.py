import pfrl

from hrl.models.policy import PolicyNetwork
from hrl.models.value_function import ValueFunctionNetwork
from hrl.models.utils import apply_ortho_init


class ActorCritic:
	"""
	an actor critic network with a policy network with a value function network
	both the networks are composed of linear sequential layers
	"""
	def __init__(self, obs_size, action_size):
		self.policy = PolicyNetwork(obs_size, action_size).model
		self.vf = ValueFunctionNetwork(obs_size).model

		# orthogonal initialization 
		apply_ortho_init(self.policy, self.vf)

		self.model = pfrl.nn.Branched(self.policy, self.vf)
