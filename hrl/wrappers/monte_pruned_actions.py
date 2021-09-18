from gym import Wrapper
from gym.spaces.discrete import Discrete


class MontePrunedActions(Wrapper):
	"""
	Prune the action set for Monte, so that there are no redundant actions
	this will make training a skill easier, because there are less actions to choose from
	"""
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		self.meaningful_actions = [
			'UP',
			'RIGHT',
			'LEFT',
			'DOWN',
			'UPFIRE',
			'UPRIGHTFIRE',
			'UPLEFTFIRE',
		]
		self.action_space = Discrete(len(self.meaningful_actions))
	
	def get_action_meanings(self):
		return self.meaningful_actions
