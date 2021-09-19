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
		# overried unwrapped method for easier access
		self.env.unwrapped.get_action_meanings = self.get_action_meanings
	
	def get_action_meanings(self):
		return self.meaningful_actions
	
	def step(self, action):
		if action not in self.action_space:
			raise RuntimeError('action not in range for pruned actions')
		action_to_original_action = {
			0: 2,  # UP
			1: 3,  # RIGHT
			2: 4,  # LEFT
			3: 5,  # DOWN
			4: 10,  # UPFIRE
			5: 13,  # UPRIGHTFIRE
			6: 14,  # UPLEFTFIRE
		}
		original_action = action_to_original_action[action]
		return self.env.step(original_action)
