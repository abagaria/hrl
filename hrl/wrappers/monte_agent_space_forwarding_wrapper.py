from pathlib import Path

import numpy as np
from gym import Wrapper
from gym.spaces.box import Box

from hrl.option.utils import set_player_position


class MonteAgentSpaceForwarding(Wrapper):
	"""
	forwards the agent to another state when the agent starts
	this just overrides the reset method and make it start in another position
	"""
	def __init__(self, env, forwarding_target_state: Path, forwarding_target_pos: Path):
		"""
		forward the agent to start in state `forwarding_target`
		Args:
			forwarding_target_state: a previously saved .npy file that contains the start state
			forwarding_target_pos: a previously saved .txt file that contains the start state position
		"""
		super().__init__(env)
		self.env = env
		self.target_state = np.load(forwarding_target_state)
		self.target_pos = np.loadtxt(forwarding_target_pos)
	
	def reset(self):
		self.env.reset()
		set_player_position(self.env, *self.target_pos)
		return self.target_state
