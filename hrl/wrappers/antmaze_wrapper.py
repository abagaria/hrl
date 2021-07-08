import numpy as np
import torch

from hrl.wrappers.gc_mdp_wrapper import GoalConditionedMDPWrapper


class D4RLAntMazeWrapper(GoalConditionedMDPWrapper):
	def __init__(self, env, start_state, goal_state):
		self.norm_func = lambda x: np.linalg.norm(x, axis=-1) if isinstance(x, np.ndarray) else torch.norm(x, dim=-1)
		super().__init__(env, start_state, goal_state)
	
	def sparse_gc_reward_func(self, states, goals):
		"""
		overwritting sparse gc reward function for antmaze
		"""
		# assert input is np array or torch tensor
		assert isinstance(states, (np.ndarray, torch.Tensor))
		assert isinstance(goals, (np.ndarray, torch.Tensor))

		current_positions = states[:, :2]
		goal_positions = goals[:, :2]
		distances = self.norm_func(current_positions-goal_positions)
		dones = distances <= self.goal_tolerance

		rewards = np.zeros_like(distances)
		rewards[dones==1] = +0.
		rewards[dones==0] = -1.

		return rewards, dones
	
	def dense_gc_reward_func(self, states, goals):
		"""
		overwritting dense gc reward function for antmaze
		"""
		assert isinstance(states, (np.ndarray, torch.Tensor))
		assert isinstance(goals, (np.ndarray, torch.Tensor))

		current_positions = states[:, :2]
		goal_positions = goals[:, :2]
		distances = self.norm_func(current_positions - goal_positions)
		dones = distances <= self.goal_tolerance

		assert distances.shape == dones.shape == (states.shape[0], ) == (goals.shape[0], )

		rewards = -distances
		rewards[dones==True] = 0

		return rewards, dones
	
	def step(self, action):
		# TODO:
		pass

	def is_start_region(self, states):
		dist_to_start = self.norm_func(states - self.start_state)
		return dist_to_start <= self.goal_tolerance
	
	def is_goal_region(self, states):
		dist_to_goal = self.norm_func(states - self.goal_state)
		return dist_to_goal <= self.goal_tolerance
	
	def extract_features_for_initiation_classifier(self, states):
		"""
		for antmaze, the features are the x, y coordinates (first 2 dimensions)
		"""
		return states[:, :2]

    # --------------------------------
    # Used for visualizations only
    # --------------------------------

	def _determine_x_y_lims(self):
		observations = self.env.get_dataset()["observations"]
		x = [obs[0] for obs in observations]
		y = [obs[1] for obs in observations]
		xlow, xhigh = min(x), max(x)
		ylow, yhigh = min(y), max(y)
		self.xlims = (xlow, xhigh)
		self.ylims = (ylow, yhigh)