from copy import deepcopy

import numpy as np
import torch

from hrl.wrappers.gc_mdp_wrapper import GoalConditionedMDPWrapper


class D4RLAntMazeWrapper(GoalConditionedMDPWrapper):
	def __init__(self, env, start_state, goal_state, use_dense_reward=False):
		self.env = env
		self.norm_func = lambda x: np.linalg.norm(x, axis=-1) if isinstance(x, np.ndarray) else torch.norm(x, dim=-1)
		self.reward_func = self.dense_gc_reward_func if use_dense_reward else self.sparse_gc_reward_func
		self._determine_x_y_lims()
		super().__init__(env, start_state, goal_state)

	def state_space_size(self):
		return self.env.observation_space.shape[0]
	
	def action_space_size(self):
		return self.env.action_space.shape[0]
	
	def sparse_gc_reward_func(self, states, goals, batched=False):
		"""
		overwritting sparse gc reward function for antmaze
		"""
		# assert input is np array or torch tensor
		assert isinstance(states, (np.ndarray, torch.Tensor))
		assert isinstance(goals, (np.ndarray, torch.Tensor))

		if batched:
			current_positions = states[:,:2]
			goal_positions = goals[:,:2]
		else:
			current_positions = states[:2]
			goal_positions = goals[:2]
		distances = self.norm_func(current_positions-goal_positions)
		dones = distances <= self.goal_tolerance

		rewards = np.zeros_like(distances)
		rewards[dones==1] = +0.
		rewards[dones==0] = -1.

		return rewards, dones
	
	def dense_gc_reward_func(self, states, goals, batched=False):
		"""
		overwritting dense gc reward function for antmaze
		"""
		assert isinstance(states, (np.ndarray, torch.Tensor))
		assert isinstance(goals, (np.ndarray, torch.Tensor))

		if batched:
			current_positions = states[:,:2]
			goal_positions = goals[:,:2]
		else:
			current_positions = states[:2]
			goal_positions = goals[:2]
		distances = self.norm_func(current_positions - goal_positions)
		dones = distances <= self.goal_tolerance

		assert distances.shape == dones.shape == (states.shape[0], ) == (goals.shape[0], )

		rewards = -distances
		rewards[dones==True] = 0

		return rewards, dones
	
	def step(self, action):
		next_state, reward, done, info = self.env.step(action)
		reward, done = self.reward_func(next_state, self.get_current_goal())
		self.cur_state = next_state
		self.cur_done = done
		return next_state, reward, done, info

	def get_current_goal(self):
		return self.get_position(self.goal_state)

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
		assert isinstance(states, np.ndarray)
		features = states
		if "push" in self.unwrapped.spec.id:
			return features[:4]
		return features[:2]
	
	def set_xy(self, position):
		""" Used at test-time only. """
		position = tuple(position)  # `maze_model.py` expects a tuple
		self.env.env.set_xy(position)
		obs = np.concatenate((np.array(position), self.init_state[2:]), axis=0)
		self.cur_state = obs
		self.cur_done = False
		self.init_state = deepcopy(self.cur_state)

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

	def get_x_y_low_lims(self):
		return self.xlims[0], self.ylims[0]

	def get_x_y_high_lims(self):
		return self.xlims[1], self.ylims[1]
	
    # ---------------------------------
    # Used during testing only
    # ---------------------------------

	def sample_random_state(self, cond=lambda x: True):
		num_tries = 0
		rejected = True
		while rejected and num_tries < 200:
			low = np.array((self.xlims[0], self.ylims[0]))
			high = np.array((self.xlims[1], self.ylims[1]))
			sampled_point = np.random.uniform(low=low, high=high)
			rejected = self.env.env.wrapped_env._is_in_collision(sampled_point) or not cond(sampled_point)
			num_tries += 1

			if not rejected:
				return sampled_point
	
	@staticmethod
	def get_position(state):
		"""
		position in the antmaze is the x, y coordinates
		"""
		return state[:2]
