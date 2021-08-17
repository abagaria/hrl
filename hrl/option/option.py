import random
import itertools
from copy import deepcopy
from functools import reduce
from collections import deque

import numpy as np
from thundersvm import SVC, OneClassSVM

from hrl.agent.td3.TD3AgentClass import TD3
from hrl.option.utils import warp_frames, get_player_position, make_chunked_value_function_plot


class Option:
	"""
	the base class for option that all Option class shoud inherit from
	"""
	def __init__(self, name, env, params):
		self.name = name
		self.env = env
		self.params = params

		self.initiation_classifier = None
		self.termination_classifier = None
		self.initiation_positive_examples = []
		self.initiation_negative_examples = []
		self.termination_positive_examples = []
		self.termination_negative_examples = []

		self.success_curve = []
		self.success_rates = {}

		self.gestation_period = params['gestation_period']
		self.num_goal_hits = 0
		self.num_executions = 0

		# used to store trajectories of positions and finally used for plotting value function
		self.position_buffer = deque([], maxlen=self.params['buffer_length'])

		# learner for the value function 
		# don't use output normalization because this is the model free case
		self.value_learner = TD3(state_dim=reduce(lambda x, y: x*y, self.env.observation_space.shape, 1),
								action_dim=self.env.action_space.n,
								max_action=1.,
								name=f"{name}-td3-agent",
								device=self.params['device'],
								lr_c=self.params['lr_c'], 
								lr_a=self.params['lr_a'],
								buffer_length=self.params['buffer_length'],
								use_output_normalization=False)

	# ------------------------------------------------------------
	# Learning Phase Methods
	# ------------------------------------------------------------

	def get_training_phase(self):
		"""
		determine the training phase, which could only be one of two
		"""
		if self.num_goal_hits < self.gestation_period:
			return "gestation"
		return "initiation_done"

	def is_init_true(self, state):
		"""
		whether the initaition condition is true
		"""
		# initation is always true for an option during training
		if self.get_training_phase() == "gestation":
			return True
		
		# initiation is true if we are at the start state
		copied_env = deepcopy(self.env)
		if state == copied_env.reset():
			return True

		return self.initiation_classifier.predict([state])[0] == 1

	def is_term_true(self, state):
		"""
		whether the termination condition is true
		"""
		# termination is always true if the state is near the goal
		position = get_player_position(self.env.unwrapped.ale.getRAM())
		distance_to_goal = np.linalg.norm(position - self.params['goal_state_position'])
		if distance_to_goal < self.params['epsilon_within_goal']:
			return True
		# if termination classifier isn't initialized, and state is not goal state
		if self.termination_classifier is None:
			return False
		return self.termination_classifier.predict([state])[0] == 1

	# ------------------------------------------------------------
	# Control Loop Methods
	# ------------------------------------------------------------

	def act(self, state):
		"""
		return an action for the specified state according to an epsilon greedy policy
		"""
		if random.random() < self.params['epsilon']:
			return self.env.action_space.sample()
		else:
			# the action selector is the same as the value learner
			tanh_output = self.value_learner.act(state, evaluation_mode=False)
			action = np.argmax(tanh_output)
			return action
	
	def rollout(self, step_number, eval_mode=False):
		"""
		main control loop for option execution
		"""
		# reset env
		state = self.env.reset()
		self.env.unwrapped.reset()
		done = False

		if self.params['use_deepmind_wrappers']:
			state = warp_frames(state)
		assert self.is_init_true(state)

		num_steps = 0
		total_reward = 0
		visited_states = []
		option_transitions = []
		goal = self.params['goal_state']

		assert goal.shape == state.shape

		# print(f"[Step: {step_number}] Rolling out {self.name}, from {state} targeting {goal}")

		self.num_executions += 1

		# main while loop
		while not self.is_term_true(state) and not done:
			# control
			action = self.act(state.flatten())
			next_state, reward, done, _ = self.env.step(action)
			# logging
			num_steps += 1
			step_number += 1
			total_reward += reward
			visited_states.append(state.flatten())
			option_transitions.append((state.flatten(), action, reward, next_state.flatten(), done))
			state_pos = get_player_position(self.env.unwrapped.ale.getRAM())
			self.position_buffer.append(state_pos)
			state = next_state
			if step_number % self.params['logging_frequency'] == 0:
				make_chunked_value_function_plot(self.value_learner, step_number, self.params['seed'], self.params['saving_dir'], pos_replay_buffer=self.position_buffer)
		visited_states.append(state.flatten())

		# more logging
		self.success_curve.append(self.is_term_true(state))
		self.success_rates[step_number] = {'success': self.get_success_rate()}
		if self.is_term_true(state):
			self.num_goal_hits += 1
			print(f"num goal hits increased to {self.num_goal_hits}")
		
		# training
		if not eval_mode:
			self.derive_positive_and_negative_examples(visited_states)
			# this is updating the value function
			self.experience_replay(option_transitions)
			# refining your initiation/termination classifier
			self.fit_classifier(self.initiation_classifier, self.initiation_positive_examples, self.initiation_negative_examples)
			self.fit_classifier(self.termination_classifier, self.termination_positive_examples, self.termination_negative_examples)
		
		return option_transitions, total_reward

	def experience_replay(self, trajectory):
		for state, action, reward, next_state, done in trajectory:
			done = self.is_term_true(next_state)
			self.value_learner.step(state, action, reward, next_state, done)

	# ------------------------------------------------------------
	# Classifiers
	# ------------------------------------------------------------
	def derive_positive_and_negative_examples(self, visited_states):
		"""
		derive positive and negative examples used to train classifiers
		"""
		start_state = visited_states[0]
		final_state = visited_states[-1]

		if self.is_term_true(final_state):
			# positive
			positive_states = [start_state] + visited_states[-self.params['buffer_length']:]
			self.initiation_positive_examples.append(positive_states)
			self.termination_positive_examples.append(final_state)
		else:
			negative_examples = [start_state]
			self.initiation_negative_examples.append(negative_examples)
			self.termination_negative_examples.append(final_state)

	def construct_feature_matrix(self, examples):
		states = list(itertools.chain.from_iterable(examples))
		states = np.array(states).reshape(len(states), -1)  # reshape to (batch_size, state_size)
		return np.array(states)
	
	def fit_classifier(self, classifier, positive_examples, negative_examples):
		"""
		fit the initiation/termination classifier using positive and negative examples
		"""
		assert classifier is self.initiation_classifier or self.termination_classifier
		if len(negative_examples) > 0 and len(positive_examples) > 0:
			self.train_two_class_classifier(classifier, positive_examples, negative_examples)
		elif len(positive_examples) > 0:
			self.train_one_class_svm(classifier, positive_examples)

	def train_one_class_svm(self, classifier, positive_examples, nu=0.1):
		positive_feature_matrix = self.construct_feature_matrix(positive_examples)
		classifier = OneClassSVM(kernel="rbf", nu=nu)  # or nu=nu/10. for pessimestic
		classifier.fit(positive_feature_matrix)

	def train_two_class_classifier(self, classifier, positive_examples, negative_examples, nu=0.1):
		positive_feature_matrix = self.construct_feature_matrix(positive_examples)
		negative_feature_matrix = self.construct_feature_matrix(negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		if negative_feature_matrix.shape[0] >= 10:
			kwargs = {"kernel": "rbf", "gamma": "auto", "class_weight": "balanced"}
		else:
			kwargs = {"kernel": "rbf", "gamma": "auto"}

		classifier = SVC(**kwargs)
		classifier.fit(X, Y)
	
	# ------------------------------------------------------------
	# testing
	# ------------------------------------------------------------

	def get_success_rate(self):
		if len(self.success_curve) == 0:
			return 0.
		return np.mean(self.success_curve)
