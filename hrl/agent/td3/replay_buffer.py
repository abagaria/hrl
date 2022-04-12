import time
import torch
import pickle
import ipdb
import numpy as np


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device("cuda")):
		self.max_size = max_size
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))
		self.info = [None] * max_size

		self.device = device

	def add(self, state, action, reward, next_state, done, info=None):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done
		self.info[self.ptr] = info

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)

	def __len__(self):
		return self.size

	def __getitem__(self, i):
		if i < self.size:
			return self.state[i], self.action[i], self.reward[i], self.next_state[i], self.done[i]
		raise IndexError(f"Tried to access index {i} when length is {self.size}")

	def clear(self):
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, self.state_dim))
		self.action = np.zeros((self.max_size, self.action_dim))
		self.next_state = np.zeros((self.max_size, self.state_dim))
		self.reward = np.zeros((self.max_size, 1))
		self.done = np.zeros((self.max_size, 1))
		self.info = [None] * self.max_size

	def save(self, filename):
		t0 = time.time()
		save_dict = dict(
			state=self.state[:self.size],
			action=self.action[:self.size],
			reward=self.reward[:self.size],
			next_state=self.next_state[:self.size],
			done=self.done[:self.size]
		)
		with open(filename, "wb+") as f:
			pickle.dump(save_dict, f)
		print(f"Took {time.time()-t0}s to save replay buffer")


class DistanceLearnerReplayBuffer(object):
	def __init__(self, state_dim, max_size=int(1e6), device=torch.device("cuda")):
		self.max_size = max_size
		self.state_dim = state_dim

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.episode_end = np.zeros((max_size, 1)) # index of last state in episode

		self.device = device

	def add(self, states):
		assert self.states.shape[0] < self.max_size, "replay buffer is too small to fit episode"

		episode_end = (self.ptr + self.states.shape[0]) % self.max_size
		for state in states:
			self.state[self.ptr] = state
			self.episode_end[self.ptr] = episode_end
			
			self.ptr = (self.ptr + 1) % self.max_size
			self.size = min(self.size + 1, self.max_size)

	def sample_pairs(self, batch_size):
		ipdb.set_trace()
		start_ind = np.random.randint(0, self.size, size=batch_size)
		end_range = self.episode_end[start_ind]
		end_range[end_range < start_ind] += self.size

		end_ind = np.random.uniform(start_ind, end_range)
		dist = end_ind - start_ind
		end_ind = end_ind % self.size
		state_pairs = np.concatenate((self.state[start_ind], self.state[end_ind]), axis=1)

		return (
			torch.FloatTensor(state_pairs).to(self.device),
			torch.IntTensor(dist)
		)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return torch.FloatTensor(self.state[ind]).to(self.device)

	def __len__(self):
		return self.size

	def __getitem__(self, i):
		if i < self.size:
			return self.state[i], self.episode_end[i]
		raise IndexError(f"Tried to access index {i} when length is {self.size}")
