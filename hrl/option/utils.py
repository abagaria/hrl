import os

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

cv2.ocl.setUseOpenCL(False)


def warp_frames(state):
	"""
	warp frames from (210, 160, 3) to (1, 84, 84) as in the nature paper
	this mimics the WarpFrame wrapper:
	https://github.com/pfnet/pfrl/blob/7b0c7e938ba2c0c56a941c766c68635d0dad43c8/pfrl/wrappers/atari_wrappers.py#L156
	"""
	size = (1, 84, 84)
	warped = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
	warped = cv2.resize(warped, (84, 84), interpolation=cv2.INTER_AREA)
	observation_space = gym.spaces.Box(
		low=0, high=255, shape=size, dtype=np.uint8
	)
	return warped.reshape(observation_space.low.shape)

def get_player_position(ram):
	"""
	given the ram state, get the position of the player
	"""
	def _getIndex(address):
		assert type(address) == str and len(address) == 2
		row, col = tuple(address)
		row = int(row, 16) - 8
		col = int(col, 16)
		return row * 16 + col
	def getByte(ram, address):
		# Return the byte at the specified emulator RAM location
		idx = _getIndex(address)
		return ram[idx]
	# return the player position at a particular state
	x = int(getByte(ram, 'aa'))
	y = int(getByte(ram, 'ab'))
	return x, y


def set_player_position(env, x, y):
	"""
	set the player position, specifically made for monte envs
	"""
	state_ref = env.unwrapped.ale.cloneState()
	state = env.unwrapped.ale.encodeState(state_ref)
	env.unwrapped.ale.deleteState(state_ref)

	state[331] = x
	state[335] = y

	new_state_ref = env.unwrapped.ale.decodeState(state)
	env.unwrapped.ale.restoreState(new_state_ref)
	env.unwrapped.ale.deleteState(new_state_ref)
	env.step(0)  # NO-OP action to update the RAM state


def make_chunked_value_function_plot(solver, step, seed, save_dir, pos_replay_buffer, chunk_size=1000):
	"""
	helper function to visualize the value function
	"""
	replay_buffer = solver.replay_buffer
	states = np.array([exp[0] for exp in replay_buffer])
	actions = np.array([exp[1] for exp in replay_buffer])

	# Chunk up the inputs so as to conserve GPU memory
	num_chunks = int(np.ceil(states.shape[0] / chunk_size))

	if num_chunks == 0:
		return 0.

	state_chunks = np.array_split(states, num_chunks, axis=0)
	action_chunks = np.array_split(actions, num_chunks, axis=0)
	qvalues = np.zeros((states.shape[0],))
	current_idx = 0

	for state_chunk, action_chunk in zip(state_chunks, action_chunks):
		state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
		action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
		chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
		current_chunk_size = len(state_chunk)
		qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
		current_idx += current_chunk_size

	x_pos = np.array([pos[0] for pos in pos_replay_buffer])
	y_pos = np.array([pos[1] for pos in pos_replay_buffer])
	try:
		plt.scatter(x_pos, y_pos, c=qvalues)
	except ValueError:
		num_points = min(len(x_pos), len(qvalues))
		x_pos = x_pos[:num_points]
		y_pos = y_pos[:num_points]
		qvalues = qvalues[:num_points]
		plt.scatter(x_pos, y_pos, c=qvalues)
	plt.xlim(0, 160)  # set the limits to the monte frame
	plt.ylim(145, 240)
	plt.colorbar()
	file_name = f"{solver.name}_value_function_seed_{seed}_step_{step}.png"
	save_path = os.path.join(save_dir, file_name)
	plt.savefig(save_path)
	plt.close()

	return qvalues.max()
