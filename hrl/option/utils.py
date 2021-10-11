import argparse
import os
import logging
from pathlib import Path

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import pfrl

from hrl import utils

cv2.ocl.setUseOpenCL(False)


class SingleOptionTrial:
	"""
	a base class for every class that deals with training/executing a single option
	"""
	def __init__(self):
		pass

	def get_common_arg_parser(self):
		parser = argparse.ArgumentParser(
			formatter_class=argparse.ArgumentDefaultsHelpFormatter,
			add_help=False,
		)
		# common args
        # system 
		parser.add_argument("--experiment_name", type=str, default='monte',
							help="Experiment Name, also used as the directory name to save results")
		parser.add_argument("--results_dir", type=str, default='results',
							help='the name of the directory used to store results')
		parser.add_argument("--device", type=str, default='cuda:1',
							help="cpu/cuda:0/cuda:1")
		# environments
		parser.add_argument("--environment", type=str, default='MontezumaRevengeNoFrameskip-v4',
							help="name of the gym environment")
		parser.add_argument("--render", action='store_true', default=False, 
							help="save the images of states while training")
		parser.add_argument("--agent_space", action='store_true', default=False,
							help="train with the agent space")
		parser.add_argument("--down_sample", "-d", action='store_true', default=False, 
							help="down sample the monte frame by converting to grey scale and divide by 255")
		parser.add_argument("--use_deepmind_wrappers", action='store_true', default=True,
							help="use the deepmind wrappers")
		parser.add_argument("--suppress_action_prunning", action='store_true', default=False,
							help='do not prune the action space of monte')
		parser.add_argument("--seed", type=int, default=0,
							help="Random seed")
		# start state
		parser.add_argument("--info_dir", type=Path, default="resources/monte_info",
							help="where the goal state files are stored")

		parser.add_argument("--start_state", type=str, default=None,
							help='a path to the file that saved the starting state obs. e.g: right_ladder_top_agent_space.npy')
		parser.add_argument("--start_state_pos", type=str, default=None,
							help='a path to the file that saved the starting state position. e.g: right_ladder_top_pos.txt')
		# hyperparams
		parser.add_argument('--hyperparams', type=str, default='hyperparams/monte.csv',
							help='path to the hyperparams file to use')
		return parser

	def parse_common_args(self, parser):
		args, unknown = parser.parse_known_args()
		other_args = {
			(utils.remove_prefix(key, '--'), val)
			for (key, val) in zip(unknown[::2], unknown[1::2])
		}
		args.other_args = other_args
		return args

	def load_hyperparams(self, args):
		"""
		load the hyper params from args to a params dictionary
		"""
		params = utils.load_hyperparams(args.hyperparams)
		for arg_name, arg_value in vars(args).items():
			if arg_name == 'hyperparams':
				continue
			params[arg_name] = arg_value
		for arg_name, arg_value in args.other_args:
			utils.update_param(params, arg_name, arg_value)
		return params

	def make_env(self, env_name, env_seed):
		from hrl.wrappers.monte_agent_space_wrapper import MonteAgentSpace
		from hrl.wrappers.monte_agent_space_forwarding_wrapper import MonteAgentSpaceForwarding
		from hrl.wrappers.monte_pruned_actions import MontePrunedActions
		from hrl.wrappers.monte_dm_agent_space import MonteDeepMindAgentSpace

		if self.params['use_deepmind_wrappers']:
			env = pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60)  # 30 min with 60 fps
			env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
				env,
				episode_life=True,
				clip_rewards=True,
				frame_stack=True,
				scale=False,
				fire_reset=False,
				channel_order="chw",
				flicker=False,
			)
			# make agent space
			if self.params['agent_space']:
				env = MonteDeepMindAgentSpace(env)
				print('using the agent space to train the optin right now')
		else:
			env = gym.make(env_name)
			# make agent space
			if self.params['agent_space']:
				env = MonteAgentSpace(env)
				print('using the agent space to train the option right now')
		# prunning actions
		if not self.params['suppress_action_prunning']:
			env = MontePrunedActions(env)
		# make the agent start in another place if needed
		if self.params['start_state'] is not None and self.params['start_state_pos'] is not None:
			start_state_path = self.params['info_dir'].joinpath(self.params['start_state'])
			start_state_pos_path = self.params['info_dir'].joinpath(self.params['start_state_pos'])
			env = MonteAgentSpaceForwarding(env, start_state_path, start_state_pos_path)
		# down sacle observations
		if self.params['down_sample']:
			pass
		logging.info(f'making environment {env_name}')
		env.seed(env_seed)
		env.action_space.seed(env_seed)
		return env


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


def make_done_state_plot(replay_buffer, episode_idx, save_dir):
	"""
	visualize the done positions
	"""
	if not replay_buffer:
		# if replay buffer is empty
		return
	states = np.array([exp[0] for exp in replay_buffer])
	dones = np.array([exp[-1] for exp in replay_buffer])
	done_states = states[dones]

	for i in range(len(done_states)):
		s = done_states[i]
		frame = s[:24*24].reshape((24, 24))  # states in replay buffer are flattened, reshape to monte frame
		file_name = save_dir.joinpath(f"done_state_plot_at_episode_{episode_idx}__{i}.jpg")
		try:
			plt.imsave(file_name, frame)
		except ValueError:
			# cannot plot image with channel 1
			plt.imsave(file_name, frame.squeeze())


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
