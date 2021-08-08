import argparse
import logging
import shutil
import pfrl
import os

import numpy as np

from hrl import utils


class PlayGame:
	"""
	use the class to step through a gym environment and play it with rendering view
	"""
	def __init__(self):
		args = self.parse_args()
		self.params = self.load_hyperparams(args)
		self.setup()
	
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

	def parse_args(self):
		"""
		parse the inputted argument
		"""
		parser = argparse.ArgumentParser()
		# system 
		parser.add_argument("--experiment_name", type=str, default='monte_info',
							help="Experiment Name, also used as the directory name to save results")
		parser.add_argument("--results_dir", type=str, default='results',
							help='the name of the directory used to store results')
		# environments
		parser.add_argument("--environment", type=str, default='MontezumaRevengeNoFrameskip-v4',
							help="name of the gym environment")
		parser.add_argument("--seed", type=int, default=0,
							help="Random seed")
		# hyperparams
		parser.add_argument('--hyperparams', type=str, default='hyperparams/monte.csv',
							help='path to the hyperparams file to use')
		args, unknown = parser.parse_known_args()
		other_args = {
			(utils.remove_prefix(key, '--'), val)
			for (key, val) in zip (unknown[::2], unknown[1::2])
		}
		args.other_args = other_args
		return args
	
	def setup(self):
		# setting random seeds
		pfrl.utils.set_random_seed(self.params['seed'])

		# saving
		saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
		if os.path.exists(saving_dir):  # remove all existing contents
			shutil.rmtree(saving_dir)
		utils.create_log_dir(saving_dir)
		self.saving_dir = saving_dir

		# make env
		self.env = make_env(self.params['environment'], self.params['seed'], render=True)

	def play(self):
		"""
		play through the environment, with user-input actions
		"""
		print(self.env.unwrapped.get_action_meanings())
		state = self.env.reset()
		while True:
			# user input an action to take
			action_input = input() 
			if action_input == 'save':
				action = 0  # NOOP
				save_path = os.path.join(self.saving_dir, 'goal_state.npy')
				np.save(file=save_path, arr=state)
			else:
				action = int(action_input)

			# take the action
			next_state, r, done, info = self.env.step(action)
			state = next_state
			if done:
				break


def make_env(env_name, env_seed, render=True):
	env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
		pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60),  # 30 min with 60 fps
		episode_life=True,
		clip_rewards=True,
		flicker=False,
		frame_stack=False,
	)
	logging.info(f'making environment {env_name}')
	env.seed(env_seed)
	if render:
		env = pfrl.wrappers.Render(env)
	return env


def main():
	game = PlayGame()
	game.play()


if __name__ == "__main__":
	main()
