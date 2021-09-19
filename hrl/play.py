import argparse
import logging
import pfrl
import os
from pathlib import Path

import numpy as np

from hrl import utils
from hrl.option.utils import get_player_position
from hrl.wrappers.monte_agent_space_wrapper import MonteAgentSpace
from hrl.wrappers.monte_agent_space_forwarding_wrapper import MonteAgentSpaceForwarding
from hrl.wrappers.monte_pruned_actions import MontePrunedActions


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
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		# system 
		parser.add_argument("--experiment_name", type=str, default='monte_info',
							help="Experiment Name, also used as the directory name to save results")
		parser.add_argument("--results_dir", type=str, default='resources',
							help='the name of the directory used to store results')
		# environments
		parser.add_argument("--environment", type=str, default='MontezumaRevengeNoFrameskip-v4',
							help="name of the gym environment")
		parser.add_argument("--agent_space", action='store_true', default=False,
							help="train with the agent space")
		parser.add_argument("--use_deepmind_wrappers", action='store_true', default=False,
							help="use the deepmind wrappers")
		parser.add_argument("--suppress_action_prunning", action='store_true', default=False,
							help='do not prune the action space of monte')
		parser.add_argument("--seed", type=int, default=0,
							help="Random seed")
		parser.add_argument("--render", action='store_true', default=False,
							help='render the environment as the game is played')
		parser.add_argument("--get_player_position", action='store_true', default=False,
							help="print out the agent's position at every state")
        # start state
		parser.add_argument("--goal_state_dir", type=Path, default="resources/monte_info",
							help="where the goal state files are stored")
		parser.add_argument("--start_state", type=str, default=None,
                            help='a path to the file that saved the starting state obs. e.g: right_ladder_top_agent_space.npy')
		parser.add_argument("--start_state_pos", type=str, default=None,
                            help='a path to the file that saved the starting state position. e.g: right_ladder_top_pos.txt')
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
		self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
		utils.create_log_dir(self.saving_dir)

		# make env
		self.env = self.make_env(self.params['environment'], self.params['seed'], render=self.params['render'])

	def play(self):
		"""
		play through the environment, with user-input actions
		"""
		print([(i, meaning) for i, meaning in enumerate(self.env.unwrapped.get_action_meanings())])
		state = self.env.reset()
		if self.params['get_player_position']:  # get position
			pos = get_player_position(self.env.unwrapped.ale.getRAM())
			print(f"current position is {pos}")
		while True:
			# user input an action to take
			action_input = input() 
			if action_input == 'save':
				if self.params['agent_space']:
					save_path = os.path.join(self.saving_dir, 'agent_space_goal_state.npy')
				else:
					save_path = os.path.join(self.saving_dir, 'goal_state.npy')
				np.save(file=save_path, arr=state)
				print(f'saved numpy array {state} of shape {state.shape} to {save_path}')
				break
			elif action_input == 'save_position':
				assert self.params['get_player_position']
				save_path = os.path.join(self.saving_dir, "goal_state_pos.txt")
				np.savetxt(fname=save_path, X=pos)
				print(f"saved numpy array {pos} to {save_path}")
				break
			else:
				action = int(action_input)

			# take the action
			next_state, r, done, info = self.env.step(action)
			try:
				assert next_state.shape == (56, 40, 3)
			except AssertionError:
				print(next_state.shape)
			print(f'taking action {action} and got reward {r}')
			state = next_state
			if self.params['get_player_position']:  # get position
				pos = get_player_position(self.env.unwrapped.ale.getRAM())
				print(f"current position is {pos}")
			if done:
				break

	def make_env(self, env_name, env_seed, render=True):
		env = pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60)  # 30 min with 60 fps
		if self.params['use_deepmind_wrappers']:
			env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
				env,
				episode_life=True,
				clip_rewards=True,
				flicker=False,
				frame_stack=False,
			)
		# prunning actions
		if not self.params['suppress_action_prunning']:
			env = MontePrunedActions(env)
		if self.params['agent_space']:
			env = MonteAgentSpace(env)
        # make the agent start in another place if needed
		if self.params['start_state'] is not None and self.params['start_state_pos'] is not None:
			start_state_path = self.params['goal_state_dir'].joinpath(self.params['start_state'])
			start_state_pos_path = self.params['goal_state_dir'].joinpath(self.params['start_state_pos'])
			env = MonteAgentSpaceForwarding(env, start_state_path, start_state_pos_path)
		if render:
			env = pfrl.wrappers.Render(env)
		logging.info(f'making environment {env_name}')
		env.seed(env_seed)
		return env


def main():
	game = PlayGame()
	game.play()


if __name__ == "__main__":
	main()