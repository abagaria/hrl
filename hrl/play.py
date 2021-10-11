import argparse
import os

import seeding
import numpy as np
import pfrl

from hrl import utils
from hrl.option.utils import get_player_position, SingleOptionTrial


class PlayGame(SingleOptionTrial):
	"""
	use the class to step through a gym environment and play it with rendering view
	"""
	def __init__(self):
		super().__init__()
		args = self.parse_args()
		self.params = self.load_hyperparams(args)
		self.setup()

	def parse_args(self):
		"""
		parse the inputted argument
		"""
		parser = argparse.ArgumentParser(
			formatter_class=argparse.ArgumentDefaultsHelpFormatter,
			parents=[self.get_common_arg_parser()]
		)
		parser.add_argument("--get_player_position", action='store_true', default=False,
							help="print out the agent's position at every state")
		args = self.parse_common_args(parser)
		return args
	
	def setup(self):
		# setting random seeds
		seeding.seed(self.params['seed'], np)

		# saving
		self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
		utils.create_log_dir(self.saving_dir)

		# make env
		self.env = self.make_env(self.params['environment'], self.params['seed'])

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
			# render env
			self.env.unwrapped.render()
			print(f"state shape is {state.shape}")
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
			print(f'taking action {action} and got reward {r}')
			state = next_state
			if self.params['get_player_position']:  # get position
				pos = get_player_position(self.env.unwrapped.ale.getRAM())
				print(f"current position is {pos}")
			if done:
				break


def main():
	game = PlayGame()
	game.play()


if __name__ == "__main__":
	main()
