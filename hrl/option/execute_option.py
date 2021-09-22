import pickle
import random
import argparse
import os
import shutil
from pathlib import Path

import pfrl
import torch
import seeding
import numpy as np

from hrl import utils
from hrl.option.utils import make_env


class ExecuteOptionTrial:
    """
	a class for running experiments to train an option
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
        parser.add_argument(
            "--experiment_name",
            type=str,
            default='monte_execute_option',
            help="Experiment Name, also used as the directory name to save results")
        parser.add_argument(
            "--results_dir",
            type=str,
            default='results',
            help='the name of the directory used to store results')
        parser.add_argument("--device",
                            type=str,
                            default='cuda:1',
                            help="cpu/cuda:0/cuda:1")
        # environments
        parser.add_argument("--environment",
                            type=str,
                            default='MontezumaRevenge-v0',
                            help="name of the gym environment")
        parser.add_argument("--agent_space",
                            action='store_true',
                            default=True,
                            help="train with the agent space")
        parser.add_argument("--use_deepmind_wrappers",
                            action='store_true',
                            default=False,
                            help="use the deepmind wrappers")
        parser.add_argument("--suppress_action_prunning", action='store_true', default=False,
                            help='do not prune the action space of monte')
        parser.add_argument("--render", action='store_true', default=False, 
                            help="render the environment as it goes")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--saved_option",
            type=str,
            default='results/monte-agent-space/trained_option.pkl',
            help='path the a stored trained option')
        # start state
        parser.add_argument(
            "--info_dir",
            type=Path,
            default="resources/monte_info",
            help="the directory where monte state info is saved",
        )
        parser.add_argument(
            "--start_state",
            type=str,
            default=None,
            help='a path to the file that saved the starting state obs, e.g. right_ladder_top_agent_space.npy'
        )
        parser.add_argument(
            "--start_state_pos",
            type=str,
            default=None,
            help='a path to the file that saved the starting state position, e.g. right_ladder_top_pos.txt'
        )
        # hyperparams
        parser.add_argument('--hyperparams',
                            type=str,
                            default='hyperparams/monte.csv',
                            help='path to the hyperparams file to use')
        args, unknown = parser.parse_known_args()
        other_args = {(utils.remove_prefix(key, '--'), val)
                      for (key, val) in zip(unknown[::2], unknown[1::2])}
        args.other_args = other_args
        return args

    def setup(self):
        """
		do set up for the experiment
		"""
        # setting random seeds
        seeding.seed(0, random, torch, np)
        pfrl.utils.set_random_seed(self.params['seed'])

        # set up env and the forwarding target
        self.env = make_env(self.params['environment'], self.params['seed'])

        # create the saving directories
        self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        if os.path.exists(self.saving_dir):  # remove all existing contents
            shutil.rmtree(self.saving_dir)
        utils.create_log_dir(self.saving_dir)
        self.params['saving_dir'] = self.saving_dir

        # setup global option and the only option that needs to be learned
        with open(self.params['saved_option'], 'rb') as f:
            self.option = pickle.load(f)
            self.option.params = self.params
            self.option.env = self.env

    def exec_option(self):
        """
		run the actual experiment to train one option
		"""
        step_number = 0
        for _ in range(5):
            print(f"step {step_number}")
            option_transitions, total_reward = self.option.rollout(
                step_number=step_number, eval_mode=True, rendering=self.params['render']
            )
            step_number += len(option_transitions)


def main():
    trial = ExecuteOptionTrial()
    trial.exec_option()


if __name__ == "__main__":
    main()
