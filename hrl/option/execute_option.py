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
from hrl.option.utils import SingleOptionTrial


class ExecuteOptionTrial(SingleOptionTrial):
    """
	a class for running experiments to train an option
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
        parser.add_argument(
            "--saved_option",
            type=str,
            default='results/monte-agent-space/trained_option.pkl',
            help='path the a stored trained option')
        args = self.parse_common_args(parser)
        return args

    def setup(self):
        """
		do set up for the experiment
		"""
        # setting random seeds
        seeding.seed(0, random, torch, np)
        pfrl.utils.set_random_seed(self.params['seed'])

        # set up env and the forwarding target
        self.env = self.make_env(self.params['environment'], self.params['seed'])

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
