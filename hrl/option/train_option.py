import logging
import time
import os
import random
import argparse

import gym
import pfrl
import torch
import seeding
import numpy as np

from hrl import utils
from hrl.option import ModelBasedOption


class TrainOptionTrial:
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
        parser = argparse.ArgumentParser()
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
        parser.add_argument("--seed", type=int, default=0,
                            help="Random seed")
        parser.add_argument("--goal_state", nargs="+", type=float, required=True,
                            help="specify the goal state of the environment, (0, 8) for example")
        # hyperparams
        parser.add_argument('--hyperparams', type=str, default='hyperparams/default.csv',
                            help='path to the hyperparams file to use')
        args, unknown = parser.parse_known_args()
        other_args = {
            (utils.remove_prefix(key, '--'), val)
            for (key, val) in zip (unknown[::2], unknown[1::2])
        }
        args.other_args = other_args
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        pass
    
    def setup(self):
        """
        do set up for the experiment
        """
        self.check_params_validity()

        # setting random seeds
        seeding.seed(0, random, torch, np)
        pfrl.utils.set_random_seed(self.params['seed'])

        # create the saving directories
        saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        utils.create_log_dir(saving_dir)

        # save the hyperparams
        utils.save_hyperparams(os.path.join(saving_dir, "hyperparams.csv"), self.params)

        # set up env
        self.env = make_env(self.params['environment'], self.params['seed'])

        # setup target salient event
        self.target_salient_event = None

        # setup global option and the only option that needs to be learned
        self.global_option = self.create_global_model_based_option()
        self.option = self.create_model_based_option(name='skill', parent=None)

    def create_model_based_option(self, name, parent=None):
        """
        create a model based option
        """
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelBasedOption(parent=parent, 
                                mdp=self.env,
                                buffer_length=self.params['buffer_length'],
                                global_init=False,
                                gestation_period=self.params['gestation_period'],
                                timeout=200,
                                max_steps=self.params['steps'],
                                device=torch.device(self.params['device']),
                                target_salient_event=self.target_salient_event,  # TODO: define this
                                name=name,
                                path_to_model="",
                                global_solver=self.global_option.solver,
                                use_vf=self.params['use_value_function'],
                                use_global_vf=self.params['use_global_value_function'],
                                use_model=self.params['use_model'],
                                dense_reward=self.params['use_dense_rewards'],
                                global_value_learner=self.global_option.value_learner,
                                option_idx=option_idx,
                                lr_c=self.params['lr_c'], 
                                lr_a=self.params['lr_a'],
                                multithread_mpc=self.params['multithread_mpc'])
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None,
                                  mdp=self.env,
                                  buffer_length=self.params['buffer_length'],
                                  global_init=True,
                                  gestation_period=self.params['gestation_period'],
                                  timeout=200, 
                                  max_steps=self.params['steps'],
                                  device=torch.device(self.params['device']),
                                  target_salient_event=self.target_salient_event,  # TODO: define this
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.params['use_value_function'],
                                  use_global_vf=self.params['use_global_value_function'],
                                  use_model=self.params['use_model'],
                                  dense_reward=self.params['use_dense_rewards'],
                                  global_value_learner=None,
                                  option_idx=0,
                                  lr_c=self.params['lr_c'],
                                  lr_a=self.params['lr_a'],
                                  multithread_mpc=self.params['multithread_mpc'])
        return option

    def train_option(self):
        """
        run the actual experiment to train one option
        """
        start_time = time.time()
        
        # create the option
        while self.option.get_training_phase() == "gestation":
            option_transitions, total_reward = self.option.rollout(step_number=self.params['steps'], rollout_goal=self.params['goal_state'], eval_mode=False)

        end_time = time.time()

        print("Time taken: ", end_time - start_time)


def make_env(env_name, env_seed):
	env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
		pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60),  # 30 min with 60 fps
		episode_life=True,
		clip_rewards=True,
		flicker=False,
		frame_stack=True,
	)
	logging.info(f'making environment {env_name}')
	env.seed(env_seed)
	return env


def main():
    trial = TrainOptionTrial()
    trial.train_option()


if __name__ == "__main__":
    main()
