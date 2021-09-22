import time
import pickle
import os
import argparse
import shutil
from pathlib import Path

import pfrl
import torch
import numpy as np

from hrl import utils
from hrl.option.utils import make_env
from hrl.option.option import Option
from hrl.plot import main as plot_learning_curve


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
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # system 
        parser.add_argument("--experiment_name", type=str, default='monte',
                            help="Experiment Name, also used as the directory name to save results")
        parser.add_argument("--results_dir", type=str, default='results',
                            help='the name of the directory used to store results')
        parser.add_argument("--device", type=str, default='cuda:1',
                            help="cpu/cuda:0/cuda:1")
        # environments
        parser.add_argument("--environment", type=str, default='MontezumaRevenge-v0',
                            help="name of the gym environment")
        parser.add_argument("--render", action='store_true', default=False, 
                            help="save the images of states while training")
        parser.add_argument("--agent_space", action='store_true', default=False,
                            help="train with the agent space")
        parser.add_argument("--use_deepmind_wrappers", action='store_true', default=False,
                            help="use the deepmind wrappers")
        parser.add_argument("--suppress_action_prunning", action='store_true', default=False,
                            help='do not prune the action space of monte')
        parser.add_argument("--seed", type=int, default=0,
                            help="Random seed")
        # start state
        parser.add_argument("--start_state", type=str, default=None,
                            help='a path to the file that saved the starting state obs. e.g: right_ladder_top_agent_space.npy')
        parser.add_argument("--start_state_pos", type=str, default=None,
                            help='a path to the file that saved the starting state position. e.g: right_ladder_top_pos.txt')
        # goal state
        parser.add_argument("--goal_state_dir", type=Path, default="resources/monte_info",
                            help="where the goal state files are stored")
        parser.add_argument("--goal_state", type=str, default="middle_ladder_bottom.npy",
                            help="a file in goal_state_dir that stores the image of the agent in goal state")
        parser.add_argument("--goal_state_agent_space", type=str, default="middle_ladder_bottom_agent_space.npy",
                            help="a file in goal_state_dir that store the agent space image of agent in goal state")
        parser.add_argument("--goal_state_pos", type=str, default="middle_ladder_bottom_pos.txt",
                            help="a file in goal_state_dir that store the x, y coordinates of goal state")
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
        pfrl.utils.set_random_seed(self.params['seed'])

        # torch benchmark
        torch.backends.cudnn.benchmark = True

        # create the saving directories
        self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        if os.path.exists(self.saving_dir):  # remove all existing contents
            shutil.rmtree(self.saving_dir)
        utils.create_log_dir(self.saving_dir)
        self.params['saving_dir'] = self.saving_dir

        # save the hyperparams
        utils.save_hyperparams(os.path.join(self.saving_dir, "hyperparams.csv"), self.params)

        # set up env and its goal
        self.env = make_env(self.params['environment'], self.params['seed'])
        if self.params['agent_space']:
            goal_state_path = self.params['goal_state_dir'].joinpath(self.params['goal_state_agent_space'])
        else:
            goal_state_path = self.params['goal_state_dir'].joinpath(self.params['goal_state'])
        goal_state_pos_path = self.params['goal_state_dir'].joinpath(self.params['goal_state_pos'])
        self.params['goal_state'] = np.load(goal_state_path)
        self.params['goal_state_position'] = np.loadtxt(goal_state_pos_path)
        print(f"aiming for goal location {self.params['goal_state_position']}")

        # setup global option and the only option that needs to be learned
        self.option = Option(name='only-option', env=self.env, params=self.params)

    def train_option(self):
        """
        run the actual experiment to train one option
        """
        start_time = time.time()
        
        # create the option
        step_number = 0
        episode_idx = 0
        while self.option.get_training_phase() == "gestation":
            print(f"starting episode {episode_idx} at step {step_number}")
            option_transitions, total_reward = self.option.rollout(step_number=step_number, eval_mode=False, rendering=self.params['render'])
            step_number += len(option_transitions)
            episode_idx += 1

            # save the results
            if episode_idx % self.params['saving_frequency'] == 0:
                success_curves_file_name = 'success_curves.pkl'
                self.save_results()
                plot_learning_curve(self.params['experiment_name'], log_file_name=success_curves_file_name)

            # plot_two_class_classifier(self.option, self.option.num_executions, self.params['experiment_name'], plot_examples=True)

        end_time = time.time()

        print("Time taken: ", end_time - start_time)
    
    def save_results(self, success_curves_file_name='success_curves.pkl', option_file_name='trained_option.pkl'):
        """
        save the results into csv files
        """
        # save success curve
        success_curve_save_path = os.path.join(self.saving_dir, success_curves_file_name)
        with open(success_curve_save_path, 'wb+') as f:
            pickle.dump(self.option.success_rates, f)
        # save trained option
        option_save_path = os.path.join(self.saving_dir, option_file_name)
        with open(option_save_path, 'wb') as f:
            pickle.dump(self.option, f)


def main():
    trial = TrainOptionTrial()
    trial.train_option()


if __name__ == "__main__":
    main()
