import logging
import time
import pickle
import os
import argparse
import shutil

import pfrl
import numpy as np

from hrl import utils
from hrl.option.option import Option
from hrl.plot import main as plot_learning_curve
from hrl.wrappers.monte_agent_space_wrapper import MonteAgentSpace


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
        parser.add_argument("--environment", type=str, default='MontezumaRevengeNoFrameskip-v4',
                            help="name of the gym environment")
        parser.add_argument("--render", action='store_true', default=False, 
                            help="save the images of states while training")
        parser.add_argument("--agent_space", action='store_true', default=False,
                            help="train with the agent space")
        parser.add_argument("--use_deepmind_wrappers", action='store_true', default=False,
                            help="use the deepmind wrappers")
        parser.add_argument("--seed", type=int, default=0,
                            help="Random seed")
        parser.add_argument("--goal_state_dir", type=str, default="resources/monte_info",
                            help="specify the goal state of the environment, (0, 8) for example")
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

        # create the saving directories
        self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        if os.path.exists(self.saving_dir):  # remove all existing contents
            shutil.rmtree(self.saving_dir)
        utils.create_log_dir(self.saving_dir)
        self.params['saving_dir'] = self.saving_dir

        # save the hyperparams
        utils.save_hyperparams(os.path.join(self.saving_dir, "hyperparams.csv"), self.params)

        # set up env and its goal
        self.env = self.make_env(self.params['environment'], self.params['seed'])
        if self.params['agent_space']:
            goal_state_path = os.path.join(self.params['goal_state_dir'], 'middle_ladder_bottom_agent_space.npy')
        else:
            goal_state_path = os.path.join(self.params['goal_state_dir'], 'middle_ladder_bottom.npy')
        goal_state_pos_path = os.path.join(self.params['goal_state_dir'], 'middle_ladder_bottom_pos.txt')
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
            # plot_two_class_classifier(self.option, self.option.num_executions, self.params['experiment_name'], plot_examples=True)

        end_time = time.time()

        # save the results
        success_curves_file_name = 'success_curves.pkl'
        self.save_results()
        plot_learning_curve(self.params['experiment_name'], log_file_name=success_curves_file_name)

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

    def make_env(self, env_name, env_seed):
        env = pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60)  # 30 min with 60 fps
        if self.params['use_deepmind_wrappers']:
            env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
                env,
                episode_life=True,
                clip_rewards=True,
                flicker=False,
                frame_stack=False,
            )
        if self.params['agent_space']:
            env = MonteAgentSpace(env)
            print('using the agent space to train the option right now')
        logging.info(f'making environment {env_name}')
        env.seed(env_seed)
        return env


def main():
    trial = TrainOptionTrial()
    trial.train_option()


if __name__ == "__main__":
    main()
