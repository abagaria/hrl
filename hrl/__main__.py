import time
import os
import random
import argparse

import gym
import d4rl
import pfrl
import torch
import seeding
import numpy as np

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl import utils
from hrl.agent.dsc.dsc import RobustDSC


class Trial:
    """
    a class for running experiments
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
        parser.add_argument("--experiment_name", type=str, 
                            help="Experiment Name")
        parser.add_argument("--results_dir", type=str, default='results',
                            help='the name of the directory used to store results')
        parser.add_argument("--device", type=str, 
                            help="cpu/cuda:0/cuda:1")
        parser.add_argument("--logging_frequency", type=int, default=50, help="Draw init sets, etc after every _ episodes")
        parser.add_argument("--generate_init_gif", action="store_true", default=False)
        parser.add_argument("--evaluation_frequency", type=int, default=10)
        # environments
        parser.add_argument("--environment", type=str, choices=["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"], 
                            help="name of the gym environment")
        parser.add_argument("--seed", type=int, default=0,
                            help="Random seed")
        parser.add_argument("--goal_state", nargs="+", type=float, default=[],
                            help="specify the goal state of the environment, (0, 8) for example")
        parser.add_argument('--num_envs', type=int, default=1,
                            help='Number of env instances to run in parallel')
        # learning specs
        parser.add_argument("--gestation_period", type=int, default=3)
        parser.add_argument("--buffer_length", type=int, default=50)
        parser.add_argument("--episodes", type=int, default=150)
        parser.add_argument("--steps", type=int, default=1000)
        parser.add_argument("--warmup_episodes", type=int, default=5)
        parser.add_argument("--use_value_function", action="store_true", default=False)
        parser.add_argument("--use_global_value_function", action="store_true", default=False)
        parser.add_argument("--use_model", action="store_true", default=False)
        parser.add_argument("--multithread_mpc", action="store_true", default=False)
        parser.add_argument("--use_diverse_starts", action="store_true", default=False)
        parser.add_argument("--use_dense_rewards", action="store_true", default=False)
        parser.add_argument("--use_global_option_subgoals", action="store_true", default=False)
        parser.add_argument("--clear_option_buffers", action="store_true", default=False)
        parser.add_argument("--lr_c", type=float, help="critic learning rate")
        parser.add_argument("--lr_a", type=float, help="actor learning rate")

        args = parser.parse_args()
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        assert self.params['use_model'] or self.params['use_value_function']

        if not self.params['use_value_function']:
            assert not self.params['use_global_value_function']

        if self.params['clear_option_buffers']:
            assert not self.params['use_global_value_function']
    
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
        utils.create_log_dir(os.path.join(saving_dir, "initiation_set_plots/"))
        utils.create_log_dir(os.path.join(saving_dir, "value_function_plots/"))

        # save the hyperparams
        utils.save_hyperparams(os.path.join(saving_dir, "hyperparams.csv"), self.params)

        # set up env and experiment
        self.env = make_batch_env(self.params['environment'], self.params['num_envs'], self.params['seed'], self.params['goal_state'], self.params['use_dense_rewards'])
        self.exp = RobustDSC(mdp=env,
                        gestation_period=args.gestation_period,
                        experiment_name=args.experiment_name,
                        device=torch.device(args.device),
                        warmup_episodes=args.warmup_episodes,
                        max_steps=args.steps,
                        use_model=args.use_model,
                        use_vf=args.use_value_function,
                        use_global_vf=args.use_global_value_function,
                        use_diverse_starts=args.use_diverse_starts,
                        use_dense_rewards=args.use_dense_rewards,
                        multithread_mpc=args.multithread_mpc,
                        logging_freq=args.logging_frequency,
                        evaluation_freq=args.evaluation_frequency,
                        buffer_length=args.buffer_length,
                        generate_init_gif=args.generate_init_gif,
                        seed=args.seed,
                        lr_c=args.lr_c,
                        lr_a=args.lr_a,
                        clear_option_buffers=args.clear_option_buffers,
                        use_global_option_subgoals=args.use_global_option_subgoals)

    def run(self):
        """
        run the actual experiment
        """
        start_time = time.time()
        durations = self.exp.run_loop(self.params['episodes'], self.params['steps'])
        end_time = time.time()

        print("Time taken: ", end_time - start_time)


def make_env(env_name, env_seed, goal_state=None, use_dense_rewards=True):
    if env_name in ["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"]:
        env = gym.make(env_name)
        # pick a goal state for the env
        if goal_state:
            goal_state = np.array(goal_state)
        else:
            # default to D4RL goal state
            goal_state = np.array(env.target_goal)
        print(f'using goal state {goal_state} in env {env_name}')
        env = D4RLAntMazeWrapper(env, start_state=((0, 0)), goal_state=goal_state, use_dense_reward=use_dense_rewards)
        # seed the environment
        env.seed(env_seed)
    else:
        raise NotImplementedError("Environment not supported!")
    return env


def make_batch_env(env_name, num_envs, base_seed, goal_state=None, use_dense_rewards=True):
    # get a batch of seeds
    process_seeds = np.arange(num_envs) + base_seed * num_envs
    assert process_seeds.max() < 2 ** 32
    # make vector env
    vec_env = pfrl.envs.MultiprocessVectorEnv(
        [
            (lambda: make_env(env_name, process_seeds[idx], goal_state))
            for idx, env in enumerate(range(num_envs))
        ]
    )
    # default to Frame Stacking
    vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
    return vec_env


def main():
    trial = Trial()
    trial.run()


if __name__ == "__main__":
    main()
