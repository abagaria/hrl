import time
import os
import random
import argparse
import shutil
import logging

import gym
import d4rl
import pfrl
from pfrl.agents import PPO
import torch
import seeding
import numpy as np

from hrl.wrappers import D4RLAntMazeWrapper, VectorEnvWrapper
from hrl import utils
from hrl.agent.dsc.dsc import RobustDSC
from hrl.envs import MultiprocessVectorEnv
from hrl.plot import main as plot_learning_curve
from hrl.models.sequential import SequentialModel
from hrl.models.utils import phi

logging.basicConfig(level=20)


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
        parser.add_argument("--experiment_name", type=str, default='test',
                            help="Experiment Name, also used as the directory name to save results")
        parser.add_argument("--results_dir", type=str, default='results',
                            help='the name of the directory used to store results')
        parser.add_argument("--device", type=str, default='cuda:1',
                            help="cpu/cuda:0/cuda:1")
        parser.add_argument("--generate_init_gif", action="store_true", default=False,
                            help='whether to generate initiation area gifs')
        # environments
        parser.add_argument("--environment", type=str, required=True, 
                            help="name of the gym environment")
        parser.add_argument('--agent', type=str, default='dsc', choices=['dsc', 'ppo'],
                            help='choose which agent to run')
        parser.add_argument("--seed", type=int, default=1,
                            help="Random seed")
        parser.add_argument("--goal_state", nargs="+", type=float, default=[],
                            help="specify the goal state of the environment, (0, 8) for example")
        parser.add_argument('--num_envs', type=int, default=8,
                            help='Number of env instances to run in parallel')
        # hyperparams
        parser.add_argument('--hyperparams', type=str, default='hyperparams/default.csv',
                            help='path to the hyperparams file to use')
        args, unknown = parser.parse_known_args()
        other_args = {
            (utils.remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        args.other_args = other_args
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
        seeding.seed(0, random, np)
        pfrl.utils.set_random_seed(self.params['seed'])

        # create the saving directories
        saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        if os.path.exists(saving_dir):  # remove all existing contents
            shutil.rmtree(saving_dir)
        utils.create_log_dir(saving_dir)
        utils.create_log_dir(os.path.join(saving_dir, "initiation_set_plots/"))
        utils.create_log_dir(os.path.join(saving_dir, "value_function_plots/"))
        self.saving_dir = saving_dir

        # save the hyperparams
        utils.save_hyperparams(os.path.join(saving_dir, "hyperparams.csv"), self.params)

        # set up env and experiment
        self.env = make_batch_env(self.params['environment'], self.params['num_envs'], self.params['seed'], self.params['goal_state'], self.params['use_dense_rewards'])
        if self.params['agent'] == 'dsc':
            self.exp = RobustDSC(mdp=self.env, params=self.params)
        elif self.params['agent'] == 'ppo':
            self.agent = self.make_agent()
    
    def make_agent(self):
        """
        create the agent
        """
        model = SequentialModel(obs_n_channels=self.env.observation_space.low.shape[0], n_actions=self.env.action_space.n).model
        opt = torch.optim.Adam(model.parameters(), lr=self.params['lr'], eps=1e-5)
        gpu = 0 if 'cuda' in self.params['device'] else -1
        agent = PPO(
            model,
            opt,
            gpu=gpu,
            phi=phi,
            update_interval=self.params['update_interval'],
            minibatch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            clip_eps=0.1,
            clip_eps_vf=None,
            standardize_advantages=True,
            entropy_coef=1e-2,
            recurrent=False,
            max_grad_norm=0.5,
        )
        return agent
    
    def train(self):
        """
        train an agent
        """
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            pfrl.experiments.LinearInterpolationHook(self.params['steps'], self.params['lr'], 0, lr_setter)
        )

        pfrl.experiments.train_agent_batch_with_evaluation(
            agent=self.agent,
            env=self.env,
            eval_env=make_batch_env(self.params['environment'], self.params['num_envs'], self.params['seed'], self.params['goal_state'], self.params['use_dense_rewards'], test=True),
            outdir=self.saving_dir,
            steps=self.params['steps'],
            eval_n_steps=None,
            eval_n_episodes=10,
            checkpoint_freq=None,
            eval_interval=self.params['evaluation_frequency'],
            log_interval=self.params['logging_frequency'],
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
            logger=logging.getLogger().setLevel(logging.INFO),
        )

    def run(self):
        """
        run the actual experiment
        """
        start_time = time.time()
        if self.params['agent'] == 'dsc':
            durations = self.exp.run_loop(self.params['episodes'], self.params['steps'])
        elif self.params['agent'] == 'ppo':
            self.train()
        else:
            raise NotImplementedError('specified agent not implemented')
        end_time = time.time()

        # plot the learning curve when experiemnt is done
        plot_learning_curve(self.params['experiment_name'])

        print("Time taken: ", end_time - start_time)


def make_env(env_name, env_seed, goal_state=None, use_dense_rewards=True, test=False):
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
    else:
        # can also make atari/gym environments
        env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
            pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60),  # 30 min with 60 fps
            episode_life=not test,
            clip_rewards=not test,
            flicker=False,
            frame_stack=False,
        )
     # seed the environment
    env_seed = 2 ** 32 - 1 - env_seed if test else env_seed
    env.seed(env_seed)
    return env


def make_batch_env(env_name, num_envs, base_seed, goal_state=None, use_dense_rewards=True, test=False):
    # get a batch of seeds
    process_seeds = np.arange(num_envs) + base_seed * num_envs
    assert process_seeds.max() < 2 ** 32
    # make vector env
    vec_env = MultiprocessVectorEnv(
        [
            (lambda: make_env(env_name, int(process_seeds[idx]), goal_state, use_dense_rewards, test))
            for idx, env in enumerate(range(num_envs))
        ]
    )
    # default to Frame Stacking
    # vec_env = VectorEnvWrapper(vec_env)
    vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
    return vec_env


def main():
    trial = Trial()
    trial.run()


if __name__ == "__main__":
    main()
