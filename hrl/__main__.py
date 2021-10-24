import time
import os
import math
import argparse
import shutil
import logging
from pathlib import Path

import gym
import d4rl
import pfrl
import seeding
import numpy as np

from hrl.wrappers import D4RLAntMazeWrapper
from hrl.train_loop import train_agent_batch_with_eval
from hrl import utils
from hrl.agent.dsc.dsc import RobustDSC
from hrl.agent.make_agent import make_sac_agent, make_td3_agent
from hrl.envs.vector_env import EpisodicSyncVectorEnv
from hrl.plot import main as plot_learning_curve


logging.basicConfig(level=logging.INFO)


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
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # system 
        parser.add_argument("--experiment_name", type=str, default='test',
                            help="Experiment Name, also used as the directory name to save results")
        parser.add_argument("--results_dir", type=str, default='results',
                            help='the name of the directory used to store results')
        parser.add_argument('--disable_gpu', default=False, action='store_true',
                            help='enforce training on CPU')
        parser.add_argument("--generate_init_gif", action="store_true", default=False,
                            help='whether to generate initiation area gifs')
        # environments
        parser.add_argument("--environment", type=str, required=True, 
                            help="name of the gym environment")
        parser.add_argument('--agent', type=str, default='dsc', choices=['dsc', 'sac', 'td3'],
                            help='choose which agent to run')
        parser.add_argument("--seed", type=int, default=1,
                            help="Random seed")
        parser.add_argument("--goal_state", nargs="+", type=float, default=[0, 8],
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
        if self.params['agent'] == 'dsc':
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
        self.params['device'] = utils.determine_device(self.params['disable_gpu'])

        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # create the saving directories
        saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        if os.path.exists(saving_dir):  # remove all existing contents
            shutil.rmtree(saving_dir)
        utils.create_log_dir(saving_dir)
        self.saving_dir = saving_dir

        # save the hyperparams
        utils.save_hyperparams(os.path.join(saving_dir, "hyperparams.csv"), self.params)

        # set up env and experiment
        self.env = self.make_batch_env(num_envs=self.params['num_envs'], test=False)
        self.test_env = self.make_batch_env(num_envs=1, test=True)
        if self.params['agent'] == 'dsc':
            self.exp = RobustDSC(mdp=self.env,
                            gestation_period=self.params['gestation_period'],
                            experiment_name=self.params['experiment_name'],
                            device=self.params['device'],
                            warmup_episodes=self.params['warmup_episodes'],
                            max_steps=self.params['steps'],
                            use_model=self.params['use_model'],
                            use_vf=self.params['use_value_function'],
                            use_global_vf=self.params['use_global_value_function'],
                            use_diverse_starts=self.params['use_diverse_starts'],
                            use_dense_rewards=self.params['use_dense_rewards'],
                            multithread_mpc=self.params['multithread_mpc'],
                            logging_freq=self.params['logging_frequency'],
                            evaluation_freq=self.params['evaluation_frequency'],
                            buffer_length=self.params['buffer_length'],
                            generate_init_gif=self.params['generate_init_gif'],
                            seed=self.params['seed'],
                            lr_c=self.params['lr_c'],
                            lr_a=self.params['lr_a'],
                            clear_option_buffers=self.params['clear_option_buffers'],
                            use_global_option_subgoals=self.params['use_global_option_subgoals'])
        else:
            self.agent = self.make_agent()
    
    def make_agent(self):
        """
        create the agent
        """
        if self.params['agent'] == 'sac':
            return make_sac_agent(self.env.observation_space, self.env.action_space, self.params)
        elif self.params['agent'] == 'td3':
            return make_td3_agent(self.env.observation_space, self.env.action_space, self.params)
        else:
            raise NotImplementedError
    
    def train(self):
        """
        train an agent
        """
        if self.params['goal_conditioned']:
            dummy_env = self.make_env(env_seed=0)
            goal_state = dummy_env.goal_state
            reward_fn = dummy_env.reward_func
        train_agent_batch_with_eval(
            agent=self.agent,
            env=self.env,
            num_episodes=self.params['episodes'],
            test_env=self.test_env,
            num_test_episodes=self.params['eval_n_episodes'],
            goal_conditioned=self.params['goal_conditioned'],
            goal_state=goal_state if self.params['goal_conditioned'] else None,
            logging_freq=self.params['logging_frequency'],
            testing_freq=self.params['testing_frequency'],
            plotting_freq=self.params['plotting_frequency'],
            saving_freq=self.params['saving_frequency'],
            saving_dir=self.saving_dir,
            state_to_goal_fn=dummy_env.get_position,
            reward_fn=reward_fn,
        )

    def run(self):
        """
        run the actual experiment
        """
        start_time = time.time()
        if self.params['agent'] == 'dsc':
            durations = self.exp.run_loop(self.params['episodes'], self.params['steps'])
        else:
            self.train()
        end_time = time.time()

        # plot the learning curve when experiemnt is done
        plot_learning_curve(experiment_name=self.params['experiment_name'])

        print("Time taken: ", end_time - start_time)

        time_file = Path(self.saving_dir).joinpath('time_taken.txt')
        with open(time_file, 'w') as f:
            seconds = end_time - start_time
            hours = seconds // 3600
            f.write(f'{seconds} seconds\n')
            f.write(f'{hours} hours {(seconds - hours * 3600)/60} minutes')


    def make_env(self, env_seed, test=False):
        # make the env
        if utils.check_is_atari(self.params['environment']):
            # atari domains
            env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
                pfrl.wrappers.atari_wrappers.make_atari(self.params['environment'], max_frames=30*60*60),  # 30 min with 60 fps
                episode_life=not test,
                clip_rewards=not test,
                flicker=False,
                frame_stack=False,
            )
        else:
            # mujoco envs
            env = gym.make(self.params['environment'])
            env = pfrl.wrappers.CastObservationToFloat32(env)
            if self.params['normalize_action_space']:
                env = pfrl.wrappers.NormalizeActionSpace(env)
        # pick a goal state for the antmaze env
        if utils.check_is_antmaze(self.params['environment']):
            if self.params['goal_state']:
                goal_state = np.array(self.params['goal_state'])
                env.target_goal = goal_state  # TODO: this might be dangerous
            else:
                # default to D4RL goal state
                goal_state = np.array(env.target_goal)
            print(f"using goal state {goal_state} in env {self.params['environment']}")
            env.env_seed = env_seed
            env = D4RLAntMazeWrapper(env, 
                                    start_state=((0, 0)), 
                                    goal_state=goal_state, 
                                    use_dense_reward=self.params['use_dense_rewards'],
                                    use_diverse_starts=self.params['use_diverse_starts'])
        # seed the environment
        env_seed = 2 ** 32 - 1 - env_seed if test else env_seed
        env.seed(env_seed)
        return env


    def make_batch_env(self, num_envs, test=False):
        # get a batch of seeds
        process_seeds = np.arange(num_envs) + self.params['seed'] * num_envs
        assert process_seeds.max() < 2 ** 32
        # make vector env
        vec_env = EpisodicSyncVectorEnv(
            [
                (lambda i: lambda: self.make_env(env_seed=int(process_seeds[i]), test=test))(idx)
                for idx, env in enumerate(range(num_envs))
            ],
            max_episode_len=self.params['max_episode_len']
        )
        return vec_env


def main():
    trial = Trial()
    trial.run()


if __name__ == "__main__":
    main()
