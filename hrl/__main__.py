import time
import os
import random
import argparse

import gym
import d4rl
import torch
import seeding
import numpy as np

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.utils import create_log_dir
from hrl.agent.dsc.dsc import RobustDSC
from hrl.agent.dsc.dst import RobustDST


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--environment", type=str, choices=["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"], 
                        help="name of the gym environment")
    parser.add_argument("--seed", type=int, help="Random seed")

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
    parser.add_argument("--logging_frequency", type=int, default=50, help="Draw init sets, etc after every _ episodes")
    parser.add_argument("--generate_init_gif", action="store_true", default=False)
    parser.add_argument("--evaluation_frequency", type=int, default=10)

    parser.add_argument("--goal_state", nargs="+", type=float, default=[],
                        help="specify the goal state of the environment, (0, 8) for example")
    parser.add_argument("--use_global_option_subgoals", action="store_true", default=False)
    parser.add_argument("--lr_c", type=float, help="critic learning rate")
    parser.add_argument("--lr_a", type=float, help="actor learning rate")
    parser.add_argument("--use_skill_trees", action="store_true", default=False)
    parser.add_argument("--max_num_children", type=int, default=1, help="Max number of children per option in the tree")
    
    # Off policy init learning configs
    parser.add_argument("--init_classifier_type", type=str, default="position-clf")
    parser.add_argument("--optimistic_threshold", type=float, default=0.5)
    parser.add_argument("--pessimistic_threshold", type=float, default=0.75)
    parser.add_argument("--use_initiation_gvf", action="store_true", default=False)
    args = parser.parse_args()

    assert args.use_model or args.use_value_function

    possible_thresholders = "critic-threshold", "ope-threshold"
    possible_clfs = "position-clf", "pos-critic-clf", "pos-ope-clf",\
                    "state-clf", "state-critic-clf", "state-ope-clf"\
                    "state-svm", "state-critic-svm"
    assert args.init_classifier_type in possible_thresholders or \
           args.init_classifier_type in possible_clfs

    if not args.use_value_function:
        assert not args.use_global_value_function

    if args.use_skill_trees:
        assert args.max_num_children > 1, f"{args.use_skill_trees, args.max_num_children}"

    if args.environment in ["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"]:
        env = gym.make(args.environment)
        # pick a goal state for the env
        if args.goal_state:
            goal_state = np.array(args.goal_state)
        else:
            # default to D4RL goal state
            goal_state = np.array(env.target_goal)
        print(f'using goal state {goal_state} in env {args.environment}')
        env = D4RLAntMazeWrapper(
            env,
            start_state=np.array((0, 0)),
            goal_state=goal_state,
            init_truncate="position" in args.init_classifier_type,
            use_dense_reward=args.use_dense_rewards
        )

        torch.manual_seed(0)
        seeding.seed(0, random, np)
        seeding.seed(args.seed, gym, env)

    else:
        raise NotImplementedError("Environment not supported!")

    kwargs = {
            "mdp":env,
            "gestation_period": args.gestation_period,
            "experiment_name": args.experiment_name,
            "device": torch.device(args.device),
            "warmup_episodes": args.warmup_episodes,
            "max_steps": args.steps,
            "use_model": args.use_model,
            "use_vf": args.use_value_function,
            "use_global_vf": args.use_global_value_function,
            "use_diverse_starts": args.use_diverse_starts,
            "use_dense_rewards": args.use_dense_rewards,
            "multithread_mpc": args.multithread_mpc,
            "logging_freq": args.logging_frequency,
            "evaluation_freq": args.evaluation_frequency,
            "buffer_length": args.buffer_length,
            "generate_init_gif": args.generate_init_gif,
            "seed": args.seed,
            "lr_c": args.lr_c,
            "lr_a": args.lr_a,
            "max_num_children": args.max_num_children,
            "init_classifier_type": args.init_classifier_type, 
            "optimistic_threshold": args.optimistic_threshold,
            "pessimistic_threshold": args.pessimistic_threshold,
            "use_initiation_gvf": args.use_initiation_gvf
    }

    exp = RobustDST(**kwargs) if args.use_skill_trees else RobustDSC(**kwargs)

    # create the saving directories
    saving_dir = os.path.join(args.results_dir, args.experiment_name)
    create_log_dir(saving_dir)
    create_log_dir(os.path.join(saving_dir, "initiation_set_plots/"))
    create_log_dir(os.path.join(saving_dir, "value_function_plots/"))

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    end_time = time.time()

    print("Time taken: ", end_time - start_time)

