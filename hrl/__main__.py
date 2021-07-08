import time
import argparse

import gym
import d4rl
import torch
import seeding
import numpy as np

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.utils import create_log_dir
from hrl.agent.dsc.dsc import RobustDSC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
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

    parser.add_argument("--maze_type", type=str)
    parser.add_argument("--use_global_option_subgoals", action="store_true", default=False)

    parser.add_argument("--clear_option_buffers", action="store_true", default=False)
    parser.add_argument("--lr_c", type=float, help="critic learning rate")
    parser.add_argument("--lr_a", type=float, help="actor learning rate")
    args = parser.parse_args()

    assert args.use_model or args.use_value_function

    if not args.use_value_function:
        assert not args.use_global_value_function

    if args.clear_option_buffers:
        assert not args.use_global_value_function

    if args.environment in ["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"]:
        env = gym.make(args.environment)
        env = D4RLAntMazeWrapper(env, start_state=((0, 0)), goal_state=np.array((0, 8)))
        seeding.seed(0, torch, np)
        seeding.seed(args.seed, gym, env)
    else:
        raise NotImplementedError("Environment not supported!")

    exp = RobustDSC(mdp=env,
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
                    maze_type=args.maze_type,
                    use_global_option_subgoals=args.use_global_option_subgoals)

    create_log_dir(args.experiment_name)
    create_log_dir("initiation_set_plots/")
    create_log_dir("value_function_plots/")
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    end_time = time.time()

    print("Time taken: ", end_time - start_time)

