import os
import time
import json
import pfrl
import random
import pickle
import argparse

from hrl.utils import create_log_dir
from hrl.agent.dq_demonstrations.supervised_rainbow import SupervisedRainbow
from pfrl.wrappers import atari_wrappers
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper
from hrl.agent.dq_demonstrations.dataloader import load_trajectory

def make_env(env_name, seed, terminal_on_loss_of_life=False):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=30 * 60 * 60),
        episode_life=terminal_on_loss_of_life,
        clip_rewards=False
    )

    env.seed(seed)

    return MontezumaInfoWrapper(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--n_atoms", type=int, default=51)
    parser.add_argument("--v_max", type=float, default=+10.)
    parser.add_argument("--v_min", type=float, default=-10.)
    parser.add_argument("--noisy_net_sigma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--n_steps", type=int, default=3)
    parser.add_argument("--replay_start_size", type=int, default=80_000)
    parser.add_argument("--replay_buffer_size", type=int, default=10**6)
    parser.add_argument("--num_training_steps", type=int, default=int(13e6))
    parser.add_argument("--use_her", action="store_true", default=False)
    parser.add_argument("--goal_conditioned", action="store_true", default=False)
    parser.add_argument("--terminal_on_loss_of_life", action="store_true", default=False)
    args = parser.parse_args()

    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")

    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/rainbow_log.pkl"

    env = make_env(args.environment_name,
                   seed=args.seed, terminal_on_loss_of_life=args.terminal_on_loss_of_life)
    
    pfrl.utils.set_random_seed(args.seed)

    rainbow_agent = SupervisedRainbow(
        n_actions=env.action_space.n,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        noisy_net_sigma=args.noisy_net_sigma,
        lr=args.lr,
        n_steps=args.n_steps,
        betasteps=args.num_training_steps / 4,
        replay_start_size=args.replay_start_size,
        replay_buffer_size=args.replay_buffer_size,
        demonstration_buffer_size=args.replay_buffer_size,
        gpu=args.gpu_id,
        goal_conditioned=args.goal_conditioned
    )

    data_dir = os.path.expanduser("~/Documents/research/code/montezuma/atari_v1")
    trajectory = load_trajectory(data_dir, 1)
    rainbow_agent.add_demonstration_trajectory(trajectory)

    t0 = time.time()
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0

    while current_step_number < args.num_training_steps:

        s0,_ = env.reset()

        episodic_reward, episodic_duration, max_episodic_reward = rainbow_agent.rollout(
            env,
            s0,
            current_episode_number,
            max_episodic_reward
        )

        current_episode_number += 1
        current_step_number += episodic_duration

    print(f"Finished after {(time.time() - t0) / 3600.} hrs")

