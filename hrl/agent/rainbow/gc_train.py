import os
import json
import time
import pfrl
import random
import pickle
import argparse
import numpy as np

from hrl.utils import create_log_dir
from pfrl.wrappers import atari_wrappers
from hrl.agent.rainbow.rainbow import Rainbow
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper


BOTTOM_RIGHT_POSITION = (123, 148)


def make_env(env_name, seed, episode_life):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=30 * 60 * 60),
        episode_life=episode_life,
        clip_rewards=False
    )

    env.seed(seed)
    
    return MontezumaInfoWrapper(env)


def load_goal_images_from_disk(dir_path, max_frames=10):
    def load(file_name):
        path = os.path.join(dir_path, file_name)
        with open(path, "rb") as f:
            states = pickle.load(f)
        frames = [state if isinstance(state, atari_wrappers.LazyFrames) else state.frame for state in states]
        return frames[:max_frames]

    def load_bottom_right_goals():
        return load("bottom_right_states.pkl")

    def load_start_state_goals():
        return load("start_states.pkl")

    def load_rope_goals():
        return load("rope_states.pkl")

    return load_bottom_right_goals(), load_rope_goals()


def is_close(pos1, pos2, tol):
    return abs(pos1[0] - pos2[0]) <= tol and \
           abs(pos1[1] - pos2[1]) <= tol


def select_goal(env, bottom_right_frames, rope_frames):
    def get_rope_goal():
        return random.choice(rope_frames)

    def get_bottom_right_goal():
        return random.choice(bottom_right_frames)

    current_position = env.get_player_position()

    if is_close(current_position, BOTTOM_RIGHT_POSITION, tol=2):
        return get_rope_goal(), "rope"
    
    return get_bottom_right_goal(), "bottom_right"


def bottom_right_reward_function(info):
    pos1 = info["player_x"], info["player_y"]
    done = is_close(pos1, BOTTOM_RIGHT_POSITION, tol=2)
    reward = float(done)
    return reward, done


def rope_reward_function(info):
    done = info["on_rope"]
    reward = float(done)
    return reward, done


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

    _log_file = f"logs/{args.experiment_name}/{args.seed}/rainbow_her_log.pkl"
    _path_to_goals = os.path.join(os.path.expanduser("~"), "git-repos/hrl/logs/goal_states")

    env = make_env(args.environment_name, seed=args.seed, episode_life=args.terminal_on_loss_of_life)

    pfrl.utils.set_random_seed(args.seed)

    rainbow_agent = Rainbow(n_actions=env.action_space.n,
                            n_atoms=args.n_atoms,
                            v_min=args.v_min,
                            v_max=args.v_max,
                            noisy_net_sigma=args.noisy_net_sigma,
                            lr=args.lr,
                            n_steps=args.n_steps,
                            betasteps=args.num_training_steps / 4,
                            replay_start_size=args.replay_start_size,
                            gpu=args.gpu_id,
                            goal_conditioned=args.goal_conditioned
                    )

    bottom_right_goals, rope_goals = load_goal_images_from_disk(_path_to_goals)

    t0 = time.time()
    needs_reset = True
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0
    num_training_steps = 13_000_000

    _log_steps = []
    _log_rewards = []
    _log_max_rewards = []

    rainbow_agent.set_rope_reward_func(rope_reward_function)
    rainbow_agent.set_bottom_right_reward_func(bottom_right_reward_function)

    while current_step_number < num_training_steps:
        
        if needs_reset:
            s0 = env.reset()

        g0, g_str = select_goal(env, bottom_right_goals, rope_goals)

        episodic_reward, episodic_duration, max_episodic_reward, reached, needs_reset = \
                                                                  rainbow_agent.gc_rollout(env,
                                                                                           s0, 
                                                                                           g0,
                                                                                           g_str,
                                                                                           current_episode_number,
                                                                                           max_episodic_reward)

        current_episode_number += 1
        current_step_number += episodic_duration

        _log_steps.append(current_step_number)
        _log_rewards.append(episodic_reward)
        _log_max_rewards.append(max_episodic_reward)

        with open(_log_file, "wb+") as f:
            episode_metrics = {
                            "step": _log_steps, 
                            "reward": _log_rewards,
                            "max_reward": _log_max_rewards
            }
            pickle.dump(episode_metrics, f)

    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
