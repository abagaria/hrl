import os
import json
import time
import pfrl
import random
import pickle
import argparse

from pfrl.wrappers import atari_wrappers
from hrl.utils import create_log_dir
from hrl.agent.rainbow.rainbow import Rainbow
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper
from hrl.agent.rainbow.goal_states import ALL_GOAL_POSITIONS, ALL_GOAL_FILES


def make_env(env_name, seed, episode_life, max_frames):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=max_frames),
        episode_life=episode_life,
        clip_rewards=False
    )
    env.seed(seed)
    return MontezumaInfoWrapper(env)


def load_goal_state(dir_path, file):
    file_name = os.path.join(dir_path, file)
    with open(file_name, "rb") as f:
        goals = pickle.load(f)
    goal = random.choice(goals)
    if hasattr(goal, "frame"):
        return goal.frame
    if isinstance(goal, atari_wrappers.LazyFrames):
        return goal
    return goal.obs


def load_all_goal_states(dir_path):
    goal_observations = []
    for file_path in ALL_GOAL_FILES:
        goal_observations.append(
            load_goal_state(dir_path, file_path)
        )
    return goal_observations


def is_close(pos1, pos2, tol=2.):
    return abs(pos1[0] - pos2[0]) <= tol and \
           abs(pos1[1] - pos2[1]) <= tol


def reward_func(info, goal_pos):
    pos = info['player_x'], info['player_y']
    reached = is_close(pos, goal_pos)
    return float(reached), reached


def select_goal(goal_observations, current_pos):
    target_positions = [i for i, pos in enumerate(ALL_GOAL_POSITIONS) if not is_close(pos, current_pos)]
    sampled_idx = random.choice(target_positions)
    return ALL_GOAL_POSITIONS[sampled_idx], goal_observations[sampled_idx]


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
    parser.add_argument("--replay_buffer_size", type=int, default=int(3e5))
    parser.add_argument("--num_training_steps", type=int, default=int(2e6))
    parser.add_argument("--max_frames_per_episode", type=int, default=30*60*60)  # 30 mins
    args = parser.parse_args()

    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")

    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/rainbow_her_log.pkl"
    _path_to_goals = os.path.join(os.path.expanduser("~"), "git-repos/hrl/logs/goal_states")

    env = make_env(args.environment_name, seed=args.seed, 
                   episode_life=False, max_frames=args.max_frames_per_episode)

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
                            goal_conditioned=True,
                            replay_buffer_size=args.replay_buffer_size,
                    )

    goal_observations = load_all_goal_states(_path_to_goals)

    t0 = time.time()
    needs_reset = True
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0

    _log_steps = []
    _log_rewards = []
    _log_max_rewards = []
    _log_pursued_goals = []

    while current_step_number < args.num_training_steps:
        
        if needs_reset:
            s0, info0 = env.reset()
            current_episode_number += 1
            print("="*80); print(f"Episode {current_episode_number}"); print("="*80)

        gpos, gobs = select_goal(goal_observations, env.get_current_position())

        episodic_reward, episodic_duration, max_episodic_reward, needs_reset = \
        \
        rainbow_agent.gc_rollout(
            env,
            s0,
            gobs,
            gpos,
            reward_func,
            current_episode_number,
            max_episodic_reward
        )

        current_step_number += episodic_duration

        _log_steps.append(current_step_number)
        _log_rewards.append(episodic_reward)
        _log_max_rewards.append(max_episodic_reward)
        _log_pursued_goals.append(gpos)

        with open(_log_file, "wb+") as f:
            episode_metrics = {
                            "step": _log_steps, 
                            "reward": _log_rewards,
                            "max_reward": _log_max_rewards,
                            "goal": _log_pursued_goals
            }
            pickle.dump(episode_metrics, f)

    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
