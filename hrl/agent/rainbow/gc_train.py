import os
import json
import time
import pfrl
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import collections
import pdb

from tqdm import tqdm
from hrl.agent.rainbow.graph import Graph
from pfrl.wrappers import atari_wrappers
from hrl.utils import create_log_dir
from hrl.agent.rainbow.rainbow import Rainbow
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper
from hrl.agent.rainbow.goal_states import ALL_GOAL_POSITIONS, ALL_GOAL_FILES

room_one = Graph()
location_to_lf = {}

def build_room_one_graph():
    edges = [
        # does not include ladder walk from 77, 235 to 77, x.
        # does not include position at rope
        # does not include key 
        # does not include door left or door right

        ((20, 148), (21, 192)),
        ((20, 148), (25, 148)),
        ((25, 148), (38, 148)),
        ((38, 148), (50, 148)),
        ((50, 148), (62, 148)),
        ((62, 148), (75, 148)),
        ((75, 148), (99, 148)),
        ((99, 148), (114, 148)),
        ((114, 148), (123, 148)),
        ((123, 148), (130, 192)),
        ((77, 235), (130, 192))
    ]

    nodes = []

    for edge in edges:
        nodes.append(edge[0])
        nodes.append(edge[1])

    nodes = list(set(nodes))
    for node in nodes:
        env.reset()
        env.set_player_position(*node)
        obs, _, _, _ = env.env.step(0) # NO-OP
        location_to_lf[node] = obs


    for start, end in edges:
        room_one.addEdge(start, end)
        room_one.addEdge(end, start)


def make_env(env_name, seed, episode_life, max_frames):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=max_frames),
        episode_life=episode_life,
        clip_rewards=False
    )
    env.seed(seed)
    return MontezumaInfoWrapper(env)


# def load_goal_state(dir_path, file):
#     file_name = os.path.join(dir_path, file)
#     with open(file_name, "rb") as f:
#         goals = pickle.load(f)
#     goal = random.choice(goals)
#     if hasattr(goal, "frame"):
#         return goal.frame
#     if isinstance(goal, atari_wrappers.LazyFrames):
#         return goal
#     return goal.obs


# def load_all_goal_states(dir_path):
#     goal_observations = []
#     for file_path in ALL_GOAL_FILES:
#         goal_observations.append(
#             load_goal_state(dir_path, file_path)
#         )
#     return goal_observations


def is_close(pos1, pos2, tol=2.):
    return abs(pos1[0] - pos2[0]) <= tol and \
           abs(pos1[1] - pos2[1]) <= tol


def reward_func(info, goal_pos):
    pos = info['player_x'], info['player_y']
    reached = is_close(pos, goal_pos)
    return float(reached), reached


# def select_goal(goal_observations, current_pos):
#     target_positions = [i for i, pos in enumerate(ALL_GOAL_POSITIONS) if not is_close(pos, current_pos)]
#     sampled_idx = random.choice(target_positions)
#     return ALL_GOAL_POSITIONS[sampled_idx], goal_observations[sampled_idx]

def select_start():
    return random.choice(list(location_to_lf.keys()))

def select_goal(start):
    non_start_goals = [node for node in room_one.getAllNodes() if node != start]
    goal = random.choice(non_start_goals)

    path = room_one.getPath(start, goal)
    gloc = random.choice(path)
    gobs = location_to_lf[gloc]
    return gloc, gobs

def make_chunked_gc_value_function_plot(pfrl_agent, states, pos_buffer, goal, goal_pos, episode, seed, experiment_name, chunk_size=1000):
    assert isinstance(pfrl_agent, Rainbow)

    def get_augmented_states(s, g):
        assert isinstance(g, (np.ndarray, atari_wrappers.LazyFrames)), type(g)
        g = g._frames[-1] if isinstance(g, atari_wrappers.LazyFrames) else g
        
        augmented_states = [atari_wrappers.LazyFrames(ss._frames+[g], stack_axis=0) for ss in s]
        return augmented_states

    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    augmented_states = get_augmented_states(states, goal)
    augmented_state_chunks = get_chunks(augmented_states, chunk_size)
    values = np.zeros((len(augmented_states),))
    current_idx = 0

    for state_chunk in tqdm(augmented_state_chunks, desc="Making VF plot"):
        chunk_values = pfrl_agent.value_function(state_chunk).cpu().numpy()
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values
        current_idx += current_chunk_size

    x = [state[0] for state in pos_buffer]
    y = [state[1] for state in pos_buffer]

    plt.scatter(x, y, c=values)
    plt.xlabel(f'goal: {goal_pos}')
    plt.colorbar()
    create_log_dir(f'plots/{experiment_name}')
    create_log_dir(f'plots/{experiment_name}/{seed}')
    file_name = f"rainbow_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"plots/{experiment_name}/{seed}/{file_name}-{goal_pos}.png")
    plt.close()

    return values.max()


def make_generalized_value_function_plot(pfrl_agent, start, goal, ram_buffer, pos_buffer, episode, seed, experiment_name):
    path = room_one.getPath(start, goal)
    pairs = []
    
    max_episodic_reward = 0

    # exclude start and end from pairs
    it1 = 0
    it2 = len(path) - 1
    while it2 >= 0:
        pairs.append((path[0], path[it2]))
        # it1 += 1
        it2 -= 1

    print(pairs) # len(path) / 2

    make_chunked_gc_value_function_plot(pfrl_agent, ram_buffer, pos_buffer, location_to_lf[goal], goal, episode, seed, "test")

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
    parser.add_argument("--replay_start_size", type=int, default=1000) # TODO maybe change
    parser.add_argument("--replay_buffer_size", type=int, default=int(3e5))
    parser.add_argument("--num_training_steps", type=int, default=int(1e6))
    parser.add_argument("--max_frames_per_episode", type=int, default=1000)  # 30 mins
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

    # goal_observations = load_all_goal_states(_path_to_goals)
    build_room_one_graph()

    t0 = time.time()
    needs_reset = True
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0

    _log_steps = []
    _log_rewards = []
    _log_max_rewards = []
    _log_pursued_goals = []

    state_buffer = collections.deque(maxlen=10000)
    pos_buffer = collections.deque(maxlen=10000)

    while current_step_number < args.num_training_steps:
        
        if needs_reset:
            s0, info0 = env.reset()
            current_episode_number += 1
            print("="*80); print(f"Episode {current_episode_number}"); print("="*80)

        # TODO select start
        spos = select_start()
        s0 = env.set_player_position(*spos)
        gpos, gobs = select_goal(spos)

        episodic_reward, episodic_duration, max_episodic_reward, needs_reset, state_trajectory, pos_trajectory = \
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

        state_buffer.extend(state_trajectory)
        pos_buffer.extend(pos_trajectory)

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

    make_generalized_value_function_plot(rainbow_agent, spos, gpos, state_buffer, pos_buffer, current_episode_number, args.seed, "gcrl")
    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
