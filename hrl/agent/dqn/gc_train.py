import os
import json
import time
import pfrl
import random
import pickle
import argparse
import numpy as np

from pfrl.wrappers import atari_wrappers
from hrl.utils import create_log_dir
from hrl.agent.dqn.dqn import DQN
from hrl.montezuma.montezuma_mdp import MontezumaMDP


STARTING_POSITION = (77, 235)
BOTTOM_RIGHT_POSITION = (123, 148)

""" 
TODO: 
1. [Done] Goal-conditioned architecture for the Q-function
2. Function to load the goal images from disk
   i)  [Done] bottom right of the screen
   ii) [Remaining] initial spawn player position
3. [Done] Function to select a goal from the set of possible goals
4. Goal-conditioned reward function
5. Augmenting state and goal
6. Hindsight experience replay
7. Evaluation rollouts
"""


def make_env(env_name, seed, test_mode, episode_life, test_epsilon=0.05, max_frames=None):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=max_frames),
        episode_life=episode_life,
        clip_rewards=False
    )

    env.seed(seed)

    if test_mode:
        env = pfrl.wrappers.RandomizeAction(env, test_epsilon)
    
    return MontezumaMDP(env, render=False)


def load_goal_images_from_disk(dir_path):
    def load_bottom_right_goals():
        file_name = os.path.join(dir_path, "bottom_right_states.pkl")
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def load_start_state_goals():
        file_name = os.path.join(dir_path, "start_states.pkl")
        with open(file_name, "rb") as f:
            return pickle.load(f)

    return load_start_state_goals(), load_bottom_right_goals()


def select_goal(state, bottom_right_frames, init_frames):
    if (state.position == STARTING_POSITION).all():
        return random.choice(bottom_right_frames)

    return random.choice(init_frames)


def goal_conditioned_reward_function(state, goal):
    done = (state.position == goal.position).all()
    reward = float(done)
    return reward, done


def reward_function(state):
    done = np.isclose(state.position, np.array([123., 148.]))
    reward = float(done)
    return reward, done

    
def reset(mdp):
    mdp.reset()

    while mdp.cur_state.position[0] != 77 and mdp.cur_state.position[1] != 235:
        mdp.execute_agent_action(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--end_epsilon", type=float, default=0.1)
    parser.add_argument("--use_double_dqn", action="store_true", default=False)
    parser.add_argument("--terminal_on_loss_of_life", action="store_true", default=False)
    args = parser.parse_args()

    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")

    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/dqn_log.pkl"
    _path_to_goals = os.path.join(os.path.expanduser("~"), "git-repos/hrl/logs/goal_states")

    mdp = make_env(args.environment_name, test_mode=False, seed=args.seed,
                   max_frames=4500, episode_life=args.terminal_on_loss_of_life)

    pfrl.utils.set_random_seed(args.seed)

    dqn_agent = DQN(n_actions=len(mdp.actions),
                    gpu=args.gpu_id,
                    use_double_dqn=args.use_double_dqn,
                    lr=args.lr,
                    end_eps=args.end_epsilon,
                    goal_conditioned=False)

    init_goals, bottom_right_goals = load_goal_images_from_disk(_path_to_goals)

    t0 = time.time()
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0
    num_training_steps = 13_000_000

    _log_steps = []
    _log_rewards = []
    _log_max_rewards = []

    while current_step_number < num_training_steps:
        mdp.reset()

        s0 = mdp.cur_state
        g0 = None

        episodic_reward, episodic_duration, max_episodic_reward = dqn_agent.gc_rollout(mdp,
                                                                                       s0,
                                                                                       g0,
                                                                                       current_episode_number,
                                                                                       max_episodic_reward,
                                                                                       reward_func=reward_function)

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
