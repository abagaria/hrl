# GOAL: collect data to train distance learners.
# preliminary goal: Figure out how good our policy is. Where can it reach/not reach the goal from?

import gym
import d4rl
import json
import pfrl
import time
import torch
import pickle
import argparse
import numpy as np

from hrl.utils import create_log_dir
from hrl.agent.td3.TD3AgentClass import TD3 as OldTD3
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.wrappers.environments.ant_maze_env import AntMazeEnv
from hrl.agent.td3.utils import make_chunked_value_function_plot, make_reward_function_plot
from hrl.experiments.TD3DeployAgentClass import TD3


def make_env(name, start, goal, dense_reward, seed, horizon=1000):
    if "reacher" not in name.lower():
        env = gym.make(name)
    else:
        gym_mujoco_kwargs = {
            'maze_id': 'Reacher',
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 3,
            'color_str': ""
        }
        env = AntMazeEnv(**gym_mujoco_kwargs)

    goal_reward = 0. if dense_reward else 1.

    env = D4RLAntMazeWrapper(env,
                            start_state=start,
                            goal_state=goal,
                            use_dense_reward=dense_reward,
                            goal_reward=goal_reward,
                            step_reward=0.)

    env = pfrl.wrappers.CastObservationToFloat32(env)
    env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=horizon)
    env.env.seed(seed)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--random_action_freq", type=float)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--num_deploy_episodes", type=int, default=1000)
    parser.add_argument("--max_episode_length", type=int, default=200)
    args = parser.parse_args()


    args.experiment_name += f"_random_freq_{args.random_action_freq}"
    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")


    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _buffer_log_file = f"logs/{args.experiment_name}/{args.seed}/td3_replay_buffer.pkl"

    env = make_env(args.environment_name,
                   start=np.array([0., 0.]),
                   goal=np.array([0., 8.]),
                   seed=args.seed,
                   dense_reward=False,
                   horizon=args.max_episode_length)
    
    pfrl.utils.set_random_seed(args.seed)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    with open(f"logs/ant_baseline_policy_dense/0/model.pkl", 'rb') as f:
        agent = pickle.load(f)

    agent = TD3(
        agent.actor,
        agent.critic,
        obs_size,
        action_size,
        max_action=1.,
        random_action_freq=args.random_action_freq,
        sample_random_action=env.action_space.sample,
        use_output_normalization=False,
        device=torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id > -1 else "cpu"
        ),
        store_extra_info=True
    )
    
    t0 = time.time()
    
    for current_episode in range(args.num_deploy_episodes):
        env.reset()
        # TODO: Don't reset env manually
        env.set_xy(env.sample_random_state(reject_cond=lambda x: env.is_goal_region(x) or x[1] < 6))
        
        s0 = env.cur_state.astype(np.float32, copy=False)
        episode_reward, episode_length = agent.rollout(env, s0, current_episode)
            
    agent.replay_buffer.save(_buffer_log_file)


    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
