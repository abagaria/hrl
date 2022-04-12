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
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.wrappers.environments.ant_maze_env import AntMazeEnv
from hrl.agent.td3.utils import make_chunked_value_function_plot, make_reward_function_plot


def make_env(name, seed, horizon=1000):
    env = gym.make(name, exclude_current_positions_from_observation=False)
    env = pfrl.wrappers.CastObservationToFloat32(env)
    env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=horizon)
    env.env.seed(seed)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--quantile", type=float)
    parser.add_argument("--num_training_episodes", type=int, default=1000)
    args = parser.parse_args()

    args.experiment_name += f"_quantile_{args.quantile}"
    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")
    
    create_log_dir("plots")
    create_log_dir(f"plots/{args.experiment_name}")
    create_log_dir(f"plots/{args.experiment_name}/{args.seed}")

    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/td3_log.pkl"
    _buffer_log_file = f"logs/{args.experiment_name}/{args.seed}/td3_replay_buffer.pkl"

    env = make_env(args.environment_name,
                   seed=args.seed)
    
    pfrl.utils.set_random_seed(args.seed)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = TD3(obs_size,
                action_size,
                max_action=1.,
                distance_function_quantile=args.quantile,
                use_output_normalization=False,
                device=torch.device(
                    f"cuda:{args.gpu_id}" if args.gpu_id > -1 else "cpu"
                ),
                store_extra_info=False
    )


    t0 = time.time()
    
    _log_steps = []
    _log_rewards = []

    for current_episode in range(args.num_training_episodes):
        s0 = env.reset().astype(np.float32, copy=False)
        
        episode_reward, episode_length = agent.rollout(env, s0, current_episode)

        _log_steps.append(episode_length)
        _log_rewards.append(episode_reward)

        with open(_log_file, "wb+") as f:
            episode_metrics = {
                            "step": _log_steps, 
                            "reward": _log_rewards,
            }
            pickle.dump(episode_metrics, f)

        if current_episode % 20 == 0:
            agent.distance_function.plot(f"plots/{args.experiment_name}/{args.seed}/episode_{current_episode}_distance_function.png")
            

    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
