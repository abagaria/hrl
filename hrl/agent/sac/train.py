import ipdb
import gym
import d4rl
import json
import pfrl
import time
import pickle
import argparse
import numpy as np

from hrl.agent.sac.sac import SAC
from hrl.utils import create_log_dir
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
from hrl.agent.sac.utils import make_chunked_value_function_plot 
from hrl.agent.sac.utils import make_chunked_action_value_function_plot


def make_env(name, start, goal, seed, horizon=1000):
    env = gym.make(name)
    env = D4RLAntMazeWrapper(env,
                            start_state=start,
                            goal_state=goal,
                            use_dense_reward=False,
                            goal_reward=1.,
                            step_reward=0.)
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
    parser.add_argument("--num_training_episodes", type=int, default=1000)
    parser.add_argument("--use_random_starts", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_start_size", type=int, default=10_000)
    parser.add_argument("--replay_buffer_size", type=int, default=10**6)
    parser.add_argument("--policy_output_scale", type=float, default=1.0)
    parser.add_argument("--plot_value_function", action="store_true", default=False)
    args = parser.parse_args()

    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")
    
    create_log_dir("plots")
    create_log_dir(f"plots/{args.experiment_name}")
    create_log_dir(f"plots/{args.experiment_name}/{args.seed}")

    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/sac_log.pkl"

    env = make_env(args.environment_name,
                   start=np.array([0., 0.]),
                   goal=np.array([0., 8.]),
                   seed=args.seed)
    
    pfrl.utils.set_random_seed(args.seed)

    sac_agent = SAC(obs_size=env.observation_space.shape[0],
                    action_size=env.action_space.shape[0],
                    batch_size=args.batch_size,
                    replay_start_size=args.replay_start_size,
                    replay_buffer_size=args.replay_buffer_size,
                    policy_output_scale=args.policy_output_scale,
                    action_space_low=env.action_space.low,
                    action_space_high=env.action_space.high,
                    gpu=args.gpu_id)

    t0 = time.time()
    
    _log_steps = []
    _log_rewards = []

    for current_episode in range(args.num_training_episodes):
        env.reset()

        if args.use_random_starts:
            env.set_xy(
                env.sample_random_state(
                    reject_cond=env.is_goal_region
                )
            )
        
        s0 = env.cur_state.astype(np.float32, copy=False)
        episode_reward, episode_length = sac_agent.rollout(env, s0, current_episode)

        _log_steps.append(episode_length)
        _log_rewards.append(episode_reward)

        with open(_log_file, "wb+") as f:
            episode_metrics = {
                            "step": _log_steps, 
                            "reward": _log_rewards,
            }
            pickle.dump(episode_metrics, f)

        if args.plot_value_function:
            make_chunked_value_function_plot(sac_agent,
                                            current_episode,
                                            args.seed,
                                            args.experiment_name)
            make_chunked_action_value_function_plot(sac_agent,
                                                    current_episode,
                                                    args.seed,
                                                    args.experiment_name)

    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
