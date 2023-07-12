import ipdb
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
from hrl.agent.td3.utils import save
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
# from hrl.wrappers.environments.ant_maze_env import AntMazeEnv
from hrl.agent.td3.utils import make_chunked_value_function_plot 


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


# TODO(rafa): Make sure that the goal is not achieved at the start state.
def sample_goal(env) -> np.ndarray:
  """Output the deltax and deltay."""
  return env.sample_random_state()


def run_episode(agent, env, state, current_episode, num_steps=1000):
  done = False
  
  goal = sample_goal(env)
  env.set_goal(goal)
  print(f'Episode: {current_episode} Goal: {goal}')
  
  total_reward = 0.
  trajectory = []

  for i in range(num_steps):
    augmented_state = np.concatenate((state, goal), axis=0)
    action = agent.act(augmented_state)
    
    next_state, reward, done, info = env.step(action)
    augmented_next_state = np.concatenate((next_state, goal), axis=0)
    transition = (augmented_state, action, reward, augmented_next_state, done)
    
    agent.step(*transition)

    trajectory.append(transition)
    state = next_state
    total_reward += reward

    if done:
      break

  print(f'Episode: {current_episode} Reward: {total_reward} Steps: {i}')
  return trajectory, state, done


def train(agent, env, num_episodes, num_steps_per_episode):
  for episode in range(num_episodes):
    s0 = env.reset()
    trajectory, state, reached = run_episode(
      agent, env, s0, episode, num_steps_per_episode)
    
    if not reached:
      hindsight_goal = state[:2]
      her(agent, env, trajectory, hindsight_goal)


def her(agent, env, trajectory, new_goal):
  assert new_goal.shape == (2, )
  for state, action, _, next_state, _ in trajectory:
    augmented_state = state.copy()
    augmented_state[-2:] = new_goal
    augmented_next_state = next_state.copy()
    augmented_next_state[-2:] = new_goal
    reward, done = env.sparse_gc_reward_func(next_state, new_goal)
    agent.step(
      augmented_state, action, reward, augmented_next_state, done
    )
  


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int)
  parser.add_argument("--gpu_id", type=int)
  parser.add_argument("--experiment_name", type=str)
  parser.add_argument("--environment_name", type=str)
  parser.add_argument("--num_training_episodes", type=int, default=1000)
  parser.add_argument("--use_random_starts", action="store_true", default=False)
  parser.add_argument("--plot_value_function", action="store_true", default=False)
  parser.add_argument("--use_dense_rewards", action="store_true", default=False)
  parser.add_argument("--save_replay_buffer", action="store_true", default=False)
  parser.add_argument("--save_agent", action="store_true", default=False)
  args = parser.parse_args()

  create_log_dir("logs")
  create_log_dir(f"logs/{args.experiment_name}")
  create_log_dir(f"logs/{args.experiment_name}/{args.seed}")
  
  create_log_dir("plots")
  create_log_dir(f"plots/{args.experiment_name}")
  create_log_dir(f"plots/{args.experiment_name}/{args.seed}")

  create_log_dir("saved_modes")
  create_log_dir(f"saved_models/{args.experiment_name}")
  create_log_dir(f"saved_models/{args.experiment_name}/{args.seed}")

  with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
    json.dump(args.__dict__, _args_file, indent=2)

  _log_file = f"logs/{args.experiment_name}/{args.seed}/td3_log.pkl"
  _buffer_log_file = f"logs/{args.experiment_name}/{args.seed}/td3_replay_buffer.pkl"

  env = make_env(args.environment_name,
           start=np.array([8., 0.]),
           goal=np.array([0., 0.]),
           seed=args.seed,
           dense_reward=args.use_dense_rewards)
  
  pfrl.utils.set_random_seed(args.seed)

  obs_size = env.observation_space.shape[0]
  action_size = env.action_space.shape[0]

  agent = TD3(obs_size,
        action_size,
        max_action=1.,
        use_output_normalization=False,
        device=torch.device(
          f"cuda:{args.gpu_id}" if args.gpu_id > -1 else "cpu"
        ),
        store_extra_info=args.save_replay_buffer
  )

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
    episode_reward, episode_length = agent.rollout(env, s0, current_episode)

    _log_steps.append(episode_length)
    _log_rewards.append(episode_reward)

    with open(_log_file, "wb+") as f:
      episode_metrics = {
              "step": _log_steps, 
              "reward": _log_rewards,
      }
      pickle.dump(episode_metrics, f)

    if args.plot_value_function and current_episode % 10 == 0:
      make_chunked_value_function_plot(agent,
                      current_episode,
                      args.seed,
                      args.experiment_name)
    
    if args.save_replay_buffer and current_episode % 100 == 0:
      agent.replay_buffer.save(_buffer_log_file)

    if args.save_agent and current_episode % 500 == 0:
      save(agent, f"saved_models/{args.experiment_name}/{args.seed}/td3_episode_{current_episode}")

  print(f"Finished after {(time.time() - t0) / 3600.} hrs")