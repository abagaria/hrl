import json
import time
import pfrl
import pickle
import argparse
from hrl.utils import create_log_dir
from hrl.agent.dqn.dqn import DQN
from pfrl.wrappers import atari_wrappers


def make_env(env_name, seed, test_mode, episode_life, test_epsilon=0.05):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=None),
        episode_life=episode_life,
        clip_rewards=False
    )

    env.seed(seed)

    if test_mode:
        env = pfrl.wrappers.RandomizeAction(env, test_epsilon)
    
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--use_double_dqn", action="store_true", default=False)
    parser.add_argument("--terminal_on_loss_of_life", action="store_true", default=False)
    args = parser.parse_args()

    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")

    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/dqn_log.pkl"

    env = make_env(args.environment_name, test_mode=False,
                   seed=args.seed, episode_life=args.terminal_on_loss_of_life)

    pfrl.utils.set_random_seed(args.seed)

    dqn_agent = DQN(n_actions=env.action_space.n, gpu=args.gpu_id, use_double_dqn=args.use_double_dqn)

    t0 = time.time()
    current_step_number = 0
    max_episodic_reward = 0
    current_episode_number = 0
    num_training_steps = 13000000

    _log_steps = []
    _log_rewards = []
    _log_max_rewards = []

    while current_step_number < num_training_steps:
        s0 = env.reset()

        episodic_reward, episodic_duration, max_episodic_reward = dqn_agent.rollout(env,
                                                                                    s0,
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
