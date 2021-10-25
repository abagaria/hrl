import os
import json
import time
import pfrl
import pickle
import random
import argparse

from hrl.utils import create_log_dir
from hrl.agent.dsc.dsc import RobustDSC
from pfrl.wrappers import atari_wrappers
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper


def make_env(env_name, seed, terminal_on_loss_of_life=False):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=30 * 60 * 60),
        episode_life=terminal_on_loss_of_life,
        clip_rewards=False
    )

    env.seed(seed)

    return MontezumaInfoWrapper(env)


def load_goal_state(dir_path):
    file_name = os.path.join(dir_path, "bottom_right_states.pkl")
    with open(file_name, "rb") as f:
        goals = pickle.load(f)
    return random.choice(goals).frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--gestation_period", type=int, default=10)
    parser.add_argument("--buffer_length", type=int, default=50)
    parser.add_argument("--num_training_steps", type=int, default=int(2e6))
    parser.add_argument("--terminal_on_loss_of_life", action="store_true", default=False)
    parser.add_argument("--use_oracle_rf", action="store_true", default=False)
    args = parser.parse_args()

    create_log_dir("logs")
    create_log_dir(f"logs/{args.experiment_name}")
    create_log_dir(f"logs/{args.experiment_name}/{args.seed}")
    create_log_dir("plots")
    create_log_dir(f"plots/{args.experiment_name}")
    create_log_dir(f"plots/{args.experiment_name}/{args.seed}")
    create_log_dir(f"plots/{args.experiment_name}/{args.seed}/initiation_set_plots")
    create_log_dir(f"plots/{args.experiment_name}/{args.seed}/value_function_plots")


    with open(f"logs/{args.experiment_name}/{args.seed}/hyperparameters.txt", "w+") as _args_file:
        json.dump(args.__dict__, _args_file, indent=2)

    _log_file = f"logs/{args.experiment_name}/{args.seed}/dsc_log.pkl"

    env = make_env(args.environment_name,
                   seed=args.seed, terminal_on_loss_of_life=args.terminal_on_loss_of_life)

    s0 = env.reset()
    p0 = env.get_current_position()
    g0 = load_goal_state(os.path.join(os.path.expanduser("~"), "git-repos/hrl/logs/goal_states"))

    pfrl.utils.set_random_seed(args.seed)

    dsc_agent = RobustDSC(env,
                          args.gestation_period,
                          args.buffer_length,
                          args.experiment_name,
                          args.gpu_id,
                          args.use_oracle_rf,
                          s0, p0, g0, (123, 148),
                          args.seed,
                          _log_file)

    t0 = time.time()
    dsc_agent.run_loop(args.num_training_steps)
    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
