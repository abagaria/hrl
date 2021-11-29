import os
import json
import time
import pfrl
import pickle
import random
import argparse

from pfrl.wrappers import atari_wrappers
from hrl.utils import create_log_dir
from hrl.agent.dsc.dsc import RobustDSC
from hrl.agent.dsg.dsg import SkillGraphAgent
from hrl.agent.dsg.trainer import DSGTrainer
from hrl.salient_event.salient_event import SalientEvent
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper


def make_env(env_name, seed, max_frames):
    env = atari_wrappers.wrap_deepmind(
        atari_wrappers.make_atari(env_name, max_frames=max_frames),
        episode_life=False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--gestation_period", type=int, default=10)
    parser.add_argument("--buffer_length", type=int, default=50)
    parser.add_argument("--num_training_steps", type=int, default=int(2e6))
    parser.add_argument("--use_oracle_rf", action="store_true", default=False)
    parser.add_argument("--use_pos_for_init", action="store_true", default=False)
    parser.add_argument("--max_num_options", type=int, default=5)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--max_frames_per_episode", type=int, default=30*60*60)  # 30 mins
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

    _log_file = f"logs/{args.experiment_name}/{args.seed}/dsg_log.pkl"

    env = make_env(args.environment_name,
                   seed=args.seed,
                   max_frames=args.max_frames_per_episode)

    s0, _ = env.reset()
    p0 = env.get_current_position()

    goal_dir_path = os.path.join(os.path.expanduser("~"), "git-repos/hrl/logs/goal_states")
    
    gpos = (123, 148)
    gpos1 = (132, 192)
    gpos2 = (24, 235)
    gpos3 = (130, 235)
    gpos4 = (77, 192)
    
    g0 = load_goal_state(goal_dir_path, file="bottom_right_states.pkl")
    g1 = load_goal_state(goal_dir_path, file="top_bottom_right_ladder_states.pkl")
    g2 = load_goal_state(goal_dir_path, file="left_door_goal.pkl")
    g3 = load_goal_state(goal_dir_path, file="right_door_goal.pkl")
    g4 = load_goal_state(goal_dir_path, file="bottom_mid_ladder_goal.pkl")

    pfrl.utils.set_random_seed(args.seed)

    beta0 = SalientEvent(s0, p0, tol=2.)
    beta1 = SalientEvent(g0, gpos, tol=2.)
    beta2 = SalientEvent(g1, gpos1, tol=2.)
    beta3 = SalientEvent(g2, gpos2, tol=2.)
    beta4 = SalientEvent(g3, gpos3, tol=2.)
    beta5 = SalientEvent(g4, gpos4, tol=2.)

    dsc_agent = RobustDSC(env,
                          args.gestation_period,
                          args.buffer_length,
                          args.experiment_name,
                          args.gpu_id,
                          beta0,
                          args.use_oracle_rf,
                          args.use_pos_for_init,
                          args.gamma,
                          args.max_num_options,
                          args.seed,
                          _log_file)

    dsg_agent = SkillGraphAgent(dsc_agent)
    
    trainer = DSGTrainer(env, dsc_agent, dsg_agent, 1000, 100,
                         [beta1, beta2, beta3, beta4, beta5])
    
    t0 = time.time()
    trainer.run_loop(0, int(1e4))
    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
