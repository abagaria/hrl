import os
import json
import time
import ipdb
import pfrl
import torch
import pickle
import random
import argparse

from pfrl.wrappers import atari_wrappers
from hrl.utils import create_log_dir
from hrl.agent.dsc.dsc import RobustDSC
from hrl.agent.dsg.dsg import SkillGraphAgent
from hrl.agent.dsg.trainer import DSGTrainer
from hrl.salient_event.salient_event import SalientEvent
from hrl.montezuma.info_wrapper import MontezumaInfoWrapper, Reshape


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
    if isinstance(goals, (list, tuple)):
        goal = random.choice(goals)
    else:
        goal = goals
    if hasattr(goal, "frame"):
        return goal.frame
    if isinstance(goal, atari_wrappers.LazyFrames):
        return goal
    return goal.obs


def get_exploration_agent(rnd_base_dir):
    """ Return the exploration runner from BBE. """ 
    from dopamine.discrete_domains import run_experiment
    from hrl.agent.bonus_based_exploration.run_experiment import create_exploration_runner as create_runner
    from hrl.agent.bonus_based_exploration.run_experiment import create_exploration_agent as create_agent

    _gin_files = [
        os.path.expanduser("~/git-repos/hrl/hrl/agent/bonus_based_exploration/configs/rainbow_rnd.gin")
    ]

    run_experiment.load_gin_configs(_gin_files, [])
    return create_runner(rnd_base_dir, create_agent, schedule='episode_wise')


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
    parser.add_argument("--max_frames_per_episode", type=int, default=30*60*60)  # 30 mins
    parser.add_argument("--p_her", type=float, default=1.)
    parser.add_argument("--goal_selection_criterion", type=str, default="random")

    parser.add_argument("--use_rf_on_pos_traj", action="store_true", default=False)
    parser.add_argument("--use_rf_on_neg_traj", action="store_true", default=False)
    parser.add_argument("--replay_original_goal_on_pos", action="store_true", default=False)

    parser.add_argument("--distance_metric", type=str, default="euclidean")
    parser.add_argument("--n_kmeans_clusters", type=int, default=99)
    parser.add_argument("--n_sift_keypoints", type=int, default=30)
    parser.add_argument("--enable_rnd_logging", action="store_true", default=False)
    parser.add_argument("--disable_graph_expansion", action="store_true", default=False)
    parser.add_argument("--purpose", type=str, default="", help="Optional notes about the current experiment")

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

    _rnd_base_dir = f"logs/{args.experiment_name}/{args.seed}/"
    _rnd_log_file = os.path.join(_rnd_base_dir, "rnd_log.pkl")

    exploration_agent = get_exploration_agent(_rnd_base_dir)
    
    env = MontezumaInfoWrapper(
        atari_wrappers.FrameStack(
            Reshape(
                pfrl.wrappers.ContinuingTimeLimit(
                    exploration_agent._environment, args.max_frames_per_episode
                ),
                channel_order="chw"
            ),
            k=4, channel_order="chw"   
        )
    )

    s0, _ = env.reset()
    p0 = env.get_current_position()

    goal_dir_path = os.path.join(os.path.expanduser("~"), "git-repos/hrl/logs/goal_states")
    
    gpos = (123, 148)
    gpos1 = (132, 192)
    gpos2 = (24, 235)
    gpos3 = (130, 235)
    gpos4 = (77, 192)
    gpos5 = (23, 148)
    
    g0 = load_goal_state(goal_dir_path, file="bottom_right_states.pkl")
    g1 = load_goal_state(goal_dir_path, file="top_bottom_right_ladder_states.pkl")
    g2 = load_goal_state(goal_dir_path, file="left_door_goal.pkl")
    g3 = load_goal_state(goal_dir_path, file="right_door_goal.pkl")
    g4 = load_goal_state(goal_dir_path, file="bottom_mid_ladder_goal.pkl")
    g5 = load_goal_state(goal_dir_path, file="bottom_left_goal.pkl")

    pfrl.utils.set_random_seed(args.seed)

    beta0 = SalientEvent(s0, p0, tol=2.)
    beta1 = SalientEvent(g0, gpos, tol=2.)
    beta2 = SalientEvent(g1, gpos1, tol=2.)
    beta3 = SalientEvent(g2, gpos2, tol=2.)
    beta4 = SalientEvent(g3, gpos3, tol=2.)
    beta5 = SalientEvent(g4, gpos4, tol=2.)
    beta6 = SalientEvent(g5, gpos5, tol=2.)

    dsc_agent = RobustDSC(env,
                          args.gestation_period,
                          args.buffer_length,
                          args.experiment_name,
                          args.gpu_id,
                          beta0,
                          args.use_oracle_rf,
                          args.use_rf_on_pos_traj,
                          args.use_rf_on_neg_traj,
                          args.replay_original_goal_on_pos,
                          args.use_pos_for_init,
                          args.p_her,
                          args.max_num_options,
                          args.seed,
                          _log_file,
                          args.n_kmeans_clusters,
                          args.n_sift_keypoints)

    dsg_agent = SkillGraphAgent(dsc_agent, exploration_agent, args.distance_metric)
    
    trainer = DSGTrainer(env, dsc_agent, dsg_agent, exploration_agent,
                         50, 10, 
                         _rnd_log_file,
                         args.goal_selection_criterion,
                         [beta1, beta2, beta3, beta4, beta5, beta6],
                         args.enable_rnd_logging)

    print(f"[Seed={args.seed}] Device count: {torch.cuda.device_count()} Device Name: {torch.cuda.get_device_name(0)}")
    
    t0 = time.time()
    trainer.run_loop(0, int(1e5))
    print(f"Finished after {(time.time() - t0) / 3600.} hrs")
