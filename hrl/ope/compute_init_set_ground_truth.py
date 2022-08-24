import gym
import argparse

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
import d4rl
from hrl.ope.utils import load_chain

parser = argparse.ArgumentParser()
parser.add_argument("--base_fname", type=str, required=True)
args = parser.parse_args()

global_option, chain = load_chain(args.base_fname)

subgoal = global_option.get_goal_for_rollout()
option_transitions, total_reward = global_option.rollout(step_number=0, goal=subgoal)