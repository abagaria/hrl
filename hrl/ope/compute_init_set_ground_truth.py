import gym
import argparse
import numpy as np

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
import d4rl
from hrl.ope.utils import load_chain

parser = argparse.ArgumentParser()
parser.add_argument("--base_fname", type=str, required=True)
args = parser.parse_args()

global_option, chain = load_chain(args.base_fname)

for i in range(10):
    subgoal = chain[0].get_goal_for_rollout()
    option_transitions, total_reward = chain[0].rollout(
        step_number=0,
        goal=subgoal,
        initial_state_xy=np.array([1.0, 9.0]))
    print(total_reward)