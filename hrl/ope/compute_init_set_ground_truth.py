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

test_option = chain[1]
num_tries = 400
successes = 0
for i in range(num_tries):
    subgoal = test_option.get_goal_for_rollout()
    initial_state_xy = test_option.initiation_classifier.sample()
    initial_state_xy= test_option.extract_goal_dimensions(initial_state_xy)
    option_transitions, total_reward = test_option.rollout(
        step_number=0,
        goal=subgoal,
        initial_state_xy=np.array(initial_state_xy)
    )
    if total_reward > -200:
        successes += 1
print("Success proba: {}".format(successes/num_tries))