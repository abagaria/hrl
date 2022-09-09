import gym
import argparse
import numpy as np
import pickle
from copy import deepcopy

from hrl.wrappers.antmaze_wrapper import D4RLAntMazeWrapper
import d4rl
from hrl.ope.utils import load_chain

parser = argparse.ArgumentParser()
parser.add_argument("--base_fname", type=str, required=True)
args = parser.parse_args()

global_option, chain = load_chain(args.base_fname)
test_option = chain[1]

def in_domain(s):
    return not ((s[0] < 6) and (2 < s[1] < 6))

elements = 20
ss = np.linspace(-2, 10, elements)
truth_mat = - np.ones((elements, elements))
classifier_mat = - np.ones((elements, elements))
for i0, s0 in enumerate(ss):
    for i1, s1 in enumerate(ss):
        pos = np.array([s0, s1])
        if in_domain(pos):
            full_state = deepcopy(test_option.mdp.cur_state)
            full_state[:2] = pos
            classifier_mat[i0, i1] = test_option.is_init_true(pos)
            subgoal = test_option.get_goal_for_rollout()
            option_transitions, total_reward = test_option.rollout(
                step_number=0,
                goal=subgoal,
                initial_state_xy=np.array(pos)
            )
            truth_mat[i0, i1] = int(total_reward > -200)
    truth_mat_fname = args.base_fname + '_truth_mat_just_pos.pkl'
    with open(truth_mat_fname, 'wb') as f:
        pickle.dump(truth_mat, f)
    classifier_mat_fname = args.base_fname + '_classifier_mat_just_pos.pkl'
    with open(classifier_mat_fname, 'wb') as f:
        pickle.dump(classifier_mat, f)


# num_tries = 100
# successes = 0
# initial_state_xy = test_option.initiation_classifier.sample()
# initial_state_xy = test_option.extract_goal_dimensions(initial_state_xy)
# for i in range(num_tries):
#     subgoal = test_option.get_goal_for_rollout()
#     # initial_state_xy = test_option.initiation_classifier.sample()
#     # initial_state_xy = test_option.extract_goal_dimensions(initial_state_xy)
#     option_transitions, total_reward = test_option.rollout(
#         step_number=0,
#         goal=subgoal,
#         initial_state_xy=np.array(initial_state_xy)
#     )
#     print(total_reward)
#     if total_reward > -200:
#         successes += 1
# print("Success proba: {}".format(successes/num_tries))