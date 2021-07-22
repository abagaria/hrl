from copy import deepcopy
from hrl.mdp.SalientEventClass import SalientEvent
import numpy as np
from gym import Wrapper


class GoalConditionedMDPWrapper(Wrapper):
    """
    this abstract class is a wrapper that represents a goal conditioned MDP
    user must specify a start and goal state, and a goal tolerance that represents
    an ball around the goal state.
    All the methods in the class should be batched
    """
    def __init__(self, env, start_state, goal_state, goal_tolerance=0.6):
        super().__init__(env)
        self.env = env
        self.start_state = start_state
        self.goal_state = goal_state
        self.salient_positions = [goal_state]
        self.goal_tolerance = np.asarray(goal_tolerance)

        # set initial states
        self.cur_state = self.reset()
        self.cur_done = False
    
    def get_original_target_events(self):
        """
        return the original salient events
        """
        saleint_events = [SalientEvent(pos, event_idx=i+1) for i, pos in enumerate(self.salient_positions)]
        return saleint_events
    
    def sparse_gc_reward_func(self, states, goals):
        """
        always overwrite this function to provide the sparse reward func
        """
        pass

    def dense_gc_reward_func(self, states, goals):
        """
        always overwrite this function to provide the dense reward func
        """
        pass

    def reset(self):
        self.init_state = self.env.reset()
        self.cur_state = deepcopy(self.init_state)
        self.cur_done = False
        return self.init_state
    
    def step(self, action):
        """
        overwrite the step function for gc MDP.
        """
        next_state, reward, done, info = self.env.step(action)
        self.cur_state = next_state
        self.cur_done = done
        return next_state, reward, done, info
    
    def is_start_region(self, states):
        """
        given a batch of states, return a boolean array indicating whether states are in start region
        always overwrite this function
        """
        pass

    def is_goal_region(self, states):
        """
        given a batch of states, return a boolean array indicating whether states are in goal region
        always overwrite this function
        """
        pass

    def extract_features_for_initiation_classifier(self, states):
        """
        take as input a batch of `states` of shape `N x D` and return the state 
        dimensions relevant for learning the initiation set classifier 
        (shape `N x K`; e.g, K=2 for navigation).
        always overwrite this function
        """
        pass
