import ipdb
import numpy as np
from copy import deepcopy
from collections import deque
from scipy.spatial import distance
from pfrl.wrappers import atari_wrappers

from hrl.agent.rainbow.rainbow import Rainbow
from .classifier.position_classifier import PositionInitiationClassifier
from .classifier.image_classifier import ImageInitiationClassifier
from hrl.salient_event.salient_event import SalientEvent
from .datastructures import TrainingExample


class ModelFreeOption(object):
    def __init__(self, *, name, option_idx, parent, env, global_solver, global_init,
                 buffer_length, gestation_period, timeout, gpu_id,
                 init_salient_event, target_salient_event, n_training_steps,
                 gamma, use_oracle_rf, max_num_options, use_pos_for_init, chain_id):
        self.env = env  # TODO: remove as class var and input to rollout()
        self.name = name
        self.gamma = gamma
        self.parent = parent
        self.gpu_id = gpu_id
        self.timeout = timeout
        self.chain_id = chain_id
        self.global_solver = global_solver
        self.n_training_steps = n_training_steps
        self.use_oracle_rf = use_oracle_rf
        self.use_pos_for_init = use_pos_for_init
        
        self.global_init = global_init
        self.buffer_length = buffer_length

        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event

        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.initiation_classifier = self._get_initiation_classifier()
        self.solver = self._get_model_free_solver()

        self.children = []
        self.success_curve = []
        self.effect_set = deque([], maxlen=50)

        print(f"Created model-free option {self.name} with option_idx={self.option_idx}")

        self.is_last_option = self.option_idx == max_num_options

    def _get_model_free_solver(self):
        if self.global_init:
            return Rainbow(n_actions=self.env.action_space.n,
                            n_atoms=51,
                            v_min=-10.,
                            v_max=+10.,
                            noisy_net_sigma=0.5,
                            lr=6.25e-5,
                            n_steps=3,
                            betasteps=self.n_training_steps / 4,
                            replay_start_size=80_000,
                            replay_buffer_size=(10**6) // 2,
                            gpu=self.gpu_id,
                            use_her=True,
                            goal_conditioned=True,
                    )

        return self.global_solver

    def _get_initiation_classifier(self):
        if self.use_pos_for_init:
            return PositionInitiationClassifier()
        return ImageInitiationClassifier(gamma=self.gamma)

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def get_training_phase(self):
        if self.num_goal_hits < self.gestation_period:
            return "gestation"
        return "initiation_done"

    def is_init_true(self, state, info):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        pos = np.array([info["player_x"], info["player_y"]])
        
        if self.is_last_option and self.init_salient_event(pos):
            return True

        x = self.extract_init_features(state, info)
        
        return self.initiation_classifier.optimistic_predict(x) \
            or self.pessimistic_is_init_true(state, info)

    def pessimistic_is_init_true(self, state, info):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        x = self.extract_init_features(state, info)
        return self.initiation_classifier.pessimistic_predict(x)

    def is_term_true(self, state, info):
        if self.parent is None:
            pos = np.array([info["player_x"], info["player_y"]])
            return self.target_salient_event(pos)

        if info["falling"] or info["dead"]:
            return False

        return self.parent.pessimistic_is_init_true(state,  info)

    def extract_init_features(self, state, info):
        if self.use_pos_for_init:
            return np.array([info["player_x"], info["player_y"]])
        
        if isinstance(state, atari_wrappers.LazyFrames):
            return state._frames[-1]

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def rf(self, pos, goal_pos):
        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and \
                   abs(pos1[1] - pos2[1]) <= tol
        
        reached = is_close(pos, goal_pos, tol=2.)
        reward = float(reached)

        return reward, reached

    def act(self, state, goal):
        assert isinstance(self.solver, Rainbow), f"{type(self.solver)}"
        augmented_state = self.solver.get_augmented_state(state, goal)
        return self.solver.act(augmented_state)

    def get_goal_for_rollout(self):
        """ Sample goal to pursue for option rollout. """

        if self.parent is None and self.target_salient_event is not None:
            return self.target_salient_event.get_target_obs(), \
                   self.target_salient_event.get_target_position()

        sampled_goal = self.parent.initiation_classifier.sample()

        # TODO: Deal with the case where sampled_goal is None

        return sampled_goal.obs, sampled_goal.pos

    def rollout(self, start_state, info, dsc_goal_salient_event, eval_mode=False):
        """ Main option control loop. """
        start_position = info["player_x"], info["player_y"]
        assert self.is_init_true(start_state, info)

        done = False
        reset = False
        reached = False
        
        num_steps = 0
        total_reward = 0
        option_transitions = []
        self.num_executions += 1

        state = deepcopy(start_state)  # TODO: Do we need to deepcopy?
        pos = deepcopy(start_position)

        visited_positions = [pos]
        visited_states = [start_state]

        goal, goal_pos = (dsc_goal_salient_event.target_obs, dsc_goal_salient_event.target_pos) 
        
        if not self.global_init:
            goal, goal_pos = self.get_goal_for_rollout()

        print(f"Rolling out {self.name}, from {start_position} targeting {goal_pos}")

        while not done and not reached and not reset and num_steps < self.timeout:

            action = self.act(state, goal)
            next_state, reward, done, info = self.env.step(action)

            reset = info.get("needs_reset", False)
            pos = info["player_x"], info["player_y"]
            reward, reached = self.rf(pos, goal_pos)

            num_steps += 1
            total_reward += reward
            visited_positions.append(pos)
            visited_states.append(next_state)

            option_transitions.append(
                                      (state,
                                      action, 
                                      np.sign(reward), 
                                      next_state, 
                                      done or reached, 
                                      reset,   # TODO: Think about done and reset for options
                                      info)
            )

            # Truncate initiation trajectories around death transitions
            if info["dead"] and not self.global_init:
                self.derive_training_examples(visited_states,
                                              visited_positions,
                                              reached_term=False)

                visited_states = []
                visited_positions = []

            state = next_state

        reached_term = self.is_term_true(state, info) if not self.global_init else dsc_goal_salient_event(pos)
        self.success_curve.append(reached_term)

        if not eval_mode:
            self.update_option_after_rollout(state, info, goal, option_transitions, 
                                             visited_states, visited_positions, reached_term)

        return state, done, reset, visited_positions, goal_pos, info

    def update_option_after_rollout(self, state, info, goal, option_transitions,
                                    visited_states, visited_positions, reached_term):
        """ After rolling out an option policy, update its effect set, policy and initiation classifier. """

        if reached_term:
            self.num_goal_hits += 1
            self.add_to_effect_set(state, info)
            print(f"{self.name} reached term set {self.num_goal_hits} times.")

            if self.parent is None and self.target_salient_event is not None:
                assert isinstance(self.target_salient_event, SalientEvent)
                self.target_salient_event.add_to_effect_set(state, info)

        assert not self.use_oracle_rf, "Deprecated"
        self.no_rf_update(option_transitions, goal, reached_term)

        if not self.global_init and len(visited_states) > 0:
            self.derive_training_examples(visited_states, visited_positions, reached_term)
        
        if not self.global_init:
            self.initiation_classifier.fit_initiation_classifier()

    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def no_rf_update(self, transitions, pursued_goal, reached_termination_region):
        """ Hindsight experience replay without requiring an oracle reward function. """

        if reached_termination_region:
            final_transition = transitions[-1]
            reached_goal = final_transition[3]
            relabeled_trajectory = self.positive_relabel(transitions)
            self.experience_replay(relabeled_trajectory, reached_goal)
        else:
            self.experience_replay(transitions, pursued_goal)
            
            hindsight_goal, hindsight_goal_idx = self.solver.pick_hindsight_goal(transitions)
            hindsight_trajectory = transitions[:hindsight_goal_idx]

            if len(hindsight_trajectory) > 0:  # TODO: Remove cause its ugly
                relabeled_trajectory = self.positive_relabel(hindsight_trajectory)
                self.experience_replay(relabeled_trajectory, hindsight_goal)

    def positive_relabel(self, trajectory):
        """ Relabel the final transition in the trajectory as a positive goal transition. """ 
        original_transition = trajectory[-1]
        trajectory[-1] = original_transition[0], original_transition[1], +1., \
                         original_transition[3], True, \
                         original_transition[5], original_transition[6]
        return trajectory

    def experience_replay(self, trajectory, goal):
        for state, action, reward, next_state, done, reset, _ in trajectory:
            augmented_state = self.solver.get_augmented_state(state, goal)
            augmented_next_state = self.solver.get_augmented_state(next_state, goal)
            self.solver.step(augmented_state, action, deepcopy(reward), augmented_next_state, deepcopy(done), reset)
    
    # ------------------------------------------------------------
    # Learning initiation classifiers
    # ------------------------------------------------------------

    def derive_training_examples(self, visited_states, visited_positions, reached_term):
        assert len(visited_states) == len(visited_positions)

        start_state = visited_states[0]
        start_position = visited_positions[0]

        if reached_term:
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            positive_positions = [start_position] + visited_positions[-self.buffer_length:]
            
            self.initiation_classifier.add_positive_examples(positive_states, positive_positions)
        else:
            negative_states = [start_state]
            negative_positions = [start_position]

            self.initiation_classifier.add_negative_examples(negative_states, negative_positions)

    # ------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------

    def value_function(self, obs, goals):
        """ Value function for a single observation and a set of goals. """
        augmented_states = [self.solver.get_augmented_state(obs, goal) for goal in goals]
        return self.solver.value_function(augmented_states)

    def distance_to_state(self, pos, metric="euclidean"):
        """ Compute the distance between the current option and the input `state`. """
        assert metric in ("euclidean", "value"), metric

        if metric == "euclidean":
            return self._euclidean_distance_to_state(pos)
        
        raise NotImplementedError("VF based distances.")

    def _euclidean_distance_to_state(self, point):
        assert isinstance(point, np.ndarray)
        assert point.shape == (2,), point.shape

        positive_point_array = self.initiation_classifier.get_states_inside_pessimistic_classifier_region()

        distances = distance.cdist(point[None, :], positive_point_array)
        return np.median(distances)

    def _value_distance_to_state(self, state):
        features = state.features() if not isinstance(state, np.ndarray) else state
        goals = self.initiation_classifier.get_states_inside_pessimistic_classifier_region()

        distances = self.value_function(features, goals)
        distances[distances > 0] = 0.
        return np.median(np.abs(distances))

    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

    def add_to_effect_set(self, obs, info):
        self.effect_set.append(TrainingExample(obs, info))

    def get_effective_effect_set(self):  # TODO: batched version
        """ Return the subset of the effect set still in the termination region. """
        if self.global_init:
            ipdb.set_trace()
        return [eg for eg in self.effect_set if self.is_term_true(eg.obs, eg.info)]

    def get_sibling_options(self):
        if self.parent is not None:
            return [option for option in self.parent.children if option != self]
        return []

    def get_option_success_rate(self):
        if self.num_executions > 0:
            return self.num_goal_hits / self.num_executions
        return 1.

    def get_success_rate(self):
        if len(self.success_curve) == 0:
            return 0.
        return np.mean(self.success_curve)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, ModelFreeOption):
            return self.name == other.name
        return False
    
    def __hash__(self):
        return hash(self.name)
