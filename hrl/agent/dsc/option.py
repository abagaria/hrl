import numpy as np
from copy import deepcopy
from collections import deque
from scipy.spatial import distance

from hrl.agent.rainbow.rainbow import Rainbow
from .classifier.position_classifier import PositionInitiationClassifier


class ModelFreeOption(object):
    def __init__(self, *, name, option_idx, parent, env, global_solver, global_init,
                 buffer_length, gestation_period, timeout, gpu_id,
                 init_salient_event, target_salient_event, n_training_steps):
        self.env = env
        self.name = name
        self.parent = parent
        self.gpu_id = gpu_id
        self.timeout = timeout
        self.global_solver = global_solver
        self.n_training_steps = n_training_steps
        
        self.global_init = global_init
        self.buffer_length = buffer_length

        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event

        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.initiation_classifier = PositionInitiationClassifier()
        self.solver = self._get_model_free_solver()

        self.children = []
        self.success_curve = []

        print(f"Created model-free option {self.name} with option_idx={self.option_idx}")

        self.is_last_option = self.option_idx == 5

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

    # ------------------------------------------------------------
    # Learning Phase Methods
    # ------------------------------------------------------------

    def get_training_phase(self):
        if self.num_goal_hits < self.gestation_period:
            return "gestation"
        return "initiation_done"

    def is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True
        
        if self.is_last_option and self.init_salient_event(state):
            return True
        
        return self.initiation_classifier.optimistic_predict(state) \
            or self.pessimistic_is_init_true(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        return self.initiation_classifier.pessimistic_predict(state)

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state)

        return self.parent.pessimistic_is_init_true(state)

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

    def is_at_local_goal(self, pos, goal_pos):
        _, reached = self.rf(pos, goal_pos)
        return reached and self.is_term_true(pos)

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
        return sampled_goal.obs, sampled_goal.pos

    def rollout(self, start_state, start_position, eval_mode=False):
        """ Main option control loop. """
        assert self.is_init_true(start_position)

        info = {}
        done = False
        reset = False
        reached = False
        
        num_steps = 0
        total_reward = 0
        option_transitions = []
        self.num_executions += 1

        state = deepcopy(start_state)
        pos = deepcopy(start_position)

        visited_positions = [pos]
        visited_states = [start_state]

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
                                      pos)
            )

            state = next_state

        reached_term = self.is_term_true(pos)
        self.success_curve.append(reached_term)

        if reached_term and not eval_mode:
            self.num_goal_hits += 1
            print(f"{self.name} reached term set {self.num_goal_hits} times.")

        if not eval_mode:
            self.solver.her(option_transitions, visited_positions, goal, goal_pos)

            if not self.global_init:
                self.derive_training_examples(visited_states, visited_positions, reached_term)
                self.initiation_classifier.fit_initiation_classifier()

        return state, done, reset, len(option_transitions)

    def derive_training_examples(self, visited_states, visited_positions, reached_term):
        assert len(visited_states) == len(visited_positions)

        start_state = visited_states[0]
        start_position = visited_positions[0]

        if reached_term:
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            positive_positions = [start_position] + visited_positions[-self.buffer_length:]
            
            self.initiation_classifier.add_positive_examples(positive_states, positive_positions)

            # if self.init_salient_event(start_position):
            #     self.is_last_option = True
        else:
            negative_states = [start_state]
            negative_positions = [start_position]

            self.initiation_classifier.add_negative_examples(negative_states, negative_positions)

    # ------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------

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
