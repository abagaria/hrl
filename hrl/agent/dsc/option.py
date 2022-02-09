import ipdb
import torch
import random
import numpy as np
from copy import deepcopy
from collections import deque
from scipy.spatial import distance
from pfrl.wrappers import atari_wrappers

from . import utils
from .datastructures import TrainingExample
from hrl.agent.rainbow.rainbow import Rainbow
from hrl.salient_event.salient_event import SalientEvent
from .classifier.sift_classifier import SiftInitiationClassifier
from .classifier.position_classifier import PositionInitiationClassifier
from .classifier.fixed_conv_classifier import FixedConvInitiationClassifier
from .classifier.single_conv_init_classifier import SingleConvInitiationClassifier
from .classifier.double_conv_init_classifier import DoubleConvInitiationClassifier


class ModelFreeOption(object):
    def __init__(self, *, name, option_idx, parent, env, global_solver, global_init,
                 buffer_length, gestation_period, timeout, gpu_id,
                 init_salient_event, target_salient_event, n_training_steps,
                 use_oracle_rf, use_rf_on_pos_traj, use_rf_on_neg_traj,
                 replay_original_goal_on_pos,
                 max_num_options, use_pos_for_init, chain_id,
                 p_her, num_kmeans_clusters, sift_threshold,
                 classifier_type, use_full_neg_traj, use_pessimistic_relabel):
        self.env = env  # TODO: remove as class var and input to rollout()
        self.name = name
        self.parent = parent
        self.gpu_id = gpu_id
        self.timeout = timeout
        self.chain_id = chain_id
        self.global_solver = global_solver
        self.n_training_steps = n_training_steps
        self.use_pos_for_init = use_pos_for_init
        self.p_her = p_her
        self.classifier_type = classifier_type
        
        self.use_oracle_rf = use_oracle_rf
        self.use_rf_on_pos_traj = use_rf_on_pos_traj
        self.use_rf_on_neg_traj = use_rf_on_neg_traj
        self.replay_original_goal_on_pos = replay_original_goal_on_pos
        
        self.global_init = global_init
        self.buffer_length = buffer_length

        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event

        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        self.num_kmeans_clusters = num_kmeans_clusters
        self.sift_threshold = sift_threshold
        self.use_full_neg_traj = use_full_neg_traj
        self.use_pessimistic_relabel = use_pessimistic_relabel

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
                            goal_conditioned=True,
                    )

        return self.global_solver

    def _get_initiation_classifier(self):
        if self.use_pos_for_init:
            return PositionInitiationClassifier()
        if self.classifier_type == "sift":
            return SiftInitiationClassifier(
                num_clusters=self.num_kmeans_clusters,
                sift_threshold=self.sift_threshold,
            )
        device = torch.device(
            f"cuda:{self.gpu_id}" if self.gpu_id > -1 else "cpu"
        )
        print(f"Creating classifier of type {self.classifier_type}")
        if self.classifier_type == "fixed-cnn":
            return FixedConvInitiationClassifier(
                device,
                gamma=6e-7,
                nu=2.5e-4
            )
        if self.classifier_type == "single-cnn":
            return SingleConvInitiationClassifier(device)
        if self.classifier_type == "double-cnn":
            return DoubleConvInitiationClassifier(device,
                pessimistic_relabel=self.pessimistic_relabel)
        if self.classifier_type == "epistemic-cnn":
            return
        raise NotImplementedError(self.classifier_type)

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
        
        if self.is_last_option and self.init_salient_event(info):
            return True

        if not self.initiation_classifier.is_initialized():
            return True

        x = self.extract_init_features(state, info)
        
        return self.initiation_classifier.optimistic_predict(x) \
            or self.pessimistic_is_init_true(state, info)

    def pessimistic_is_init_true(self, state, info):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        if not self.initiation_classifier.is_initialized():
            return False

        x = self.extract_init_features(state, info)
        return self.initiation_classifier.pessimistic_predict(x)

    def is_term_true(self, state, info):
        if self.failure_condition(info, check_falling=True):
            return False

        if self.parent is None:
            return self.target_salient_event(info)

        return self.parent.pessimistic_is_init_true(state,  info)

    def extract_init_features(self, state, info):
        if self.use_pos_for_init:
            return np.array([info["player_x"], info["player_y"]])
        
        if isinstance(state, atari_wrappers.LazyFrames):
            return state._frames[-1].squeeze()
    
    def failure_condition(self, info, check_falling=False):
        targets_start_state = self.target_salient_event.target_pos[0] == 77.\
                          and self.target_salient_event.target_pos[1] == 235.
        death_cond = (info['falling'] or info['dead']) if check_falling else info['dead']
        return death_cond and not targets_start_state 

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def local_rf(self, state, info, goal_pos, salient_event=None):
        if self.use_oracle_rf or self.use_rf_on_pos_traj:
            pos = utils.info_to_pos(info)
            return self.rf(pos, goal_pos)

        if self.global_init:
            reached = salient_event(info)
        else:
            reached = self.is_term_true(state, info)

        reward = float(reached)
        return reward, reached

    def rf(self, pos, goal_pos):
        """ Oracle position based reward function. """
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
            assert isinstance(self.target_salient_event, SalientEvent)
            
            if self.use_oracle_rf or self.use_rf_on_pos_traj:
                return self.target_salient_event.target_obs, \
                       self.target_salient_event.target_pos

            return self.target_salient_event.sample()

        sampled_goal = self.parent.initiation_classifier.sample()

        if sampled_goal is not None:
            assert isinstance(sampled_goal, TrainingExample), sampled_goal
            return sampled_goal.obs, sampled_goal.info

        # TODO: Recursively sample from initiation classifiers up the chain
        if self.parent.parent is not None:
            sampled_goal = self.parent.parent.initiation_classifier.sample()
            if sampled_goal is not None:
                assert isinstance(sampled_goal, TrainingExample), sampled_goal
                return sampled_goal.obs, sampled_goal.info

        assert isinstance(self.target_salient_event, SalientEvent)
        return self.target_salient_event.target_obs, self.target_salient_event.target_info

    def rollout(self, start_state, info, dsc_goal_salient_event, eval_mode=False):
        """ Main option control loop. """

        done = False
        reset = False
        reached = False
        
        num_steps = 0
        total_reward = 0
        option_transitions = []
        self.num_executions += 1

        state = start_state
        visited_infos = [info]
        visited_states = [start_state]

        goal, goal_info = (dsc_goal_salient_event.target_obs, dsc_goal_salient_event.target_info) 
        
        if not self.global_init:
            goal, goal_info = self.get_goal_for_rollout()

        print(f"Rolling out {self.name}, from {utils.info_to_pos(info)} targeting {goal_info}")

        while not done and not reached and not reset and num_steps < self.timeout:

            action = self.act(state, goal)
            next_state, reward, done, info = self.env.step(action)
            reset = info.get("needs_reset", False)
            
            reward, reached = self.local_rf(
                next_state, info, utils.info_to_pos(goal_info), dsc_goal_salient_event
            )

            num_steps += 1
            total_reward += reward
            visited_infos.append(info)
            visited_states.append(next_state)

            option_transitions.append(
                                      (state,
                                      action, 
                                      np.sign(reward), 
                                      next_state, 
                                      done or reached, 
                                      reset,
                                      info)
            )

            # Truncate initiation trajectories around death transitions
            if (not self.global_init) and self.failure_condition(info, check_falling=False):
                self.derive_training_examples(visited_states,
                                              visited_infos,
                                              reached_term=False)

                visited_infos = []
                visited_states = []

            state = next_state

        self.success_curve.append(reached)

        if not eval_mode:
            self.update_option_after_rollout(state, info, goal, goal_info, option_transitions, 
                                             visited_states, visited_infos, reached)
            print(f"Updated {self.name} on {len(visited_infos)} transitions")

        return state, done, reset, visited_infos, goal_info, info

    def update_option_after_rollout(self, state, info, goal, goal_info,
                                    option_transitions, visited_states, visited_infos, reached_term):
        """ After rolling out an option policy, update its effect set, policy and initiation classifier. """

        if reached_term:
            self.num_goal_hits += 1
            self.add_to_effect_set(state, info)
            print(f"{self.name} reached term set {self.num_goal_hits} times.")

            if self.parent is None and self.target_salient_event is not None:
                assert isinstance(self.target_salient_event, SalientEvent)
                self.target_salient_event.add_to_effect_set(state, info)

        if self.use_oracle_rf:
            option_positions = [utils.info_to_pos(trans[-1]) for trans in option_transitions]
            self.solver.her(option_transitions, option_positions, goal, goal_info)  # TODO: Update her to use goal_info
        else:
            reached_goal_info = info
            self.no_rf_update(option_transitions, goal, goal_info, reached_goal_info, reached_term)

        if not self.global_init and len(visited_states) > 0:
            self.derive_training_examples(visited_states, visited_infos, reached_term)
        
        if not self.global_init:
            self.initiation_classifier.fit_initiation_classifier()

    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def no_rf_update(self, transitions, pursued_goal, pursued_goal_info,
                    reached_goal_info, reached_termination_region):
        """ Hindsight experience replay without requiring an oracle reward function. """

        original_rewards = [trans[2] for trans in transitions]

        if reached_termination_region:
            final_transition = transitions[-1]
            reached_goal = final_transition[3]

            assert np.isclose(
                utils.info_to_pos(final_transition[-1]),
                utils.info_to_pos(reached_goal_info)
            ).all()

            relabeled_trajectory = self.positive_relabel(transitions)
            self.experience_replay(relabeled_trajectory, reached_goal)

            # Sanity check rewards along the positive trajectory
            relabeled_rewards = [trans[2] for trans in relabeled_trajectory]
            assert np.isclose(sum(original_rewards), 1.), original_rewards
            assert np.isclose(sum(relabeled_rewards), 1.), relabeled_rewards

            if self.replay_original_goal_on_pos:
                relabeled_transitions = self.relabel_pos_trajectory_original_goal(
                    transitions, pursued_goal_info, reached_goal_info
                )

                if relabeled_transitions is not None:
                    self.experience_replay(relabeled_transitions, pursued_goal)
        else:
            # Sanity check rewards along negative trajectory
            assert np.isclose(sum(original_rewards), 0.), f"{self, original_rewards}"
            self.experience_replay(transitions, pursued_goal)

            # HER on the negative trajectory
            self.negative_trajectory_her(transitions)
    
    def negative_trajectory_her(self, transitions):
        hindsight_goal, hindsight_goal_idx = self.solver.pick_hindsight_goal(transitions)
        hindsight_trajectory = transitions[:hindsight_goal_idx+1]

        if len(hindsight_trajectory) > 0:

            if self.use_rf_on_neg_traj:
                print("[-trajReplay] Replaying negative trajectory with oracle-rf")
                visited_positions = [utils.info_to_pos(trans[-1]) for trans in transitions]
                hindsight_goal_pos = visited_positions[hindsight_goal_idx]  
                relabeled_trajectory = self.negative_oracle_relabel(hindsight_trajectory, hindsight_goal_pos)
            else:
                print("[-trajReplay] Replaying negative trajectory with positive relabel")
                relabeled_trajectory = self.positive_relabel(hindsight_trajectory)
            
            self.experience_replay(relabeled_trajectory, hindsight_goal)

    def positive_relabel(self, trajectory):
        """ Relabel the final transition in the trajectory as a positive goal transition. """ 
        original_transition = trajectory[-1]
        trajectory[-1] = original_transition[0], original_transition[1], +1., \
                         original_transition[3], True, \
                         original_transition[5], original_transition[6]
        
        # Sanity check -- since we are only using this function on positive 
        # trajectories for now, only the final transition should be positive
        rewards = [transition[2] for transition in trajectory]
        assert np.isclose(sum(rewards), 1), rewards

        return trajectory

    def negative_relabel(self, trajectory):
        relabeled_trajectory = []
        for state, action, _, next_state, done, reset, info in trajectory:
            relabeled_trajectory.append((
                state, action, 0., next_state, done, reset, info
            ))

        # Sanity check
        rewards = [transition[2] for transition in relabeled_trajectory]
        if not np.isclose(sum(rewards), 0): ipdb.set_trace()

        return relabeled_trajectory

    def oracle_relabel_pos_trajectory_original_goal(self,
                                                    trajectory,
                                                    pursued_goal_pos,
                                                    reached_goal_pos):
        final_transition = trajectory[-1]
        final_pos = utils.info_to_pos(final_transition[-1])
        assert np.isclose(final_pos, reached_goal_pos).all(), f"{final_pos, reached_goal_pos}"
        
        if self.rf(final_pos, pursued_goal_pos)[1]:
            print("[+trajReplay] Replaying positive trajectory with *positive* original goal")
            relabeled_transitions = self.positive_relabel(trajectory)
            assert np.isclose(sum([x[2] for x in relabeled_transitions]), 1.)
            return relabeled_transitions
        
        print("[+trajReplay] Replaying positive trajectory with *negative* original goal")
        relabeled_transitions = self.negative_relabel(trajectory)
        assert np.isclose(sum([x[2] for x in relabeled_transitions]), 0.)
        return relabeled_transitions

    def relabel_pos_trajectory_original_goal(self, 
                                             trajectory, 
                                             pursued_goal_info,
                                             reached_goal_info):
        """ If the pursued goal was from a salient event and we reached that salient event, 
            we can assume that we were epsilon-close to the pursued goal. """
        final_transition = trajectory[-1]
        final_pos = utils.info_to_pos(final_transition[-1])
        
        assert np.isclose(
            final_pos, 
            utils.info_to_pos(reached_goal_info)
        ).all()
        
        if self.target_salient_event is not None and self.target_salient_event(pursued_goal_info):
            print("[+SalientEventTrajReplay] Replaying positive trajectory with *positive* original goal")
            relabeled_transitions = self.positive_relabel(trajectory)
            assert np.isclose(sum([x[2] for x in relabeled_transitions]), 1.)
            return relabeled_transitions

    def negative_oracle_relabel(self, trajectory, goal_pos):
        relabeled_trajectory = []
        for state, action, _, next_state, done, reset, info in trajectory:
            pos = utils.info_to_pos(info)
            hindsight_reward, hindsight_reached = self.rf(pos, goal_pos)
            hindsight_done = hindsight_reached or done
            relabeled_trajectory.append((
                state, action, hindsight_reward, next_state, hindsight_done, reset, info
            ))
            if hindsight_reached: break

        # Sanity check on negative transitions when using oracle on -ive trajectories
        rewards = [transition[2] for transition in relabeled_trajectory]
        if not np.isclose(sum(rewards), 1): ipdb.set_trace()

        return relabeled_trajectory

    def experience_replay(self, trajectory, goal):
        for state, action, reward, next_state, done, reset, _ in trajectory:
            augmented_state = self.solver.get_augmented_state(state, goal)
            augmented_next_state = self.solver.get_augmented_state(next_state, goal)
            self.solver.step(augmented_state, action, deepcopy(reward), augmented_next_state, deepcopy(done), reset)
    
    # ------------------------------------------------------------
    # Learning initiation classifiers
    # ------------------------------------------------------------

    def derive_training_examples(self, visited_states, visited_infos, reached_term):
        assert len(visited_states) == len(visited_infos)

        start_info = visited_infos[0]
        start_state = visited_states[0]

        if reached_term:
            positive_infos = [start_info] + visited_infos[-self.buffer_length:]
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            
            self.initiation_classifier.add_positive_examples(positive_states, positive_infos)
        else:
            negative_infos = [start_info] 
            negative_states = [start_state]

            if self.use_full_neg_traj:
                negative_infos += visited_infos[-self.buffer_length:]
                negative_states += visited_states[-self.buffer_length:]

            self.initiation_classifier.add_negative_examples(negative_states, negative_infos)

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
