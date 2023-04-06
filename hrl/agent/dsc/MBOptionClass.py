import ipdb
import torch
import random
import numpy as np
from copy import deepcopy

from scipy.spatial import distance
from hrl.agent.dynamics.mpc import MPC
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.wrappers.gc_mdp_wrapper import GoalConditionedMDPWrapper
from hrl.agent.dsc.classifier.obs_init_classifier import ObsInitiationClassifier
from hrl.agent.dsc.classifier.critic_classifier import CriticInitiationClassifier
from hrl.agent.dsc.classifier.position_classifier import PositionInitiationClassifier


class ModelBasedOption(object):
    def __init__(self, *, name, parent, mdp, global_solver, global_value_learner, buffer_length, global_init,
                 gestation_period, timeout, max_steps, device, use_vf, use_global_vf, use_model, dense_reward,
                 option_idx, lr_c, lr_a, max_num_children=1, init_salient_event=None, target_salient_event=None,
                 path_to_model="", multithread_mpc=False, init_classifier_type="position-clf",
                 optimistic_threshold=40, pessimistic_threshold=20, initiation_gvf=None):
        assert isinstance(mdp, GoalConditionedMDPWrapper), mdp

        self.mdp = mdp
        self.name = name
        self.lr_c = lr_c
        self.lr_a = lr_a
        self.parent = parent
        self.device = device
        self.use_vf = use_vf
        self.global_solver = global_solver
        self.use_global_vf = use_global_vf
        self.timeout = timeout
        self.use_model = use_model
        self.max_steps = max_steps
        self.global_init = global_init
        self.dense_reward = dense_reward
        self.buffer_length = buffer_length
        self.max_num_children = max_num_children
        self.init_salient_event = init_salient_event
        self.target_salient_event = target_salient_event
        self.multithread_mpc = multithread_mpc
        self.init_classifier_type = init_classifier_type
        self.optimistic_threshold = optimistic_threshold
        self.pessimistic_threshold = pessimistic_threshold
        self.initiation_gvf = initiation_gvf

        # TODO
        self.overall_mdp = mdp
        self.seed = 0
        self.option_idx = option_idx

        self.num_goal_hits = 0
        self.num_executions = 0
        self.gestation_period = gestation_period

        # In the model-free setting, the output norm doesn't seem to work
        # But it seems to stabilize off policy value function learning
        # Therefore, only use output norm if we are using MPC for action selection
        use_output_norm = self.use_model

        if not self.use_global_vf:
            print(f'Creating NEW local-policy for {name}')
            self.value_learner = TD3(state_dim=self.mdp.state_space_size()+2,
                                    action_dim=self.mdp.action_space_size(),
                                    max_action=1.,
                                    name=f"{name}-td3-agent",
                                    device=self.device,
                                    lr_c=lr_c, lr_a=lr_a,
                                    use_output_normalization=use_output_norm)
        else:
            self.value_learner = global_value_learner

        self.global_value_learner = global_value_learner if not self.global_init else None  # type: TD3

        if use_model:
            print(f"Using model-based controller for {name}")
            self.solver = self._get_model_based_solver()
        else:
            print(f"Using model-free controller for {name}")
            self.solver = self._get_model_free_solver()

        self.initiation_classifier = self._get_initiation_classifier()

        self.children = []
        self.success_curve = []
        self.effect_set = []

        if path_to_model:
            print(f"Loading model from {path_to_model} for {self.name}")
            self.solver.load_model(path_to_model)

        if self.use_vf and not self.use_global_vf and self.parent is not None:
            self.initialize_value_function_with_global_value_function()

        print(f"Created model-based option {self.name} with option_idx={self.option_idx}")

        self.is_last_option = False

    def _get_model_based_solver(self):
        assert self.use_model

        if self.global_init:
            return MPC(mdp=self.mdp,
                       state_size=self.mdp.state_space_size(),
                       action_size=self.mdp.action_space_size(),
                       dense_reward=self.dense_reward,
                       device=self.device,
                       multithread=self.multithread_mpc)

        assert self.global_solver is not None
        return self.global_solver

    def _get_model_free_solver(self):
        assert not self.use_model
        assert self.use_vf

        # Global option creates its own VF solver
        if self.global_init:
            assert self.value_learner is not None
            return self.value_learner

        # Local option either uses the global VF..
        if self.use_global_vf:
            assert self.global_value_learner is not None
            return self.global_value_learner

        # .. or uses its own local VF as solver
        assert self.value_learner is not None
        return self.value_learner

    def _get_initiation_classifier(self):
        if self.init_classifier_type == "position-clf":
            return PositionInitiationClassifier()
        if self.init_classifier_type == "state-clf":
            return ObsInitiationClassifier(
                self.mdp.state_space_size(),
                device=self.device,
            )
        if self.init_classifier_type == "critic-threshold":
            return CriticInitiationClassifier(
                self.solver,
                self.get_goal_for_rollout,
                self.get_augmented_state,
                optimistic_threshold=self.optimistic_threshold,
                pessimistic_threshold=self.pessimistic_threshold
            )
        
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
        
        if self.is_last_option and self.mdp.get_start_state_salient_event()(state):
            return True

        return self.initiation_classifier.optimistic_predict(state) or self.pessimistic_is_init_true(state) 

    def is_term_true(self, state):
        if self.parent is None:
            return self.target_salient_event(state)

        assert isinstance(self.parent, ModelBasedOption)
        return self.parent.pessimistic_is_init_true(state)

    def pessimistic_is_init_true(self, state):
        if self.global_init or self.get_training_phase() == "gestation":
            return True

        return self.initiation_classifier.pessimistic_predict(state)

    def is_at_local_goal(self, state, goal):
        """ Goal-conditioned termination condition. """

        reached_goal = self.mdp.sparse_gc_reward_func(state, goal)[1]
        reached_term = self.is_term_true(state)
        return reached_goal and reached_term

    # ------------------------------------------------------------
    # Control Loop Methods
    # ------------------------------------------------------------

    def _get_epsilon(self):
        if self.use_model:
            return 0.1
        if not self.dense_reward and self.num_goal_hits <= 3:
            return 0.8
        return 0.2

    def act(self, state, goal):
        """ Epsilon-greedy action selection. """

        if random.random() < self._get_epsilon():
            return self.mdp.action_space.sample()

        if self.use_model:
            assert isinstance(self.solver, MPC), f"{type(self.solver)}"
            vf = self.value_function if self.use_vf else None
            return self.solver.act(state, goal, vf=vf)

        assert isinstance(self.solver, TD3), f"{type(self.solver)}"
        augmented_state = self.get_augmented_state(state, goal)
        return self.solver.act(augmented_state, evaluation_mode=False)

    def update_model(self, state, action, reward, next_state, next_done):
        """ Learning update for option model/actor/critic. """

        self.solver.step(state, action, reward, next_state, next_done)

    def get_goal_for_rollout(self):
        """ Sample goal to pursue for option rollout. """

        if self.parent is None and self.target_salient_event is not None:
            return self.target_salient_event.get_target_position()

        sampled_goal = None
        parent = self.parent

        while parent is not None and sampled_goal is None:
            sampled_goal = parent.initiation_classifier.sample()
            parent = parent.parent

        if sampled_goal is not None:
            return self.extract_goal_dimensions(sampled_goal)

        if self.target_salient_event is not None:
            return self.target_salient_event.get_target_position()
        
        raise ValueError(f"{self, self.parent, sampled_goal, parent}")

    def rollout(self, *, goal, step_number, eval_mode=False):
        """ Main option control loop. """

        num_steps = 0
        total_reward = 0
        visited_states = []
        option_transitions = []

        state = deepcopy(self.mdp.cur_state)

        print(f"[Step: {step_number}] Rolling out {self.name}, from {state[:2]} targeting {goal}")

        self.num_executions += 1

        while not self.is_at_local_goal(state, goal) and step_number < self.max_steps and num_steps < self.timeout:

            # Control
            action = self.act(state, goal)
            next_state, reward, next_done, _ = self.mdp.step(action)

            if self.use_model:
                self.update_model(state, action, reward, next_state, next_done)

            # Logging
            num_steps += 1
            step_number += 1
            total_reward += reward
            visited_states.append(state)
            option_transitions.append((state, action, reward, next_state, next_done))
            state = deepcopy(self.mdp.cur_state)

        visited_states.append(state)
        reached_term = self.is_term_true(state)
        self.success_curve.append(reached_term)

        if self.use_vf and not eval_mode:
            self.update_value_function(option_transitions,
                                    pursued_goal=goal,
                                    reached_goal=self.extract_goal_dimensions(state))
            
            if self.initiation_gvf is not None:
                self.update_initiation_value_function(
                    option_transitions,
                    pursued_goal=goal,
                    reached_goal=self.extract_goal_dimensions(state)
                )

        is_valid_data = self.max_num_children == 1 or self.is_valid_init_data(state_buffer=visited_states)

        if reached_term and is_valid_data and not eval_mode:
            self.num_goal_hits += 1
            self.effect_set.append(state)

        if not self.global_init and is_valid_data:
            self.derive_positive_and_negative_examples(visited_states, goal)

        return option_transitions, total_reward

    # ------------------------------------------------------------
    # Hindsight Experience Replay
    # ------------------------------------------------------------

    def update_value_function(self, option_transitions, reached_goal, pursued_goal):
        """ Update the goal-conditioned option value function. """

        self.experience_replay(option_transitions, pursued_goal)
        self.experience_replay(option_transitions, reached_goal)

    def update_initiation_value_function(self, option_transitions, reached_goal, pursued_goal):
        """Update the goal-conditioned initiation general value function."""
        
        self.initiation_gvf.add_trajectory_to_replay(
            self.relabel_trajectory(option_transitions, pursued_goal)
        )

        self.initiation_gvf.add_trajectory_to_replay(
            self.relabel_trajectory(option_transitions, reached_goal)
        )

    def initialize_value_function_with_global_value_function(self):
        self.value_learner.actor.load_state_dict(self.global_value_learner.actor.state_dict())
        self.value_learner.critic.load_state_dict(self.global_value_learner.critic.state_dict())
        self.value_learner.target_actor.load_state_dict(self.global_value_learner.target_actor.state_dict())
        self.value_learner.target_critic.load_state_dict(self.global_value_learner.target_critic.state_dict())

    def extract_goal_dimensions(self, goal):
        def _extract(goal):
            goal_features = goal
            if "ant" in self.mdp.unwrapped.spec.id:
                return goal_features[:2]
            raise NotImplementedError(f"{self.mdp.env_name}")
        if isinstance(goal, np.ndarray):
            return _extract(goal)
        return goal.pos

    def get_augmented_state(self, state, goal):
        assert goal is not None and isinstance(goal, np.ndarray), f"goal is {goal}"

        goal_position = self.extract_goal_dimensions(goal)
        return np.concatenate((state, goal_position))

    def experience_replay(self, trajectory, goal_state):
        for state, action, reward, next_state, next_done in trajectory:
            augmented_state = self.get_augmented_state(state, goal=goal_state)
            augmented_next_state = self.get_augmented_state(next_state, goal=goal_state)
            done = self.is_at_local_goal(next_state, goal_state)

            reward_func = self.overall_mdp.dense_gc_reward_func if self.dense_reward \
                else self.overall_mdp.sparse_gc_reward_func
            reward, global_done = reward_func(next_state, goal_state)

            if not self.use_global_vf or self.global_init:
                self.value_learner.step(augmented_state, action, reward, augmented_next_state, done)

            # Off-policy updates to the global option value function
            if not self.global_init:
                assert self.global_value_learner is not None
                self.global_value_learner.step(augmented_state, action, reward, augmented_next_state, global_done)

    def value_function(self, states, goals):
        assert isinstance(states, np.ndarray)
        assert isinstance(goals, np.ndarray)

        if len(states.shape) == 1:
            states = states[None, ...]
        if len(goals.shape) == 1:
            goals = goals[None, ...]

        if states.shape[0] != goals.shape[0]:
            assert goals.shape[0] == 1, (states.shape, goals.shape)
            goals = np.repeat(goals, repeats=len(states), axis=0)

        goal_positions = goals[:, :2]
        augmented_states = np.concatenate((states, goal_positions), axis=1)
        augmented_states = torch.as_tensor(augmented_states).float().to(self.device)

        if self.use_global_vf and not self.global_init:
            values = self.global_value_learner.get_values(augmented_states)
        else:
            values = self.value_learner.get_values(augmented_states)

        return values
    
    def relabel_trajectory(self, trajectory, goal_state):
        def initiation_reward_func(state, goal, threshold=0.6):
            pos = self.extract_goal_dimensions(state)
            goal_pos = self.extract_goal_dimensions(goal)
            assert isinstance(pos, np.ndarray)
            assert isinstance(goal_pos, np.ndarray)
            assert pos.shape == goal_pos.shape == (2,), state.shape
            success = np.linalg.norm(pos-goal_pos) <= threshold
            return float(success), success

        relabeled_trajectory = []

        for state, action, _, next_state, _ in trajectory:
            augmented_state = self.get_augmented_state(state, goal_state)
            augmented_next_state = self.get_augmented_state(next_state, goal_state)

            reward, reached = initiation_reward_func(next_state, goal_state)

            relabeled_trajectory.append((
                augmented_state, 
                action,
                reward,
                augmented_next_state,
                reached
            ))

            if reached:
                break

        return relabeled_trajectory

    # ------------------------------------------------------------
    # Learning Initiation Classifiers
    # ------------------------------------------------------------

    def sample_from_termination_region(self):
        pessimistic_samples = self.initiation_classifier.get_states_inside_pessimistic_classifier_region()

        if len(pessimistic_samples) > 0:
            sample = random.choice(pessimistic_samples)
            return self.mdp.extract_features_for_initiation_classifier(sample)

        sample = random.choice(self.effect_set)
        return self.mdp.extract_features_for_initiation_classifier(sample)

    def derive_positive_and_negative_examples(self, visited_states, pursued_goal):

        def state2info(s, v=None):
            sg = self.get_augmented_state(s, pursued_goal)
            return dict(
                player_x=s[0],  # TODO: Dont assume that its in 0, 1
                player_y=s[1],
                augmented_state=sg,
                value=v,
            )

        start_state = visited_states[0]
        final_state = visited_states[-1]

        if self.is_term_true(final_state):
            positive_states = [start_state] + visited_states[-self.buffer_length:]
            positive_values = self.value_function(np.array(positive_states), pursued_goal)
            positive_infos = [state2info(state, value) for state, value in zip(positive_states, positive_values)]
            self.initiation_classifier.add_positive_examples(positive_states, positive_infos)
        else:
            negative_states = [start_state]
            negative_values = self.value_function(np.array(negative_states), pursued_goal)
            negative_infos = [state2info(negative_states[0], negative_values[0])]
            self.initiation_classifier.add_negative_examples(negative_states, negative_infos)

    def should_change_negative_examples(self):
        pass

    def is_valid_init_data(self, state_buffer):

        # Use the data if it could complete the chain
        if self.init_salient_event is not None:
            if any([self.init_salient_event(s) for s in state_buffer]):
                return True

        length_condition = len(state_buffer) >= (self.buffer_length // 5)

        if not length_condition:
            return False

        sibling_cond = lambda o: o.get_training_phase() != "gestation" and o.initiation_classifier.is_initialized()
        siblings = [option for option in self.get_sibling_options() if sibling_cond(option)]

        if len(siblings) > 0:
            assert self.parent is not None, "Root option has no siblings"

            sibling_count = 0.
            for state in state_buffer:
                for sibling in siblings:
                    penalize = sibling.pessimistic_is_init_true(state) and not self.parent.pessimistic_is_init_true(state)
                    sibling_count += penalize

            return 0 < (sibling_count / len(state_buffer)) <= 0.35

        return True

    # ------------------------------------------------------------
    # Distance functions
    # ------------------------------------------------------------

    def distance_to_state(self, state, metric="euclidean"):
        """ Compute the distance between the current option and the input `state`. """

        assert metric in ("euclidean", "value"), metric
        if metric == "euclidean":
            return self._euclidean_distance_to_state(state)
        return self._value_distance_to_state(state)

    def _euclidean_distance_to_state(self, state):
        point = self.mdp.get_position(state)

        assert isinstance(point, np.ndarray)
        assert point.shape == (2,), point.shape

        positive_states = self.initiation_classifier.get_states_inside_pessimistic_classifier_region()

        if isinstance(positive_states, list):
            positive_states = np.array(positive_states)

        if isinstance(positive_states, torch.Tensor):
            positive_states = positive_states.detach().cpu().numpy()

        if positive_states.shape[1] > 2:
            positive_states = positive_states[:, :2]

        distances = distance.cdist(point[np.newaxis, :], positive_states)
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
        if isinstance(other, ModelBasedOption):
            return self.name == other.name
        return False
