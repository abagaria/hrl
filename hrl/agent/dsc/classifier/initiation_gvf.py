import ipdb
import torch
import numpy as np

from pfrl.replay_buffers import ReplayBuffer
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer

from hrl.agent.ope.td0 import ContinuousTDPolicyEvaluator
from typing import Tuple


class InitiationGVF:
    """Base class for the GVF approach to init-learning."""

    def __init__(
        self,
        target_policy,
        n_actions: int,
        n_input_channels: int,
        batch_size: int = 1024,
        optimistic_threshold: float = 0.7,
        pessimistic_threshold: float = 0.8,
        use_prioritized_buffer: bool = True,
        init_replay_capacity: int = 100_000,
    ):
        super().__init__()
        self._n_actions = n_actions
        self._n_input_channels = n_input_channels

        # Function that maps batch of states to batch of actions (`batch_act()`)
        self.target_policy = target_policy

        self.batch_size = batch_size
        self.optimistic_threshold = optimistic_threshold
        self.pessimistic_threshold = pessimistic_threshold

        buffer_cls = PrioritizedReplayBuffer if use_prioritized_buffer else ReplayBuffer
        self.initiation_replay_buffer = buffer_cls(init_replay_capacity)

        self.policy_evaluation_module = ContinuousTDPolicyEvaluator(
            self.initiation_replay_buffer,
            n_actions=n_actions,
            n_input_channels=n_input_channels
        )

        self.state_goal_count_dict = {}

    def add_trajectory_to_replay(self, transitions):
        for state, action, rg, next_state, done in transitions:
            self.initiation_replay_buffer.append(
                state,
                action,
                rg,
                next_state,
                is_state_terminal=done,
                # extra_info=info
            )

    def update_pseudo_count(self, transition: np.ndarray, goal: np.ndarray):
        """Update the pseudo-count for a given state-goal pair."""

        for state, action, _, next_state, _ in transition:
            # Discretize the state & goal to a grid.
            # Why 0.6? Because that's the threshold we use for the goal.
            pos_x = int(state[0]/0.6)
            pos_y = int(state[1]/0.6)

            goal_x = int(goal[0]/0.6)
            goal_y = int(goal[1]/0.6)

            if (pos_x, pos_y, goal_x, goal_y) not in self.state_goal_count_dict:
                self.state_goal_count_dict[(pos_x, pos_y, goal_x, goal_y)] = 1
            else:
                self.state_goal_count_dict[(pos_x, pos_y, goal_x, goal_y)] += 1

    def get_value_and_uncertainty(self, states: np.ndarray, bonuses=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q1, q2 = self.policy_evaluation_module.get_twin_values(
                states, self.target_policy)
            q_online, q_target = self.policy_evaluation_module.get_online_target_values(
                states, self.target_policy)

        Q = torch.min(q1.squeeze(), q2.squeeze())
        Q1_Q2_diff = torch.unsqueeze(
            0.5*torch.abs(q1.squeeze() - q2.squeeze()), dim=0)
        Q_and_Q_targ_diff = torch.abs(q_online.squeeze() - q_target.squeeze())

        return Q, Q1_Q2_diff, Q_and_Q_targ_diff

    def optimistic_predict(self, states: np.ndarray, bonuses=None) -> np.ndarray:
        values = self.policy_evaluation_module.get_values(
            states, self.target_policy)
        if bonuses is not None:
            values += bonuses
        return values > self.optimistic_threshold

    def pessimistic_predict(self, states: np.ndarray) -> np.ndarray:
        values = self.policy_evaluation_module.get_values(
            states, self.target_policy)
        return values > self.pessimistic_threshold

    def update(self, n_updates: int = 1):
        enough_samples = len(self.initiation_replay_buffer) > self.batch_size
        if enough_samples:
            for _ in range(n_updates):
                self.policy_evaluation_module.train(self.target_policy)

    def save(self, filename: str):
        torch.save(
            dict(
                online=self.policy_evaluation_module._online_q_network.state_dict(),
                target=self.policy_evaluation_module._target_q_network.state_dict(),
            ), filename
        )
        self.initiation_replay_buffer.save(
            filename.replace('.pth', '.pkl')
        )

    def load(self, filename: str):
        buffer_cls = type(self.initiation_replay_buffer)
        replay = buffer_cls(self.initiation_replay_buffer.capacity)
        self.policy_evaluation_module = ContinuousTDPolicyEvaluator(
            replay,
            self._n_actions,
            self._n_input_channels
        )
        replay.load(filename.replace('.pth', '.pkl'))

        model_dict = torch.load(filename)
        self.policy_evaluation_module._online_q_network.load_state_dict(
            model_dict['online']
        )
        self.policy_evaluation_module._target_q_network.load_state_dict(
            model_dict['target']
        )


class GoalConditionedInitiationGVF(InitiationGVF):
    def get_values(self, states, goals):
        sg = self.get_augmeted_state(states, goals)
        return self.policy_evaluation_module.get_values(sg, self.target_policy)

    def optimistic_predict(self, states, goals, bonuses=None) -> np.ndarray:
        values = self.get_values(states, goals)
        if bonuses is not None:
            values += bonuses
        return values.max() > self.optimistic_threshold

    def pessimistic_predict(self, states, goals) -> np.ndarray:
        values = self.get_values(states, goals)
        return values.max() > self.pessimistic_threshold

    def get_augmeted_state(self, states, goals):
        if isinstance(states, list):
            states = np.asarray(states)
        if isinstance(goals, list):
            goals = np.asarray(goals)
        if goals.shape[-1] != 2:
            assert goals.shape[-1] == 29, ipdb.set_trace()
            goals = goals[:, :2]
        return np.concatenate((states, goals), axis=1)
