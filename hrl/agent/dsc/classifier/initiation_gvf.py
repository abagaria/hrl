import ipdb
import torch
import numpy as np

from pfrl.replay_buffers import ReplayBuffer
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer

from hrl.agent.ope.td0 import ContinuousTDPolicyEvaluator


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
  
  def optimistic_predict(self, states: np.ndarray, bonuses=None) -> np.ndarray:
    values = self.policy_evaluation_module.get_values(states, self.target_policy)
    if bonuses is not None:
      values += bonuses
    return values > self.optimistic_threshold

  def pessimistic_predict(self, states: np.ndarray) -> np.ndarray:
    values = self.policy_evaluation_module.get_values(states, self.target_policy)
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
