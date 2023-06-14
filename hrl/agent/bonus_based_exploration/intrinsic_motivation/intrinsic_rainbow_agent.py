# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of a Rainbow agent with intrinsic rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from hrl.agent.bonus_based_exploration import intrinsic_motivation

from hrl.agent.bonus_based_exploration.intrinsic_motivation import intrinsic_dqn_agent
from hrl.agent.bonus_based_exploration.intrinsic_motivation import intrinsic_rewards
from dopamine.agents.dqn import dqn_agent as base_dqn_agent
from dopamine.agents.rainbow import rainbow_agent as base_rainbow_agent
from dopamine.discrete_domains import atari_lib
import gin
import tensorflow.compat.v1 as tf
import numpy as np



@gin.configurable
class PixelCNNRainbowAgent(
    base_rainbow_agent.RainbowAgent,
    intrinsic_dqn_agent.PixelCNNDQNAgent):
  """A Rainbow agent paired with a pseudo count derived from a PixelCNN."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=base_dqn_agent.NATURE_DQN_DTYPE,
               stack_size=base_dqn_agent.NATURE_DQN_STACK_SIZE,
               network=atari_lib.RainbowNetwork,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=intrinsic_dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0000625, epsilon=0.00015),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expecting 2 parameters: num_actions,
        network_type. A call to this object will return an instantiation of the
        network provided. The network returned can be run with different inputs
        to create different outputs. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """
    self.intrinsic_model = intrinsic_rewards.PixelCNNIntrinsicReward(
        sess=sess,
        tf_device=tf_device)
    super(PixelCNNRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def step(self, reward, observation):
    return intrinsic_dqn_agent.PixelCNNDQNAgent.step(
        self, reward, observation)


@gin.configurable
class RNDRainbowAgent(
    base_rainbow_agent.RainbowAgent,
    intrinsic_dqn_agent.RNDDQNAgent):
  """A Rainbow agent paired with an intrinsic bonus derived from RND."""

  def __init__(self,
		  sess,
               num_actions,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=intrinsic_dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.0000625, epsilon=0.00015),
               summary_writer=None,
               summary_writing_frequency=500,
               clip_reward=False):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: tf.train.Optimizer, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      clip_reward: bool, whether or not clip the mixture of rewards.
    """
    self._clip_reward = clip_reward
    self.intrinsic_model = intrinsic_rewards.RNDIntrinsicReward(
        sess=sess,
        tf_device=tf_device,
        summary_writer=summary_writer)
    with tf.device(tf_device):
      batched_shape = (None,) + base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE + (base_dqn_agent.NATURE_DQN_STACK_SIZE,)
      self.batch_ph = tf.placeholder(base_dqn_agent.NATURE_DQN_DTYPE, batched_shape, name='observations_ph')
      obs_shape = (None,) + base_dqn_agent.NATURE_DQN_OBSERVATION_SHAPE + (1,)
      self.obs_batch_ph = tf.placeholder(tf.uint8, shape=obs_shape, name='obs_batch_ph')
    super(RNDRainbowAgent, self).__init__(
        sess=sess,
        num_actions=num_actions,
        num_atoms=num_atoms,
        vmax=vmax,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        replay_scheme=replay_scheme,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _build_networks(self):
    super()._build_networks()

    self._net_outputs = self.online_convnet(self.batch_ph)
    self.value_function = tf.reduce_max(self._net_outputs.q_values, axis=1)
    self.reward_function = (self.intrinsic_model.loss - self.intrinsic_model.reward_mean)/self.intrinsic_model.reward_std

  def _add_intrinsic_reward(self, observation, extrinsic_reward):
    return intrinsic_dqn_agent.RNDDQNAgent._add_intrinsic_reward(
        self, observation, extrinsic_reward)

  def _get_intrinsic_reward(self, observation):
    return self.intrinsic_model.compute_intrinsic_reward(observation,0,True)

  def _get_value_function(self, stacks, chunk_size=1000):

    values = np.empty(len(stacks))
    
    def get_chunks(x, n):
      # Break x into chunks of n
      for i in range(0, len(x), n):
        yield x[i: i+n]

    state_chunks = get_chunks(stacks, chunk_size)
    current_idx = 0

    for state_chunk in state_chunks:
      chunk_values = self._sess.run(self.value_function, {self.batch_ph: state_chunk})
      current_chunk_size = len(state_chunk)
      values[current_idx:current_idx + current_chunk_size] = chunk_values
      current_idx += current_chunk_size

    return values

  def begin_episode_from_point(self, starting_state):
    assert isinstance(starting_state, np.ndarray)
    assert (starting_state.shape == self.state.shape)

    self.state = starting_state
    self._observation = np.reshape(starting_state[:,:,:,-1], self.observation_shape)
    self._last_observation = np.reshape(starting_state[:,:,:,-2], self.observation_shape)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action()
    return self.action
