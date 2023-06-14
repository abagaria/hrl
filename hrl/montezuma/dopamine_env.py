# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari-specific utilities including Atari-specific network architectures.

This includes a class implementing minimal Atari 2600 preprocessing, which
is in charge of:
  . Emitting a terminal signal when losing a life (optional).
  . Frame skipping and color pooling.
  . Resizing the image before it is provided to the agent.

## Networks
We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.

More information about keras.Model API can be found here:
https://www.tensorflow.org/api_docs/python/tf/keras/models/Model

## Network Types
Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from absl import logging

import cv2
import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf

# TODO: This only works for the 1st room
START_POSITION = (77, 235)
NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])
ImplicitQuantileNetworkType = collections.namedtuple(
    'iqn_network', ['quantile_values', 'quantiles'])




@gin.configurable
def create_atari_environment(game_name="MontezumaRevenge", sticky_actions=True):
  """Wraps an Atari 2600 Gym environment with some basic preprocessing.

  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".

  The created environment is the Gym wrapper around the Arcade Learning
  Environment.

  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.

  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """
  assert game_name is not None
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
  # handle this time limit internally instead, which lets us cap at 108k frames
  # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
  # restoring states.
  env = env.env
  env = AtariPreprocessing(env)
  return env


@gin.configurable(denylist=['variables'])
def maybe_transform_variable_names(variables, legacy_checkpoint_load=False):
  """Maps old variable names to the new ones.

  The resulting dictionary can be passed to the tf.compat.v1.train.Saver to load
  legacy checkpoints into Keras models.

  Args:
    variables: list, of all variables to be transformed.
    legacy_checkpoint_load: bool, if True the variable names are mapped to
        the legacy names as appeared in `tf.slim` based agents. Use this if
        you want to load checkpoints saved before tf.keras.Model upgrade.
  Returns:
    dict or None, of <new_names, var>.
  """
  logging.info('legacy_checkpoint_load: %s', legacy_checkpoint_load)
  if legacy_checkpoint_load:
    name_map = {}
    for var in variables:
      new_name = var.op.name.replace('bias', 'biases')
      new_name = new_name.replace('kernel', 'weights')
      name_map[new_name] = var
  else:
    name_map = None
  return name_map


class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(NatureDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)

    return DQNNetworkType(self.dense2(x))


class RainbowNetwork(tf.keras.Model):
  """The convolutional network used to compute agent's return distributions."""

  def __init__(self, num_actions, num_atoms, support, name=None):
    """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to crete scope for network parameters.
    """
    super(RainbowNetwork, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)
    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


class ImplicitQuantileNetwork(tf.keras.Model):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def __init__(self, num_actions, quantile_embedding_dim, name=None):
    """Creates the layers used calculating quantile values.

    Args:
      num_actions: int, number of actions.
      quantile_embedding_dim: int, embedding dimension for the quantile input.
      name: str, used to create scope for network parameters.
    """
    super(ImplicitQuantileNetwork, self).__init__(name=name)
    self.num_actions = num_actions
    self.quantile_embedding_dim = quantile_embedding_dim
    # We need the activation function during `call`, therefore set the field.
    self.activation_fn = tf.keras.activations.relu
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.
    self.conv1 = tf.keras.layers.Conv2D(
        32, [8, 8], strides=4, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [4, 4], strides=2, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(
        64, [3, 3], strides=1, padding='same', activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(
        512, activation=self.activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

  def call(self, state, num_quantiles):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: `tf.Tensor`, contains the agent's current state.
      num_quantiles: int, number of quantile inputs.
    Returns:
      collections.namedtuple, that contains (quantile_values, quantiles).
    """
    batch_size = state.get_shape().as_list()[0]
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    state_vector_length = x.get_shape().as_list()[-1]
    state_net_tiled = tf.tile(x, [num_quantiles, 1])
    quantiles_shape = [num_quantiles * batch_size, 1]
    quantiles = tf.random.uniform(
        quantiles_shape, minval=0, maxval=1, dtype=tf.float32)
    quantile_net = tf.tile(quantiles, [1, self.quantile_embedding_dim])
    pi = tf.constant(math.pi)
    quantile_net = tf.cast(tf.range(
        1, self.quantile_embedding_dim + 1, 1), tf.float32) * pi * quantile_net
    quantile_net = tf.cos(quantile_net)
    # Create the quantile layer in the first call. This is because
    # number of output units depends on the input shape. Therefore, we can only
    # create the layer during the first forward call, not during `.__init__()`.
    if not hasattr(self, 'dense_quantile'):
      self.dense_quantile = tf.keras.layers.Dense(
          state_vector_length, activation=self.activation_fn,
          kernel_initializer=self.kernel_initializer)
    quantile_net = self.dense_quantile(quantile_net)
    x = tf.multiply(state_net_tiled, quantile_net)
    x = self.dense1(x)
    quantile_values = self.dense2(x)
    return ImplicitQuantileNetworkType(quantile_values, quantiles)


@gin.configurable
class AtariPreprocessing(object):
  """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
               screen_size=84):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))
    if screen_size <= 0:
      raise ValueError('Target screen size should be strictly positive, got {}'.
                       format(screen_size))

    self.environment = environment
    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size

    obs_dims = self.environment.observation_space
    # Stores temporary observations used for pooling over two successive
    # frames.
    self.screen_buffer = [
        np.zeros((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        np.zeros((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
    ]

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

    self.T = 0
    self.num_lives = None
    self.n_reached_interrupts = 0

    self.imaginary_ladder_locations = set()

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
               dtype=np.uint8)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    self.environment.reset()
    self.num_lives = self.get_num_lives(self.get_current_ram()) 
    self.lives = self.environment.ale.lives()
    self._fetch_grayscale_observation(self.screen_buffer[0])
    self.screen_buffer[1].fill(0)
    return self._pool_and_resize()

  def render(self, mode):
    """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
    return self.environment.render(mode)

  def step(self, action, clf=None):
    """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    self.T += 1
    accumulated_reward = 0.

    for time_step in range(self.frame_skip):
      # We bypass the Gym observation altogether and directly fetch the
      # grayscale image from the ALE. This is a little faster.
      _, reward, game_over, info = self.environment.step(action)
      accumulated_reward += reward
      
      info = self.get_current_info(info)

      if self.terminal_on_life_loss:
        new_lives = self.environment.ale.lives()
        is_terminal = game_over or new_lives < self.lives
        self.lives = new_lives
      else:
        is_terminal = game_over

      # Akhil: Removed max-pooling! And always updating the current obs
      self._fetch_grayscale_observation(self.screen_buffer[0])

      # Check if intermediate observations trigger a termination condition
      reached = clf(self._pool_and_resize(), info) if clf else False

      if (not is_terminal) and reached:
        self.n_reached_interrupts += 1

      if is_terminal or reached:
        break

    # Pool the last two observations.
    observation = self._pool_and_resize()

    self.game_over = game_over
    return observation, accumulated_reward, is_terminal, info

  def _fetch_grayscale_observation(self, output):
    """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    self.environment.ale.getScreenGrayscale(output)
    return output

  def _pool_and_resize(self):
    """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
    # Pool if there are enough screens to do so.
    if self.frame_skip > 1:
      np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                 out=self.screen_buffer[0])

    transformed_image = cv2.resize(self.screen_buffer[0],
                                   (self.screen_size, self.screen_size),
                                   interpolation=cv2.INTER_AREA)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)
  
  def get_current_info(self, info, update_lives=False):
    ram = self.get_current_ram()

    info["lives"] = self.get_num_lives(ram)
    info["player_x"] = self.get_player_x(ram)
    info["player_y"] = self.get_player_y(ram)
    info["room_number"] = self.get_room_number(ram)
    info["jumping"] = self.get_is_jumping(ram)
    info["dead"] = self.get_is_player_dead(ram)
    info["falling"] = self.get_is_falling(ram)
    info["uncontrollable"] = self.get_is_in_non_controllable_state(ram)
    info["buggy_state"] = self.get_is_climbing_imaginary_ladder(ram)
    info["left_door_open"] = self.get_is_left_door_unlocked(ram)
    info["right_door_open"] = self.get_is_right_door_unlocked(ram)
    info["inventory"] = self.get_player_inventory(ram)

    if update_lives:
      self.num_lives = info["lives"]

    return info

  def get_current_ram(self):
    return self.environment.ale.getRAM()

  def get_current_position(self):
    ram = self.get_current_ram()
    return self.get_player_x(ram), self.get_player_y(ram)

  def get_player_x(self, ram):
    return int(self.getByte(ram, 'aa'))

  def get_player_y(self, ram):
    return int(self.getByte(ram, 'ab'))

  def get_num_lives(self, ram):
    return int(self.getByte(ram, 'ba'))

  def get_is_falling(self, ram):
    return int(int(self.getByte(ram, 'd8')) != 0)

  def get_is_jumping(self, ram):
    return int(self.getByte(ram, 'd6') != 0xFF)

  def get_room_number(self, ram):
    return int(self.getByte(ram, '83'))

  def get_player_inventory(self, ram):
    # 'torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer'
    return format(self.getByte(ram, 'c1'), '08b')

  @staticmethod
  def getByte(ram, address):
    idx = AtariPreprocessing._getIndex(address)
    return ram[idx]
  
  @staticmethod
  def _getIndex(address):
      row, col = tuple(address)
      row = int(row, 16) - 8
      col = int(col, 16)
      return row*16+col

  def get_player_status(self, ram):
        status = self.getByte(ram, '9e')
        status_codes = {
            0x00: 'standing',
            0x2A: 'running',
            0x3E: 'on-ladder',
            0x52: 'climbing-ladder',
            0x7B: 'on-rope',
            0x90: 'climbing-rope',
            0xA5: 'mid-air',
            0xBA: 'dead',  # dive 1
            0xC9: 'dead',  # dive 2
            0xC8: 'dead',  # dissolve 1
            0xDD: 'dead',  # dissolve 2
            0xFD: 'dead',  # smoke 1
            0xE7: 'dead',  # smoke 2
        }
        return status_codes[status]

  def get_is_player_dead(self, ram):
    player_status = self.get_player_status(ram)
    dead = player_status == "dead"
    time_to_spawn = self.getByte(ram, "b7")
    respawning = time_to_spawn > 0
    return dead or respawning

  def get_is_in_non_controllable_state(self, ram):
    player_status = self.get_player_status(ram)
    return self.get_is_jumping(ram) or \
      player_status in ("mid-air") or \
      self.get_is_falling(ram) or \
      self.get_is_player_dead(ram)

  def get_is_climbing_imaginary_ladder(self, ram):
    screen = self.get_room_number(ram)
    x_pos = self.get_player_x(ram)
    y_pos = self.get_player_y(ram)
    position = x_pos, y_pos
    imaginary = not self.at_any_climb_region(position, screen)
    ladder = self.get_player_status(ram) == "climbing-ladder"
    climbing = imaginary and ladder
    known_imaginary = (x_pos, y_pos, screen) in self.imaginary_ladder_locations
    if climbing:
      print(f"Found climbing imaginary ladder {position, screen}")
    if climbing and not known_imaginary:
      print(f"New imaginary location, adding {x_pos, y_pos, screen} to blacklist")
      self.imaginary_ladder_locations.add((x_pos, y_pos, screen))
    return climbing

  def get_is_left_door_unlocked(self, ram):
    objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
    left_door = objects[0]
    locked = int(left_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
    return not locked

  def get_is_right_door_unlocked(self, ram):
    objects = format(self.getByte(ram, 'c2'), '08b')[-4:]
    right_door = objects[1]
    locked = int(right_door) == 1 and self.get_room_number(ram) in [1, 5, 17]
    return not locked

  def at_any_climb_region(self, pos, screen):
    climb_margin = 4
    for x_center, ylim in self.get_climb_regions(screen):
      if (ylim[0] <= pos[1] <= ylim[1]
        and abs(pos[0] - x_center) <= climb_margin):
          return True
    return False

  def get_climb_regions(self, screen):
    LEFT_LADDER = 0x14
    CENTER_LADDER = 0x4d
    RIGHT_LADDER = 0x85
    SCREEN_TOP = 0xfe
    UPPER_LEVEL = 0xeb
    MIDDLE_LEVEL_1 = 0xc0
    LOWER_LEVEL_1 = 0x94
    LOWER_LEVEL_5 = 0x9d
    LOWER_LEVEL_14 = 0xa0
    SCREEN_BOTTOM = 0x86

    regions = []
    climb_margin_y = 6

    if screen == 1:
      regions.append((CENTER_LADDER, (MIDDLE_LEVEL_1, UPPER_LEVEL)))
      regions.append((LEFT_LADDER, (LOWER_LEVEL_1, MIDDLE_LEVEL_1)))
      regions.append((RIGHT_LADDER, (LOWER_LEVEL_1, MIDDLE_LEVEL_1)))
    if screen == 5:
      regions.append((CENTER_LADDER, (SCREEN_BOTTOM, LOWER_LEVEL_5)))
      regions.append((CENTER_LADDER, (SCREEN_BOTTOM - 1, SCREEN_BOTTOM)))
    if screen == 14:
      regions.append((CENTER_LADDER, (SCREEN_BOTTOM, LOWER_LEVEL_14)))
      regions.append((CENTER_LADDER, (SCREEN_BOTTOM - 1, SCREEN_BOTTOM)))
    # tall bottom ladders
    if screen in [0, 2, 3, 4, 7, 11, 13]:
      regions.append((CENTER_LADDER, (SCREEN_BOTTOM, UPPER_LEVEL)))
      regions.append((CENTER_LADDER, (SCREEN_BOTTOM - 1, SCREEN_BOTTOM)))
    # short top ladders
    if screen in [4, 6, 9, 11, 13, 19, 21]:
      regions.append((CENTER_LADDER, (UPPER_LEVEL, SCREEN_TOP)))
      regions.append((CENTER_LADDER, (SCREEN_TOP, SCREEN_TOP + 1)))
    elif screen in [10, 22]:
      # add vertical landmark just above the bridge
      regions.append(
          (CENTER_LADDER, (UPPER_LEVEL, UPPER_LEVEL + climb_margin_y)))
      regions.append(
          (CENTER_LADDER, (UPPER_LEVEL + climb_margin_y, SCREEN_TOP)))
      regions.append((CENTER_LADDER, (SCREEN_TOP, SCREEN_TOP + 1)))
    return regions