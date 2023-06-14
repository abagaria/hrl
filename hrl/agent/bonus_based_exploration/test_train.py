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

# Lint as: python3
r"""The entry point for running an agent on Atari.

Its main purpose is to allow gin to parametrize the training procedure and
avoid a cyclical dependency on FLAGS.

The actual methods used to run the experiment can be found in run_experiment.py.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app, logging
from absl import flags
from hrl.agent.bonus_based_exploration.run_experiment import create_exploration_agent as create_agent
from hrl.agent.bonus_based_exploration.run_experiment import create_exploration_runner as create_runner
from dopamine.discrete_domains import run_experiment
import tensorflow.compat.v1 as tf

import cProfile, pstats

import numpy as np


flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_atari_environment.game_name="Pong"").')
flags.DEFINE_string(
    'schedule', 'continuous_train_and_eval',
    'The schedule with which to run the experiment and choose an appropriate '
    'Runner. Supported choices are '
    '{continuous_train, eval, continuous_train_and_eval, '
    ' train_and_eval_with_video}.')

FLAGS = flags.FLAGS




def launch_experiment(create_runner_fn, create_agent_fn):
  """Launches the experiment.

  Args:
    create_runner_fn: A function that takes as args a base directory and a
      function for creating an agent and returns a `Runner` like object.
    create_agent_fn: A function that takes as args a Tensorflow session and a
     Gym Atari 2600 environment, and returns an agent.
  """

  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn,
                            schedule='episode_wise')
  steps = 0
  iteration = 0
  # while steps <= 1000000:
  while True:
    _,_, steps = runner.rollout(iteration, steps)
    iteration += 1
    logging.info("Total steps %d", steps)

def test_plot(create_runner_fn, create_agent_fn):
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn, schedule='episode_wise')

  runner.plot()

def test_value(create_runner_fn, create_agent_fn):
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn, schedule='episode_wise')
  observations, rewards, steps = runner.rollout(0,0)

  logging.info("Finished running one episode")

  first_obs = observations[:8]
  obs = np.array(first_obs).reshape((2, 84, 84, 4))

  actual_first_obs = observations[:4]
  actual = np.array(actual_first_obs).reshape((1,84,84,4))

  q_values = runner.value_function(obs)

  logging.info(q_values)
  logging.info(runner.value_function(actual))

  logging.info('Finished')

def test_reward(create_runner_fn, create_agent_fn):
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn, schedule='episode_wise')
  
  steps = 0
  iteration = 0

  while True:
    observations, rewards, steps = runner.rollout(iteration,steps)
    iteration += 1

    logging.info("Total steps %d", steps)

    # obs = np.array(obs).reshape((1, 84, 84, 4))

    intrinsic_reward = runner.reward_function(observations)

    logging.info(len(intrinsic_reward))
    logging.info(intrinsic_reward[-50])

def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  launch_experiment(create_runner, create_agent)
  # cProfile.run('launch_experiment(create_runner, create_agent)')
  # test_value(create_runner, create_agent)
  # test_reward(create_runner, create_agent)
  # test_plot(create_runner, create_agent)


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
