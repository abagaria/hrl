import os
import ipdb
import numpy as np

from tqdm import tqdm
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.discrete_domains import atari_lib
from absl import logging
import tensorflow.compat.v1 as tf

from hrl.agent.bonus_based_exploration.helpers import env_wrapper,ram_data_replay_buffer
import matplotlib.pyplot as plt

class RNDAgent(Runner):
    def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
        tf.logging.info('Creating episode wise runner...')
        super(RNDAgent, self).__init__(
            base_dir=base_dir,
            create_agent_fn=create_agent_fn,
            create_environment_fn=create_environment_fn
        )

        self.env_wrapper = env_wrapper.MontezumaInfoWrapper(self._environment)
        self.info_buffer = ram_data_replay_buffer.MontezumaRevengeReplayBuffer(self._agent._replay.memory._replay_capacity)

        self.info_buffer.load(self._base_dir)

    def set_env(self, env):
        """ env is non-lazy-frame version of train::make_env(). """
        self._environment = env
        self.env_wrapper = env_wrapper.MontezumaInfoWrapper(env)

    def rollout(self, iteration=0, steps=0):
        """
            Execute a full trajectory of the agent interacting with the environment.

            Returns:
                List of observations that make up the episode
                List of rewards achieved
        """

        self._agent.eval_mode = False
        
        rewards = []
        observations = []
        intrinsic_rewards = []
        visited_positions = []

        step_number = 0

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state

        while True:

            player_x = self.env_wrapper.get_player_x()
            player_y = self.env_wrapper.get_player_y()
            room_num = self.env_wrapper.get_room_number()
            visited_positions.append((player_x, player_y, room_num))

            self.info_buffer.add(player_x, player_y, room_num, self._agent._replay.memory.cursor())
            observation, reward, is_terminal = self._run_one_step(action)

            intrinsic_reward = self.get_intrinsic_reward(observation)

            rewards.append(reward)
            intrinsic_rewards.append(intrinsic_reward)
            observations.append(observation)
            reward = max(min(reward, 1),-1)

            if is_terminal or (step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode
                break

            elif is_terminal:
                # If we lose a life but the episode is not over, signal artificial end of episode to agent
                self._agent._end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)

            step_number += 1

        self._end_episode(reward)

        steps += step_number

        if (iteration % 100 == 0 and iteration != 0):
            logging.info('Saving model...')
            self._checkpoint_experiment(iteration)
            self.info_buffer.save(self._base_dir)
        if (iteration % 1000 == 0 and iteration != 0):
            logging.info('Plotting')
            self.plot(iteration, steps)

        logging.info('Completed episode %d', iteration)
        logging.info('Steps taken: %d Total reward: %d', step_number, sum(rewards))

        return np.array(observations), np.array(rewards), np.array(intrinsic_rewards), np.array(visited_positions)

    def get_intrinsic_reward(self, obs):
        rf = self._agent.intrinsic_model.compute_intrinsic_reward
        scaled_intrinsic_reward = rf(
            np.array(obs).reshape((84,84)),
            self._agent.training_steps,
            eval_mode=True
        )
        scale = self._agent.intrinsic_model.reward_scale
        assert np.isscalar(scale), scale
        if scale > 0:
            return scaled_intrinsic_reward / scale
        return 0.

    def value_function(self, stacks):
        ## Observation needs to be a state from nature dqn which is 4 frames
        return self._agent._get_value_function(stacks)

    def reward_function(self, observations):
        return np.array([self.get_intrinsic_reward(obs) for obs in observations])

    def plot(self, episode=0, steps=0):
        self._agent.eval_mode = True

        # logging.info(max_range)
            # logging.info(self._agent._replay.memory.cursor())
            # logging.info(self.info_buffer.is_full())

        values = {}
        rewards = {}
        player_x = {}
        player_y = {}

        max_range = self._agent._replay.memory.cursor()

        if self._agent._replay.memory.is_full():
            max_range = self.info_buffer.replay_capacity

        for index in range(max_range):
            if self._agent._replay.memory.is_valid_transition(index):
                stack = self._agent._replay.memory.get_observation_stack(index)
                room_number = self.info_buffer.get_index('room_number', index)

                if not room_number in values:
                    values[room_number] = []
                    rewards[room_number] = []
                    player_x[room_number] = []
                    player_y[room_number] = []

                stack = stack[np.newaxis, :]
                observation = stack[:,:,:,-1]

                values[room_number].append(self.value_function(stack)[0])
                rewards[room_number].append(self.reward_function(observation)[0])
                player_x[room_number].append(self.info_buffer.get_index('player_x', index))
                player_y[room_number].append(self.info_buffer.get_index('player_y', index))

        for key in values:
            plt.scatter(player_x[key], player_y[key], c=values[key], cmap='viridis')
            plt.colorbar()
            figname = self._get_plot_name(self._base_dir, 'value', str(key), str(episode), str(steps))
            plt.savefig(figname)
            plt.clf()

            plt.scatter(player_x[key], player_y[key], c=rewards[key],cmap='viridis')
            plt.colorbar()
            figname = self._get_plot_name(self._base_dir, 'reward', str(key), str(episode), str(steps))
            plt.savefig(figname)
            plt.clf()


    def _get_plot_name(self, base_dir, type, room, episode, steps):
        plot_dir = os.path.join(base_dir, 'plots', episode)
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        return os.path.join(plot_dir, '{}_room_{}_steps_{}.png'.format(type, room, steps))





        



