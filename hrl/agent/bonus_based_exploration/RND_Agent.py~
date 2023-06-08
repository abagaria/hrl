import os
# import ipdb
import numpy as np

from dopamine.discrete_domains.run_experiment import Runner
from absl import logging
import tensorflow.compat.v1 as tf

from hrl.montezuma.info_wrapper import MontezumaInfoWrapper
from hrl.montezuma.dopamine_env import create_atari_environment
from hrl.agent.bonus_based_exploration.helpers import ram_data_replay_buffer
from hrl.agent.dsg.minigrid import MinigridInfoWrapper
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
class RNDAgent(Runner):
    def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=create_atari_environment,
               env_wrapper=MontezumaInfoWrapper):
        tf.logging.info('Creating episode wise runner...')
        super(RNDAgent, self).__init__(
            base_dir=base_dir,
            create_agent_fn=create_agent_fn,
            create_environment_fn=create_environment_fn)
            
        if (isinstance(self._environment, MinigridInfoWrapper)):
        	self.env_wrapper = self._environment
        self.env_wrapper = env_wrapper(self._environment)
        self.info_buffer = ram_data_replay_buffer.MontezumaRevengeReplayBuffer(self._agent._replay.memory._replay_capacity)
        self.env_wrapper.num_lives = 0
        self.info_buffer.load(self._base_dir)
        
        if callable(getattr(self.env_wrapper, "get_current_ram", None)):
        	self.env_wrapper.num_lives = self.env_wrapper.get_num_lives(self.env_wrapper.get_current_ram())

    def _initialize_episode_from_point(self, starting_state):
        return self._agent.begin_episode_from_point(starting_state)

    def rollout(self, initial_state=None, iteration=0, steps=0):
        """
            Execute a full trajectory of the agent interacting with the environment.

            Returns:
                List of observations that make up the episode
                List of rewards achieved
        """

        self._agent.eval_mode = False
        
        rewards = []
        observations = []
        visited_infos = []
        intrinsic_rewards = []

        step_number = 0

        if initial_state is None:
            action = self._initialize_episode()
        else:
            action = self._initialize_episode_from_point(initial_state)
        
        actions = [action]
        is_terminal = False

        # Keep interacting until we reach a terminal state

        while True:

            observation, reward, is_terminal = self._run_one_step(action)

            intrinsic_reward = self.get_intrinsic_reward(observation)

            rewards.append(reward)
            intrinsic_rewards.append(intrinsic_reward)
            observations.append(observation)
            reward = max(min(reward, 1),-1)

            info = self.env_wrapper.get_current_info(info={}, update_lives=True)
            visited_infos.append(info)
            self.info_buffer.add(info['player_x'],info['player_y'], info['room_number'],
	    		self._agent._replay.memory.cursor())

            if self._environment.game_over or (step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode
                break

            elif is_terminal:
                # If we lose a life but the episode is not over, signal artificial end of episode to agent
                self._agent._end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)

            step_number += 1
            actions.append(action)

        self._end_episode(reward)

        steps += step_number

        logging.info('Completed episode %d', iteration)
        logging.info('Steps taken: %d Total reward: %d', step_number, sum(rewards))

        return np.array(observations),\
               actions,\
               np.array(rewards),\
               np.array(intrinsic_rewards),\
               visited_infos

    def get_intrinsic_reward(self, obs):
        rf = self._agent.intrinsic_model.compute_intrinsic_reward
        self._agent.intrinsic_model._sess = self._sess
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

    def value_function(self, stacks):  # TODO: Maybe set agent eval_mode=True
        ## Observation needs to be a state from nature dqn which is 4 frames
        return self._agent._get_value_function(stacks)

    def reward_function(self, observations):
        return np.array([self.get_intrinsic_reward(obs) for obs in observations])

    def plot_value(self, episode=0, steps=0, chunk_size=1000):

        def get_chunks(x, n):
            for i in range(0, len(x), n):
                yield x[i:i+n]

        self._agent.eval_mode = True

        max_range = self._agent._replay.memory.cursor()

        if self._agent._replay.memory.is_full():
            max_range = self.info_buffer.replay_capacity

        valid_indices = np.zeros([], dtype=np.int32)
        for index in range(max_range):
            if self._agent._replay.memory.is_valid_transition(index):
                valid_indices = np.append(valid_indices, index)

        values = np.zeros(len(valid_indices))
        rooms = np.zeros(len(valid_indices), dtype=np.int16)
        player_x = np.zeros(len(valid_indices), dtype=np.int16)
        player_y = np.zeros(len(valid_indices), dtype=np.int16)

        index_chunks = get_chunks(valid_indices, chunk_size)
        current_idx = 0

        for index_chunk in index_chunks:
            current_chunk_size = len(index_chunk)
            transition_chunk = self._agent._replay.memory.sample_transition_batch(current_chunk_size, index_chunk)
            # first tuple element of transition is current state
            values[current_idx:current_idx+current_chunk_size] = self.value_function(transition_chunk[0])
            rooms[current_idx:current_idx+current_chunk_size] = self.info_buffer.get_indices('room_number', index_chunk)
            player_x[current_idx:current_idx+current_chunk_size] = self.info_buffer.get_indices('player_x', index_chunk)
            player_y[current_idx:current_idx+current_chunk_size] = self.info_buffer.get_indices('player_y', index_chunk)

            current_idx += current_chunk_size

        unique_rooms = np.unique(rooms)

        for room in unique_rooms:
            room_mask = (rooms == room)
            plt.scatter(player_x[room_mask], player_y[room_mask], c=values[room_mask], cmap='viridis')
            plt.colorbar()
            figname = self._get_plot_name(self._base_dir, 'value', str(room), str(episode), str(steps))
            plt.savefig(figname)
            plt.close()

    def plot_reward(self, episode=0, steps=0):
        self._agent.eval_mode = True

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

                if not room_number in rewards:
                    rewards[room_number] = []
                    player_x[room_number] = []
                    player_y[room_number] = []

                observation = stack[:,:,:,-1]

                rewards[room_number].append(self.reward_function(observation)[0])
                player_x[room_number].append(self.info_buffer.get_index('player_x', index))
                player_y[room_number].append(self.info_buffer.get_index('player_y', index))

        for key in rewards:
            plt.scatter(player_x[key], player_y[key], c=rewards[key],cmap='viridis')
            plt.colorbar()
            figname = self._get_plot_name(self._base_dir, 'reward', str(key), str(episode), str(steps))
            plt.savefig(figname)
            plt.clf()

    def _get_plot_name(self, base_dir, type, room, episode, steps):
        plot_dir = os.path.join(base_dir, 'plots', episode)
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        return os.path.join(plot_dir, '{}_room_{}_steps_{}.png'.format(type, room, steps))

    def _checkpoint_experiment(self, iteration):
        print(f"[RNDAgent] Iteration {iteration} Skipping checkpointing")
    
    def _save_tensorboard_summaries(self, iteration, num_episodes_train, average_reward_train, num_episodes_eval, average_reward_eval, average_steps_per_second):
        print(f"[RNDAgent] Iteration {iteration} Skipping Tensorboard logging")
