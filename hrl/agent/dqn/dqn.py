import ipdb
import numpy as np
import torch.nn as nn

import pfrl
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl import agents, explorers
from pfrl.wrappers import atari_wrappers
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead


class DQN:
    def __init__(self, n_actions, goal_conditioned, lr=2.5e-4,
                 start_eps=1.0, end_eps=0.1, decay_steps=10**6, 
                 replay_start_size=50_000, replay_buffer_size=10**6, 
                 target_update_interval=10**4, use_double_dqn=True, gpu=0):

        n_channels = 4 + int(goal_conditioned)
        self.goal_conditioned = goal_conditioned

        self.q_func = nn.Sequential(
            pnn.LargeAtariCNN(n_input_channels=n_channels),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
        
        self.opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
            self.q_func.parameters(),
            lr=lr,
            alpha=0.95,
            momentum=0.0,
            eps=1e-2,
            centered=True,
        )

        self.rbuf = replay_buffers.ReplayBuffer(replay_buffer_size)  # TODO: Change back to 1e6

        self.explorer = explorers.LinearDecayEpsilonGreedy(
            start_epsilon=start_eps,
            end_epsilon=end_eps,
            decay_steps=decay_steps,
            random_action_func=lambda: np.random.randint(n_actions),
        )

        policy_class = agents.DoubleDQN if use_double_dqn else agents.DQN

        self.agent = policy_class(
            q_function=self.q_func,
            optimizer=self.opt,
            replay_buffer=self.rbuf,
            gamma=0.99,
            explorer=self.explorer,
            gpu=gpu,
            replay_start_size=replay_start_size,
            target_update_interval=target_update_interval,
            clip_delta=True,
            update_interval=4,
            batch_accumulator="sum",
            phi=self.phi
        )

        self.T = 0

    @staticmethod
    def phi(x):
        """ Observation pre-processing for convolutional layers. """
        return np.asarray(x, dtype=np.float32) / 255.

    def act(self, state):
        """ Action selection method at the current state. """
        return self.agent.act(state)

    def step(self, state, action, reward, next_state, done, reset=False):
        """ Learning update based on a given transition from the environment. """
        self._overwrite_pfrl_state(state, action)
        self.agent.observe(next_state, reward, done, reset)

    def _overwrite_pfrl_state(self, state, action):
        """ Hack the pfrl state so that we can call act() consecutively during an episode before calling step(). """
        self.agent.batch_last_obs = [state]
        self.agent.batch_last_action = [action]

    def get_augmented_state(self, state, goal):
        features = list(state.frame._frames) + [goal.frame._frames[-1]]
        return atari_wrappers.LazyFrames(features, stack_axis=0)

    def experience_replay(self, trajectory):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        for transition in trajectory:
            self.step(*transition)

    def gc_experience_replay(self, trajectory, goal, rf):
        """ Add trajectory to the replay buffer and perform agent learning updates. """
        
        for state, action, _, next_state, done, reset in trajectory:
            augmented_state = self.get_augmented_state(state, goal)
            augmented_next_state = self.get_augmented_state(next_state, goal)
            reward, reached = rf(next_state, goal)
            relabeled_transition = augmented_state, action, reward, augmented_next_state, reached or done, reset
            self.step(*relabeled_transition)

    def rollout(self, env, state, episode, max_reward_so_far):
        """ Single episodic rollout of the agent's policy. """

        done = False
        episode_length = 0
        episode_reward = 0.
        episode_trajectory = []

        while not done:
            action = self.act(state)
            next_state, reward, done, info  = env.step(action)

            episode_trajectory.append((state,
                                       action,
                                       np.sign(reward), 
                                       next_state, 
                                       done, 
                                       info.get("needs_reset", False)))

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state

        self.experience_replay(episode_trajectory)
        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"Episode: {episode}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}")        

        return episode_reward, episode_length, max_reward_so_far

    def gc_rollout(self, env, state, goal, episode, max_reward_so_far, reward_func):
        """ Single goal-conditioned episodic rollout of the agent's policy. """

        done = False
        reached = False
        episode_length = 0
        episode_reward = 0.
        episode_trajectory = []

        while not done and not reached:
            # augmented_state = self.get_augmented_state(state, goal)
            action = self.act(state.frame)
            next_state, _, done, info  = env.execute_agent_action(action)

            reward, reached = reward_func(next_state, goal)

            episode_trajectory.append((state.frame,
                                       action,
                                       np.sign(reward), 
                                       next_state.frame, 
                                       done, 
                                       info.get("needs_reset", False)))

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state

        self.experience_replay(episode_trajectory)
        # self.gc_experience_replay(episode_trajectory, goal, reward_func)
        # self.gc_experience_replay(episode_trajectory, next_state, reward_func)
        
        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"[Goal {goal.position}] Episode: {episode}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}, Eps: {self.explorer.epsilon}")

        return episode_reward, episode_length, max_reward_so_far
