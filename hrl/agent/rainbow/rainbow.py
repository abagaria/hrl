from typing import Tuple
import torch
import pdb
import random
import numpy as np
from tqdm import tqdm
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl import agents, explorers
from pfrl.wrappers import atari_wrappers
from pfrl.q_functions import DistributionalDuelingDQN

from hrl.tasks.monte.MRRAMMDPClass import MontezumaRAMMDP


class Rainbow:
    def __init__(self, n_actions, n_atoms, v_min, v_max, noisy_net_sigma, lr, 
                 n_steps, betasteps, replay_start_size, replay_buffer_size, gpu,
                 goal_conditioned, use_her):
        self.use_her = use_her
        self.n_actions = n_actions
        n_channels = 4 + int(goal_conditioned)
        self.goal_conditioned = goal_conditioned

        self.my_dict = {}

        self.q_func = DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=n_channels)
        pnn.to_factorized_noisy(self.q_func, sigma_scale=noisy_net_sigma)

        explorer = explorers.Greedy()
        opt = torch.optim.Adam(self.q_func.parameters(), lr, eps=1.5e-4)

        self.rbuf = replay_buffers.PrioritizedReplayBuffer(
            replay_buffer_size,
            alpha=0.5, 
            beta0=0.4,
            betasteps=betasteps,
            num_steps=n_steps,
            normalize_by_max="memory"
        )

        self.agent = agents.CategoricalDoubleDQN(
            self.q_func,
            opt,
            self.rbuf,
            gpu=gpu,
            gamma=0.99,
            explorer=explorer,
            minibatch_size=32,
            replay_start_size=replay_start_size,
            target_update_interval=32_000,
            update_interval=4,
            batch_accumulator="mean",
            phi=self.phi
        ) # 84, 84, 4, 4,84,84

        self.T = 0
        self.device = torch.device(f"cuda:{gpu}" if gpu > -1 else "cpu")

    @staticmethod
    def phi(x):
        """ Observation pre-processing for convolutional layers. """
        return np.asarray(x, dtype=np.float32) / 255.

    def act(self, state):
        """ Action selection method at the current state. """
        return self.agent.act(state)

    def step(self, state, action, reward, next_state, done, reset):
        """ Learning update based on a given transition from the environment. """
        self._overwrite_pfrl_state(state, action)
        self.agent.observe(next_state, reward, done, reset)

    def _overwrite_pfrl_state(self, state, action):
        """ Hack the pfrl state so that we can call act() consecutively during an episode before calling step(). """
        self.agent.batch_last_obs = [state]
        self.agent.batch_last_action = [action]

    @torch.no_grad()
    def value_function(self, states):
        batch_states = self.agent.batch_states(states, self.device, self.phi)
        action_values = self.agent.model(batch_states).q_values
        return action_values.max(dim=1).values

    def gc_experience_replay(self, trajectory, goal, goal_position):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol
        
        for state, action, reward, next_state, done, reset, next_pos in trajectory:
            augmented_state = self.get_augmented_state(state, goal)
            augmented_next_state = self.get_augmented_state(next_state, goal)
            relabeled_transition = augmented_state, action, reward, augmented_next_state, done, reset
            self.step(*relabeled_transition)
            if done: break  # it helps to truncate the trajectory for HER strategy `future`

    def get_augmented_state(self, s, g):
        assert isinstance(g, (np.ndarray, atari_wrappers.LazyFrames)), type(g)
        return atari_wrappers.LazyFrames(s._frames+[g._frames[-1]], stack_axis=0)

    def gc_rollout(self, mdp:MontezumaRAMMDP, goal_img, goal_position: Tuple, episode, max_reward_so_far, limit=500):
        """ Single episodic rollout of the agent's policy. """

        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        def rf():
            ram = mdp.curr_state.ram
            p1 = mdp.curr_state.get_position(ram)
            p2 = goal_position
            d = is_close(p1, p2, 5)
            return float(d), d

        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        episode_positions = []
        episode_trajectory = []
        ram_trajectory = []

        while not done and not reset and not reached and episode_length < limit:
            sg = self.get_augmented_state(mdp.curr_state.image, goal_img)
            action = self.act(sg)

            if action in self.my_dict:
                self.my_dict[action] += 1
            else:
                self.my_dict[action] = 1
            prev_state = mdp.curr_state
            mdp.execute_agent_action(action)
            reward, reached = rf()

            ram = mdp.curr_state.ram
            player_pos = mdp.curr_state.get_position(ram)
            ram_trajectory.append(ram)
            episode_positions.append(player_pos)
            episode_trajectory.append(
                                      (prev_state.image,
                                       action,
                                       reward, 
                                       mdp.curr_state.image, 
                                       done or reached, 
                                       reset,
                                       player_pos
                                    )
                                )

            self.T += 1
            episode_length += 1
            episode_reward += reward

        if self.goal_conditioned:
            self.gc_experience_replay(episode_trajectory, goal_img, goal_position)
        else:
            pass

        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"Episode: {episode}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}")        

        return episode_reward, episode_length, max_reward_so_far, episode_trajectory, ram_trajectory
    
    # def her(self, trajectory, visited_positions, pursued_goal, pursued_goal_position=(123, 148)):
    #     hindsight_goal, hindsight_goal_idx = self.pick_hindsight_goal(trajectory)
    #     self.gc_experience_replay(trajectory, pursued_goal, pursued_goal_position)
    #     self.gc_experience_replay(trajectory, hindsight_goal, visited_positions[hindsight_goal_idx])

    # def pick_hindsight_goal(self, trajectory, strategy="future"):
    #     """ Select a hindsight goal from the input trajectory. """
    #     assert strategy in ("final", "future"), strategy
        
    #     goal_idx = -1
        
    #     if strategy == "future":
    #         start_idx = len(trajectory) // 2
    #         goal_idx = random.randint(start_idx, len(trajectory) - 1)

    #     goal_transition = trajectory[goal_idx]
    #     goal_state = goal_transition[3]
    #     assert isinstance(goal_state, atari_wrappers.LazyFrames), type(goal_state)
    #     return goal_state, goal_idx