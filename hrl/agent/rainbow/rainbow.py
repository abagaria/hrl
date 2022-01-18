import ipdb
import torch
import random
import numpy as np
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl import agents, explorers
from pfrl.wrappers import atari_wrappers
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl.utils import batch_states as pfrl_batch_states


class Rainbow:
    def __init__(self, n_actions, n_atoms, v_min, v_max, noisy_net_sigma, lr, 
                 n_steps, betasteps, replay_start_size, replay_buffer_size, gpu,
                 goal_conditioned, use_custom_batch_states=True):
        self.n_actions = n_actions
        n_channels = 4 + int(goal_conditioned)
        self.goal_conditioned = goal_conditioned
        self.use_custom_batch_states = use_custom_batch_states

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
            phi=self.phi,
            batch_states=self.batch_states if use_custom_batch_states else pfrl_batch_states
        )

        self.T = 0
        self.device = torch.device(f"cuda:{gpu}" if gpu > -1 else "cpu")

    @staticmethod
    def batch_states(states, device, phi):
        assert isinstance(states, list), type(states)
        assert isinstance(states[0], atari_wrappers.LazyFrames), type(states[0])
        features = np.array([phi(s) for s in states])
        return torch.as_tensor(features).to(device)

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

    def experience_replay(self, trajectory):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        for transition in trajectory:
            self.step(*transition)

    def gc_experience_replay(self, trajectory, goal, goal_position):
        """ Add trajectory to the replay buffer and perform agent learning updates. """

        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        def rf(pos, goal_pos):
            pos = np.array([pos['player_x'], pos['player_y']]) if isinstance(pos, dict) else pos
            d = is_close(pos, goal_pos, tol=2)
            return float(d), d
        
        for state, action, _, next_state, done, reset, next_pos in trajectory:
            augmented_state = self.get_augmented_state(state, goal)
            augmented_next_state = self.get_augmented_state(next_state, goal)
            reward, reached = rf(next_pos, goal_position)
            relabeled_transition = augmented_state, action, reward, augmented_next_state, reached or done, reset
            self.step(*relabeled_transition)
            if reached: break  # it helps to truncate the trajectory for HER strategy `future`

    def get_augmented_state(self, state, goal):
        assert isinstance(goal, atari_wrappers.LazyFrames), type(goal)
        assert isinstance(state, atari_wrappers.LazyFrames), type(state)
        features = list(state._frames) + [goal._frames[-1]]
        return atari_wrappers.LazyFrames(features, stack_axis=0)

    def rollout(self, env, state, episode, max_reward_so_far):
        """ Single episodic rollout of the agent's policy. """

        def is_close(pos1, pos2, tol):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        def rf(info_dict):
            p1 = info_dict["player_x"], info_dict["player_y"]
            p2 = 123, 148
            d = is_close(p1, p2, 2)
            return float(d), d

        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        episode_trajectory = []

        while not done and not reset and not reached:
            action = self.act(state)
            next_state, reward, done, info  = env.step(action)
            reset = info.get("needs_reset", False)

            reward, reached = rf(info)

            episode_trajectory.append((state,
                                       action,
                                       np.sign(reward), 
                                       next_state, 
                                       done or reached, 
                                       reset))

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state

        self.experience_replay(episode_trajectory)
        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"Episode: {episode}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}")        

        return episode_reward, episode_length, max_reward_so_far

    def gc_rollout(self, env, state, goal, goal_pos, rf, episode, max_reward_so_far):
        """ Single episodic rollout of the agent's policy. """

        info = {}
        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        episode_positions = []
        episode_trajectory = []

        while not done and not reset and not reached:
            sg = self.get_augmented_state(state, goal)
            action = self.act(sg)
            next_state, reward, done, info  = env.step(action)
            reset = info.get("needs_reset", False)

            reward, reached = rf(info, goal_pos)

            player_pos = info["player_x"], info["player_y"]
            episode_positions.append(player_pos)
            episode_trajectory.append(
                                      (state,
                                       action,
                                       np.sign(reward), 
                                       next_state, 
                                       done or reached, 
                                       reset,
                                       player_pos
                                    )
                                )

            self.T += 1
            episode_length += 1
            episode_reward += reward

            state = next_state

        self.her(episode_trajectory, episode_positions, goal, pursued_goal_position=goal_pos)

        max_reward_so_far = max(episode_reward, max_reward_so_far)
        print(f"G: {goal_pos}, T: {self.T}, Reward: {episode_reward}, Max reward: {max_reward_so_far}")        

        return episode_reward, episode_length, max_reward_so_far, done or reset
    
    def her(self, trajectory, visited_positions, pursued_goal, pursued_goal_position):
        hindsight_goal, hindsight_goal_idx = self.pick_hindsight_goal(trajectory)
        self.gc_experience_replay(trajectory, pursued_goal, pursued_goal_position)
        self.gc_experience_replay(trajectory, hindsight_goal, visited_positions[hindsight_goal_idx])

    def pick_hindsight_goal(self, trajectory, strategy="future"):
        """ Select a hindsight goal from the input trajectory. """
        assert strategy in ("final", "future"), strategy
        
        goal_idx = -1
        
        if strategy == "future":
            start_idx = len(trajectory) // 2
            goal_idx = random.randint(start_idx, len(trajectory) - 1)

        goal_transition = trajectory[goal_idx]
        goal_state = goal_transition[3]
        assert isinstance(goal_state, atari_wrappers.LazyFrames), type(goal_state)
        return goal_state, goal_idx
