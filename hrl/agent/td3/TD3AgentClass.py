import math
import numpy as np
import torch
import torch.nn.functional as F
import ipdb

from hrl.agent.td3.replay_buffer import ReplayBuffer
from hrl.agent.td3.model import Actor, Critic, NormActor
from hrl.agent.td3.utils import *
from hrl.experiments.distance_learning_network import DistanceLearner


# Adapted author implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            distance_function_quantile,
            use_output_normalization=True,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            batch_size=256,
            exploration_noise=0.1,
            lr_c=3e-4, lr_a=3e-4,
            device=torch.device("cuda"),
            name="Global-TD3-Agent",
            store_extra_info=True    
        ):

        self.critic_learning_rate = lr_c
        self.actor_learning_rate = lr_a

        if use_output_normalization:
            assert max_action == 1., "Haven't fixed max-action for output-norm yet"
            self.actor = NormActor(state_dim, action_dim).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)

        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)

        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.epsilon = exploration_noise
        self.device = device
        self.name = name
        self.store_extra_info = store_extra_info
        self.use_output_normalization = use_output_normalization

        self.trained_options = []

        self.T = 0

        self.distance_function = DistanceLearner(state_dim, batch_size, learning_rate=3e-4, device=device, quantile=distance_function_quantile)

    def act(self, state, evaluation_mode=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        selected_action = self.actor(state)

        if self.use_output_normalization:
            selected_action = self.normalize_actions(selected_action)

        selected_action = selected_action.cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.epsilon, size=self.action_dim)
        if not evaluation_mode:
            selected_action += noise
        return selected_action.clip(-self.max_action, self.max_action)

    def normalize_actions(self, actions):

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        K = torch.tensor(self.action_dim).to(self.device)
        G = torch.sum(torch.abs(actions), dim=1).view(-1, 1)
        G = G / K

        ones = torch.ones(G.size()).to(self.device)
        G_mod = torch.where(G >= 1, G, ones)

        normalized_actions = actions / G_mod

        return normalized_actions

    def step(self, state, action, reward, next_state, is_terminal, reset=False):
        info = self.compute_extra_info(state, next_state) if self.store_extra_info else None
        self.replay_buffer.add(state, action, reward, next_state, is_terminal, info=info)

        if len(self.replay_buffer) > self.batch_size:
            # update TD3 agent
            self.train(self.replay_buffer, self.batch_size)

            # update distance function
            self.distance_function.train_loop(self.replay_buffer.sample_state_pairs(self.batch_size))

    def compute_extra_info(self, state, next_state):
        s = state[np.newaxis, ...]
        sp = next_state[np.newaxis, ...]
        states = np.concatenate((s, sp), axis=0)
        values = self.get_values(states)
        next_action = self.get_actions(sp)
        return dict(
            values=values,
            next_action=next_action
        )

    def train(self, replay_buffer, batch_size):
        self.T += 1

        # Sample replay buffer - result is tensors
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            target_actions = self.target_actor(next_state)

            if self.use_output_normalization:
                target_actions = self.normalize_actions(target_actions)

            next_action = (
                    target_actions + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.T % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_epsilon(self):
        """ We are using fixed (default) epsilons for TD3 because tuning it is hard. """
        pass

    def get_qvalues(self, states, actions):
        """ Get the values associated with the input state-action pairs. """

        self.critic.eval()
        with torch.no_grad():
            q1, q2 = self.critic(states, actions)
        self.critic.train()
        return torch.min(q1, q2)

    def get_values(self, states):
        """ Get the values associated with the input states. """

        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states).float().to(self.device)

        with torch.no_grad():
            actions = self.actor(states)
            if self.use_output_normalization:
                actions = self.normalize_actions(actions)
                actions = actions.clamp(-self.max_action, self.max_action)
            q_values = self.get_qvalues(states, actions)
        return q_values.cpu().numpy()

    @torch.no_grad()
    def get_actions(self, states):
        if isinstance(states, np.ndarray):
            states = torch.as_tensor(states).float().to(self.device)
        actions = self.actor(states)
        return actions.cpu().numpy()

    def experience_replay(self, trajectory):
        """ Add trajectory to the replay buffer and perform agent learning updates. """
        for transition in trajectory:
            self.step(*transition)

    def rollout(self, env, state, episode):
        """ Single episode of interaction with the env followed by experience replay. """
        done = False
        reset = False
        reached = False
        episode_length = 0
        episode_reward = 0.
        episode_trajectory = []

        goal_state = self.select_goal_state()

        while not done and not reset and not reached:
            action = self.act(state)
            next_state, reward, done, info = env.step(action)
            reset = info.get("needs_reset", False)
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
        
        # update TD3 policy
        episode_reward = sum(transition[2] for transition in episode_trajectory)
        episode_trajectory = self.assign_episode_rewards_from_distance_function(goal_state, episode_trajectory)
        self.experience_replay(episode_trajectory)

        # update distance function
        states = [transition[0] for transition in episode_trajectory]
        self.distance_function.experience_replay(states)

        # log progress
        distance_reward = sum(transition[2] for transition in episode_trajectory)
        self.log_progress(env, episode, episode_reward, distance_reward, episode_length, goal_state)
        return episode_reward, episode_length

    def log_progress(self, env, episode, episode_reward, distance_reward, episode_length, goal_state):
        goal_state = goal_state.round(decimals=2)
        print(f"Episode: {episode}, Goal: {goal_state}")
        print(f"True Reward: {episode_reward}, Distance Reward: {distance_reward}, Length: {episode_length}")

    def distance_reward_function(self, s, goal):
        assert isinstance(s, np.ndarray)
        assert isinstance(goal, np.ndarray)
        # input is batched
        assert len(s.shape) == 2, s.shape
        assert len(goal.shape) == 2, goal.shape

        ipdb.set_trace()
        sg = np.concatenate((s, goal), axis=1)
        sg = torch.as_tensor(sg).float().to(self.device)
        distances = self.distance_function.forward(sg)
        distances[distances < 0] = 0
        rewards = -distances / self.max_distance
        return rewards.cpu().numpy()

    def select_goal_state(self):
        """Get state with largest x value"""
        ipdb.set_trace()
        ind = np.argmax(self.replay_buffer.state, axis=0)[0]
        return self.replay_buffer.state[ind]

    def assign_episode_rewards_from_distance_function(self, goal_state, episode_trajectory):
        next_states = [transition[3] for transition in episode_trajectory]
        goal_state = goal_state[np.newaxis, ...]
        goal_states = np.repeat(goal_state, len(next_states), axis=0)
        next_states = np.array(next_states)
        rewards = self.distance_reward_function(next_states, goal_states)

        episode_trajectory = [(s, a, r, ns, done, info) for r, (s, a, _, ns, done, info) in zip(rewards, episode_trajectory)]
        return episode_trajectory
