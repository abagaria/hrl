import numpy as np
from rainbow_RBFDQN.rainbow.RBFDQN_rainbow import Net
import torch
import torch.nn.functional as F
from common import utils, utils_for_q_learning

from hrl.agent.td3.replay_buffer import ReplayBuffer
from hrl.agent.td3.model import Actor, Critic, NormActor
from hrl.agent.td3.utils import *
from rainbow.dis import Net as DistributionalNets

import sys

class RBFWrapper(object):
    def __init__(
            self,
            env,
            state_dim,
            action_dim,
            max_action,
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
            name="Distributional RBF-DQN Classifier"
    ):

        params = utils_for_q_learning.get_hyper_parameters("umaze", "rbf")
        params['hyper_parameter_name'] = "umaze"
        params['experiment_name'] = 'test'
        params['run_title'] = 'test'
        params['distributional'] = True
        params['seed'] = 0
        params['per'] = False
        params['dueling'] = False
        params['nstep'] = False
        params['noisy_layers'] = False
        params['layer_normalization'] = False
        params['seed_number'] = 0
        params['random_betas'] = False
        params['num_layers_action_side'] = 2
        params['optimizer'] = "Adam"
        params['batch_size'] = 256

        # initialize dist rbf
        self.gamma = discount 
        self.device = device
        self.actor = DistributionalNets(params, env, state_size=state_dim, action_size=action_dim, device=device)
        self.target_actor = DistributionalNets(params, env, state_size=state_dim, action_size=action_dim, device=device)
        
        ### Don't need nor use the fields below ###
        self.max_action = max_action
        self.action_dim = action_dim
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.epsilon = exploration_noise
        self.device = device
        self.name = name
        self.use_output_normalization = use_output_normalization
        self.trained_options = []
        self.total_it = 0

    def act(self, state, evaluation_mode=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        _, selected_action, _ = self.actor.get_best_qvalue_and_action(state)

        if self.use_output_normalization:
            selected_action = self.normalize_actions(selected_action)

        selected_action = selected_action.cpu().data.numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.epsilon, size=self.action_dim)
        if not evaluation_mode:
            selected_action += noise
        return selected_action.clip(-self.max_action, self.max_action)

    def step(self, state, action, reward, next_state, is_terminal):
        self.actor.buffer_object.append(state, action, reward, is_terminal, next_state)
        self.actor.update(self.target_actor)

    def get_values(self, states):
        with torch.no_grad():
            return self.actor.get_best_qvalue_and_action(states)[0]


