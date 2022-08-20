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

        self.replay_buffer = self.actor.buffer_object
        
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

    def get_qvalues(self, states, actions):
        with torch.no_grad():
            return self.actor.forward(states, actions)

    def get_value_distribution(self, states):
        """ Get the value distribution for the input states. """
        states = torch.as_tensor(states).float().to(self.device)
        actions = self.actor.get_best_qvalue_and_action(states)[1]
        distribution = self.actor.forward(states, actions)
        return distribution 

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        print(f"Plotting Critic Initiation Set Classifier for {option_name}")

        chunk_size = 1000

        # Take out the original goal
        states = [exp[0] for exp in replay_buffer]
        states = [state[:-2] for state in states]
        if len(states) > 100_000:
            print(f"Subsampling {len(states)} s-a pairs to 100,000")
            idx = np.random.randint(0, len(states), size=100_000)
            states = [states[i] for i in idx]

        print(f"preparing {len(states)} states")
        states = np.array(states)

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))

        if num_chunks == 0:
            return 0.

        print("chunking")
        state_chunks = np.array_split(states, num_chunks, axis=0)
        steps = np.zeros((states.shape[0],))
        
        optimistic_predictions = np.zeros((states.shape[0],))
        pessimistic_predictions = np.zeros((states.shape[0],))

        current_idx = 0

        for state_chunk in tqdm(state_chunks, desc="Plotting Critic Init Classifier"):
            chunk_values = self.value_function(state_chunk)
            chunk_steps = self.value2steps(chunk_values).squeeze()
            current_chunk_size = len(state_chunk)

            steps[current_idx:current_idx + current_chunk_size] = chunk_steps
            optimistic_predictions[current_idx:current_idx + current_chunk_size] = self.optimistic_classifier(chunk_steps)
            pessimistic_predictions[current_idx:current_idx + current_chunk_size] = self.pessimistic_classifier(chunk_steps)

            current_idx += current_chunk_size
        
        print("plotting")
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 3, 1)
        plt.scatter(states[:, 0], states[:, 1], c=steps)
        plt.title(f"nSteps to termination region")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.scatter(states[:, 0], states[:, 1], c=optimistic_predictions)
        plt.title(f"Optimistic Classifier")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.scatter(states[:, 0], states[:, 1], c=pessimistic_predictions)
        plt.title(f"Pessimistic Classifier")
        plt.colorbar()

        plt.suptitle(f"{option_name}")
        file_name = f"{option_name}_critic_init_clf_{seed}_episode_{episode}"
        saving_path = os.path.join('results', experiment_name, 'initiation_set_plots', f'{file_name}.png')

        print("saving")
        plt.savefig(saving_path)
        plt.close()
