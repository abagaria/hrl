import os
import ipdb
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from tqdm import tqdm
from hrl.utils import flatten
from sklearn.svm import OneClassSVM, SVC
from .flipping_classifier import FlippingClassifier
from .critic_classifier import CriticInitiationClassifier
from .position_classifier import PositionInitiationClassifier

class DistributionalCriticClassifier(PositionInitiationClassifier):

    def __init__(
        self,
        agent,
        use_position,
        goal_sampler,
        augment_func,
        optimistic_threshold,
        pessimistic_threshold,
        option_name,
        maxlen=100,
        resample_goals=False,
        threshold=None,
        device=None
    ):
        self.agent = agent
        self.use_position = use_position
        self.device = device

        self.critic_classifier = CriticInitiationClassifier(
            agent,
            goal_sampler,
            augment_func,
            optimistic_threshold,
            pessimistic_threshold
        )

        self.option_name = option_name
        self.resample_goals = resample_goals
        self.threshold = threshold

        self.optimistic_predict_thresh = 0.4
        self.pessimistic_predict_thresh = 0.6

        super().__init__(maxlen)
        

    def get_weights(self, threshold, states): 
        '''
        Given state, threshold, value function, compute the flipping prob for each state
        Return 1/flipping prob which is the weights
            The formula for weights is a little more complicated, see paper Akhil will send in 
            channel
        return shape: (states.shape, ) 
        '''

        # Predict the probability that the samples will flip
        with torch.no_grad():
            states = states.to(self.device).to(torch.float32)
            best_actions = self.critic_classifier.agent.actor.get_best_qvalue_and_action(states.to(torch.float32))[1]
            best_actions = best_actions.to(self.device).to(torch.float32)
            distribution = self.critic_classifier.agent.actor.forward(states, best_actions)
            distribution = distribution - threshold
            distribution = (distribution >= 0)
            num_supports = (distribution.shape[1])
            probabilities = distribution.sum(axis=1)/num_supports
            weights = 1. / (probabilities + 1e-4)
        return weights

    def add_positive_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(sg) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_positive_examples(states, infos)
    
    def add_negative_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(s) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_negative_examples(states, infos)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        positions = [example.pos for example in examples]
        return np.array(positions)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))
        W = self.get_sample_weights(plot=True)

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y, sample_weight=W)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            self.pessimistic_classifier.fit(positive_training_examples)

    def get_sample_weights(self, plot=False):

        pos_egs = flatten(self.positive_examples)
        neg_egs = flatten(self.negative_examples)
        examples = pos_egs + neg_egs
        assigned_labels = np.concatenate((
            np.ones((len(pos_egs),)),
            np.zeros((len(neg_egs),))
        ))

        # Extract what labels the old VF *would* have assigned
        old_values = np.array([eg.info["value"].cpu() for eg in examples]).squeeze()
        old_nsteps = self.critic_classifier.value2steps(old_values)
        old_critic_labels = self.critic_classifier.pessimistic_classifier(old_nsteps)

        # Extract what labels the current VF would have assigned
        augmented_states = np.array([eg.info["augmented_state"] for eg in examples])

        if self.resample_goals:
            observations = augmented_states[:, :-2]
            new_goal = self.critic_classifier.goal_sampler()[np.newaxis, ...]
            new_goals = np.repeat(new_goal, axis=0, repeats=observations.shape[0])
            augmented_states = np.concatenate((observations, new_goals), axis=1)

        new_values = self.agent.get_values(torch.from_numpy(augmented_states).to(self.device).to(torch.float32)).squeeze()
        new_nsteps = self.critic_classifier.value2steps(new_values.cpu().detach().numpy())
        new_critic_labels = self.critic_classifier.optimistic_classifier(new_nsteps)
        # Compute the weights based on the probability that the samples will flip
        weights = self.get_weights(self.threshold, torch.from_numpy(augmented_states))
        return weights.detach().cpu()

   
    def plot_initiation_classifier(self, env, option_name, episode, experiment_name, seed):
        print(f"Plotting Critic Initiation Set Classifier for {option_name}")

        chunk_size = 1000
        replay_buffer = self.agent.actor.buffer_object.storage

        # Take out the original goal
        trans = replay_buffer.get_all_transitions()
        states = trans['obs']
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
            breakpoint()
            chunk_values = self.agent.get_values(torch.from_numpy(state_chunk).to(self.device).to(torch.float32)).squeeze()
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
