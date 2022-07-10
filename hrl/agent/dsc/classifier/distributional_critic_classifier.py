import os
import ipdb
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt

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
        resample_goals=False
    ):
        self.agent = agent
        self.use_position = use_position

        self.critic_classifier = CriticInitiationClassifier(
            agent,
            goal_sampler,
            augment_func,
            optimistic_threshold,
            pessimistic_threshold
        )

        self.option_name = option_name
        self.resample_goals = resample_goals

        super().__init__(maxlen)

    # TODO:
    # look at state_critic_bayes_classifier.py for inspiration. 
    # key lines of code (102)
    # Predict the probability that the samples will flip
    #probabilities = self.flipping_classifier(examples)
    #weights = 1. / (probabilities + 1e-4)
    def get_weights(self, threshold, states): 
        '''
        Given state, threshold, value function, compute the flipping prob for each state
        Return 1/flipping prob which is the weights
            The formula for weights is a little more complicated, see paper Akhil will send in 
            channel
        return shape: (states.shape, ) 
        '''

        # Predict the probability that the samples will flip
        distribution = self.critic_classifier.agent.forward(states)
        distribution = distribution - threshold
        distribution = (distribution >= 0)
        probabilities = distribution.sum(axis=1)
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
        old_values = np.array([eg.info["value"] for eg in examples]).squeeze()
        old_nsteps = self.critic_classifier.value2steps(old_values)
        old_critic_labels = self.critic_classifier.pessimistic_classifier(old_nsteps)

        # Extract what labels the current VF would have assigned
        augmented_states = np.array([eg.info["augmented_state"] for eg in examples])

        if self.resample_goals:
            observations = augmented_states[:, :-2]
            new_goal = self.critic_classifier.goal_sampler()[np.newaxis, ...]
            new_goals = np.repeat(new_goal, axis=0, repeats=observations.shape[0])
            augmented_states = np.concatenate((observations, new_goals), axis=1)

        new_values = self.agent.get_values(augmented_states).squeeze()
        new_nsteps = self.critic_classifier.value2steps(new_values)
        new_critic_labels = self.critic_classifier.optimistic_classifier(new_nsteps)

        # Train the flip predictor
        self.flipping_classifier.fit(
            examples,
            assigned_labels,
            old_critic_labels,
            new_critic_labels
        )
        
        # Predict the probability that the samples will flip
        probabilities = self.flipping_classifier(examples)
        weights = 1. / (probabilities + 1e-4)

        if plot:
            self.plot_flipping_labels(
                examples,
                assigned_labels, 
                old_critic_labels, 
                new_critic_labels
            )

            self.plot_flipping_probabilities(
                examples,
                assigned_labels,
                old_critic_labels,
                new_critic_labels,
                probabilities
            )

        return weights

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            return xx, yy

        def get_grid_states(mdp):
            ss = []
            x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
            x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
            for x in np.arange(x_low_lim, x_high_lim+1, 1):
                for y in np.arange(y_low_lim, y_high_lim+1, 1):
                    ss.append(np.array((x, y)))
            return ss

        def get_initiation_set_values(mdp):
            values = []
            x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
            x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
            for x in np.arange(x_low_lim, x_high_lim+1, 1):
                for y in np.arange(y_low_lim, y_high_lim+1, 1):
                    pos = np.array((x, y))
                    init = self.optimistic_predict(pos)
                    if hasattr(mdp.env, 'env'):
                        init = init and not mdp.env.env._is_in_collision(pos)
                    values.append(init)
            return values
