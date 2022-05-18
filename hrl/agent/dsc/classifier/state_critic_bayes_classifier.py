import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from hrl.utils import flatten

from .mlp_classifier import BinaryMLPClassifier
from .flipping_classifier import FlippingClassifier
from .obs_init_classifier import ObsInitiationClassifier
from .critic_classifier import CriticInitiationClassifier


class ObsCriticBayesIinitiationClassifier(ObsInitiationClassifier):
    def __init__(self,
        obs_dim,
        agent,
        goal_sampler,
        augment_func,
        option_name,
        optimistic_threshold=40,
        pessimistic_threshold=20,
        maxlen=1000
    ):
        super().__init__(
            obs_dim,
            agent.device,
            optimistic_threshold,
            pessimistic_threshold,
            maxlen=maxlen
        )

        self.critic_classifier = CriticInitiationClassifier(
            agent,
            goal_sampler,
            augment_func,
            optimistic_threshold,
            pessimistic_threshold
        )

        self.flipping_classifier = FlippingClassifier(
            obs_dim=obs_dim,
            device=agent.device,
            classifier_type="nn",
            feature_extractor_type="obs"
        )

        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)

        self.option_name = option_name

    def train_two_class_classifier(self):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = torch.ones((positive_feature_matrix.shape[0],), device=self.device)
        negative_labels = torch.zeros((negative_feature_matrix.shape[0],), device=self.device)

        X = torch.cat((positive_feature_matrix, negative_feature_matrix))
        Y = torch.cat((positive_labels, negative_labels))

        if self.classifier.should_train(Y):
            # Re-train the common classifier from scratch
            self.classifier = BinaryMLPClassifier(self.obs_dim, self.device, threshold=None)
            
            W = self.get_sample_weights()
            self.classifier.fit(X, Y, W=W)

            # Re-point the classifiers to the same object in memory
            self.optimistic_classifier = self.classifier
            self.pessimistic_classifier = self.classifier

    def get_sample_weights(self):
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
        new_values = self.critic_classifier.agent.get_values(augmented_states).squeeze()
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

        return weights

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        self.plot_training_predictions(self.option_name, episode, experiment_name, seed)

        if hasattr(env, "observations"):
            self.plot_testing_predictions(env, replay_buffer, self.option_name, episode, experiment_name, seed)

    def plot_training_predictions(self, option_name, episode, experiment_name, seed, goal=None):
        """ Plot the predictions on the traininng data. """
        if not self.is_initialized():
            return
        
        x_positive = self.construct_feature_matrix(self.positive_examples)
        x_negative = self.construct_feature_matrix(self.negative_examples)

        sample_weights = self.get_sample_weights().cpu().numpy()
        w_positive = sample_weights[:x_positive.shape[0], :]
        w_negative = sample_weights[x_positive.shape[0]:, :]

        optimistic_positive_predictions = self.optimistic_classifier.predict(x_positive, threshold=0.5) == 1
        pessimistic_positive_predictions = self.pessimistic_classifier.predict(x_positive, threshold=0.75) == 1

        optimistic_negative_predictions = self.optimistic_classifier.predict(x_negative, threshold=0.5) == 1
        pessimistic_negative_predictions = self.pessimistic_classifier.predict(x_negative, threshold=0.75) == 1

        optimistic_positive_predictions = optimistic_positive_predictions.cpu().numpy()
        pessimistic_positive_predictions = pessimistic_positive_predictions.cpu().numpy()
        optimistic_negative_predictions = optimistic_negative_predictions.cpu().numpy()
        pessimistic_negative_predictions = pessimistic_negative_predictions.cpu().numpy()

        positive_positions = self.extract_positions(self.positive_examples)
        negative_positions = self.extract_positions(self.negative_examples)

        plt.figure(figsize=(16, 10))
        plt.subplot(1, 2, 1)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=optimistic_positive_predictions, marker="+", label="positive data",
                    s=w_positive)
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1],
                    c=optimistic_negative_predictions, marker="o", label="negative data",
                    s=w_negative)
        plt.clim(0, 1)
        plt.colorbar()
        plt.legend()
        plt.title("Optimistic classifier")

        plt.subplot(1, 2, 2)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=pessimistic_positive_predictions, marker="+", label="positive data")
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1], 
                    c=pessimistic_negative_predictions, marker="o", label="negative data")
        plt.clim(0, 1)
        plt.colorbar()
        plt.legend()
        plt.title("Pessimistic classifier")
        
        if goal:
            plt.suptitle(f"Targeting {goal}")

        plt.savefig(f"results/{experiment_name}/initiation_set_plots/{option_name}_init_clf_seed_{seed}_episode_{episode}.png")
        plt.close()

    def plot_testing_predictions(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        return super().plot_testing_predictions(env, replay_buffer, option_name, episode, experiment_name, seed)
