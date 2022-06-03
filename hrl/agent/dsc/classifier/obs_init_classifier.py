import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from hrl.utils import chunked_inference
from .mlp_classifier import BinaryMLPClassifier
from .init_classifier import InitiationClassifier
from hrl.agent.dsc.datastructures import TrainingExample


class ObsInitiationClassifier(InitiationClassifier):
    def __init__(self, obs_dim, device, optimistic_threshold=0.5, pessimistic_threshold=0.75, maxlen=1000):
        self.classifier = BinaryMLPClassifier(obs_dim, device, threshold=None)
        self.optimistic_threshold = optimistic_threshold
        self.pessimistic_threshold = pessimistic_threshold
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)

        self.device = device
        self.obs_dim = obs_dim
        optimistic_classifier = self.classifier
        pessimistic_classifier = self.classifier
        
        super().__init__(optimistic_classifier, pessimistic_classifier)

    def is_initialized(self):
        return self.optimistic_classifier.is_trained and \
               self.pessimistic_classifier.is_trained

    @torch.no_grad()
    def batched_optimistic_predict(self, X):
        assert isinstance(self.optimistic_classifier, BinaryMLPClassifier)
        assert self.optimistic_classifier.is_trained
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).float().to(self.device)
        optimistic_predictions = self.optimistic_classifier.predict(X, threshold=0.5)
        return optimistic_predictions.cpu().numpy()

    @torch.no_grad()
    def batched_pessimistic_predict(self, X):
        assert isinstance(self.pessimistic_classifier, BinaryMLPClassifier)
        assert self.pessimistic_classifier.is_trained
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).float().to(self.device)
        pessimistic_predictions = self.pessimistic_classifier.predict(X, threshold=0.75)
        return pessimistic_predictions.cpu().numpy()

    def optimistic_predict(self, state):
        assert isinstance(state, (np.ndarray, torch.Tensor)), state
        assert isinstance(self.optimistic_classifier, BinaryMLPClassifier)
        assert self.optimistic_classifier.is_trained
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state).float().to(self.device)
        label = self.optimistic_classifier.predict(state, threshold=0.5) == 1
        return label.cpu().numpy()

    def pessimistic_predict(self, state):
        assert isinstance(state, (np.ndarray, torch.Tensor)), state
        assert isinstance(self.pessimistic_classifier, BinaryMLPClassifier)
        assert self.pessimistic_classifier.is_trained
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state).float().to(self.device)
        label = self.pessimistic_classifier.predict(state, threshold=0.75) == 1
        return label.cpu().numpy()

    def get_false_positive_rate(self):
        """ Fraction of the negative data that is classified as positive. """ 

        negative_examples = self.construct_feature_matrix(self.negative_examples)
        
        if len(negative_examples) > 0:
            optimistic_pred = self.optimistic_classifier.predict(negative_examples, threshold=0.5).cpu().numpy()
            pessimistic_pred = self.pessimistic_classifier.predict(negative_examples, threshold=0.75).cpu().numpy()

            return np.array(
                optimistic_pred.mean(), pessimistic_pred.mean()
            )

        return np.array([1., 1.])
    
    def add_positive_examples(self, observations, infos):
        assert len(observations) == len(infos)

        positive_examples = [TrainingExample(obs, info) for obs, info in zip(observations, infos)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, observations, infos):
        assert len(observations) == len(infos)

        negative_examples = [TrainingExample(obs, info) for obs, info in zip(observations, infos)]
        self.negative_examples.append(negative_examples)

    def construct_feature_matrix(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
        observations = np.array([example.obs for example in examples])
        obs_tensor = torch.as_tensor(observations).float().to(self.device)
        return obs_tensor

    @staticmethod
    def extract_positions(examples):
        examples = itertools.chain.from_iterable(examples)
        positions = [example.pos for example in examples]
        return np.array(positions)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
    
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
            self.classifier.fit(X, Y)

            # Re-point the classifiers to the same object in memory
            self.optimistic_classifier = self.classifier
            self.pessimistic_classifier = self.classifier

    def sample(self):
        """ Sample from the pessimistic initiation classifier. """
        num_tries = 0
        sampled_state = None
        while sampled_state is None and num_tries < 200:
            num_tries = num_tries + 1
            sampled_trajectory_idx = random.choice(range(len(self.positive_examples)))
            sampled_trajectory = self.positive_examples[sampled_trajectory_idx]
            sampled_state = self.get_first_state_in_classifier(sampled_trajectory)
        return sampled_state

    def get_first_state_in_classifier(self, trajectory):
        """ Extract the first state in the trajectory that is inside the initiation classifier. """
        observations = np.array([eg.obs for eg in trajectory])
        predictions = self.pessimistic_predict(observations)
        if predictions.any(): # grab the first positive obs in traj
            return observations[predictions.squeeze()][0]

    def get_states_inside_pessimistic_classifier_region(self):
        if self.pessimistic_classifier is not None:
            observations = self.construct_feature_matrix(self.positive_examples)
            predictions = self.pessimistic_predict(observations).squeeze()
            positive_observations = observations[predictions==1]
            return positive_observations.cpu().numpy()
        return []

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        self.plot_training_predictions(option_name, episode, experiment_name, seed)
        self.plot_testing_predictions(env, replay_buffer, option_name, episode, experiment_name, seed)

    def plot_training_predictions(self, option_name, episode, experiment_name, seed, goal=None):
        """ Plot the predictions on the traininng data. """
        if not self.is_initialized():
            return
        
        x_positive = self.construct_feature_matrix(self.positive_examples)
        x_negative = self.construct_feature_matrix(self.negative_examples)

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
                    c=optimistic_positive_predictions, marker="+", label="positive data")
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1],
                    c=optimistic_negative_predictions, marker="o", label="negative data")
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

        plt.savefig(f"results/{experiment_name}/initiation_set_plots/{option_name}_init_clf_seed{seed}_episode_{episode}.png")
        plt.close()

    def plot_testing_predictions(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        states = env.observations

        x_positions = states[:, 0]
        y_positions = states[:, 1]

        f1 = lambda x: self.batched_optimistic_predict(x).squeeze()
        f2 = lambda x: self.batched_pessimistic_predict(x).squeeze()

        optimistic_predictions = chunked_inference(states, f1)
        pessimistic_predictions = chunked_inference(states, f2)

        plt.figure(figsize=(16, 10))

        plt.subplot(1, 2, 1)
        plt.scatter(x_positions, y_positions, c=optimistic_predictions, s=5)
        plt.colorbar()
        plt.title(f"Optimistic Predictions")

        plt.subplot(1, 2, 2)
        plt.scatter(x_positions, y_positions, c=pessimistic_predictions, s=5)
        plt.colorbar()
        plt.title("Pesssimistic Predictions")

        plt.suptitle(f"{option_name} Test InitiationSet")
        plt.savefig(f"results/{experiment_name}/initiation_set_plots/{option_name}_test_clf_seed_{seed}_episode_{episode}.png")
        plt.close()
