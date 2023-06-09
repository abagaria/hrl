#import ipdb
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from collections import deque
from sklearn.svm import OneClassSVM, SVC
from hrl.agent.dsc.datastructures import TrainingExample
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FixedConvFeatureExtractor(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
        )
        self.model.to(self.device)

        for p in self.model:
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, obs):
        return self.model(obs)

    @torch.no_grad()
    def __call__(self, images):
        assert isinstance(images, np.ndarray)
        image_tensor = torch.as_tensor(images).float().to(self.device)
        feature_tensor = self.forward(image_tensor)
        return feature_tensor.view(
            feature_tensor.size(0), -1
        ).cpu().numpy()


class FixedConvInitiationClassifier(InitiationClassifier):
    """ Use a randomly initialized conv network as a feature extractor. 
    Combine with our classic 2-class and 1-class SVMs for final classification. """

    def __init__(self, device, gamma='scale', nu=0.1, maxlen=100):
        self.nu = nu
        self.gamma = gamma
        optimistic_classifier = None
        pessimistic_classifier = None
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)

        self.feature_extractor = FixedConvFeatureExtractor(device)

        super().__init__(optimistic_classifier, pessimistic_classifier)

    def expand_single_state(self, state):
        return state[None, None, ...]

    def optimistic_predict(self, state):
        assert isinstance(self.optimistic_classifier, (OneClassSVM, SVC))
        features = self.feature_extractor(
            self.expand_single_state(state)
        )
        return self.optimistic_classifier.predict(features)[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(self.pessimistic_classifier, (OneClassSVM, SVC))
        features = self.feature_extractor(
            self.expand_single_state(state)
        )
        return self.pessimistic_classifier.predict(features)[0] == 1
    
    def is_initialized(self):
        return self.optimistic_classifier is not None and \
            self.pessimistic_classifier is not None

    def add_positive_examples(self, images, positions):
        assert len(images) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, images, positions):
        assert len(images) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.negative_examples.append(negative_examples)

    def construct_feature_matrix(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
        observations = np.array([example.obs._frames[-1] for example in examples])
        if observations.shape[1:] != (1, 84, 84):
            ipdb.set_trace()
        features = self.feature_extractor(observations)
        return features

    @staticmethod
    def extract_positions(examples):
        examples = itertools.chain.from_iterable(examples)
        positions = [example.pos for example in examples]
        return np.array(positions)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def train_one_class_svm(self):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(kernel="rbf", nu=self.nu/10., gamma=self.gamma)
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)
        print(f"Fitting optimistic clasifier on input shape {X.shape}")

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
            print(f"Fitting pessimistic clasifier on input shape {positive_training_examples.shape}")
            self.pessimistic_classifier.fit(positive_training_examples)

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
        for state in trajectory:
            assert isinstance(state, TrainingExample)
            frame = state.obs._frames[-1].squeeze()
            if self.pessimistic_predict(frame):
                return state

    def plot_training_predictions(self, option_name, episode, experiment_name, seed):
        """ Plot the predictions on the traininng data. """
        if not self.is_initialized():
            return

        if len(self.positive_examples) > 0:        
            x_positive = self.construct_feature_matrix(self.positive_examples)
            optimistic_positive_predictions = self.optimistic_classifier.predict(x_positive) == 1
            pessimistic_positive_predictions = self.pessimistic_classifier.predict(x_positive) == 1
            positive_positions = self.extract_positions(self.positive_examples)

        if len(self.negative_examples) > 0:
            x_negative = self.construct_feature_matrix(self.negative_examples)
            optimistic_negative_predictions = self.optimistic_classifier.predict(x_negative) == 1
            pessimistic_negative_predictions = self.pessimistic_classifier.predict(x_negative) == 1
            negative_positions = self.extract_positions(self.negative_examples)

        plt.subplot(1, 2, 1)

        if len(self.positive_examples) > 0:
            plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                        c=optimistic_positive_predictions, marker="+", label="positive data")
            plt.clim(0, 1)
        
        if len(self.negative_examples) > 0:
            plt.scatter(negative_positions[:, 0], negative_positions[:, 1],
                        c=optimistic_negative_predictions, marker="o", label="negative data")
            plt.clim(0, 1)
        
        plt.colorbar()
        plt.legend()
        plt.title("Optimistic classifier")

        plt.subplot(1, 2, 2)

        if len(self.positive_examples) > 0:
            plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                        c=pessimistic_positive_predictions, marker="+", label="positive data")
            plt.clim(0, 1)

        if len(self.negative_examples) > 0:
            plt.scatter(negative_positions[:, 0], negative_positions[:, 1], 
                        c=pessimistic_negative_predictions, marker="o", label="negative data")
            plt.clim(0, 1)

        plt.colorbar()
        plt.legend()
        plt.title("Pessimistic classifier")

        plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option_name}_init_clf_episode_{episode}.png")
        plt.close()

    
