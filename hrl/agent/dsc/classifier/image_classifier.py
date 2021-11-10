import cv2
import ipdb
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from pfrl.wrappers import atari_wrappers
from sklearn.svm import OneClassSVM, SVC
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier


class TrainingExample:
    def __init__(self, obs, pos):
        assert isinstance(obs, atari_wrappers.LazyFrames)
        assert isinstance(pos, (tuple, np.ndarray))

        self.obs = obs
        self.pos = pos


class ImageInitiationClassifier(InitiationClassifier):
    def __init__(self, gamma="auto", size=16):
        self.size = size
        self.gamma = gamma
        optimistic_classifier = None
        pessimistic_classifier = None

        self.positive_examples = deque([], maxlen=100)
        self.negative_examples = deque([], maxlen=100)
        
        super().__init__(optimistic_classifier, pessimistic_classifier)

    def downsample(self, frame):
        frame = frame.squeeze() if frame.shape == (1, 84, 84) else frame
        assert frame.shape == (84, 84), frame.shape
        return cv2.resize(
            frame, (self.size, self.size), interpolation=cv2.INTER_AREA
        ).reshape(-1)

    def batched_downsample(self, frames):
        return np.array([self.downsample(frame) for frame in frames])

    def optimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (84, 84) or state.shape == (1, 84, 84), state.shape
        assert isinstance(self.optimistic_classifier, (OneClassSVM, SVC))
        return self.optimistic_classifier.predict([self.downsample(state)])[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (84, 84) or state.shape == (1, 84, 84), state.shape
        assert isinstance(self.pessimistic_classifier, (OneClassSVM, SVC))
        return self.pessimistic_classifier.predict([self.downsample(state)])[0] == 1

    def add_positive_examples(self, images, positions):
        assert len(images) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, images, positions):
        assert len(images) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.negative_examples.append(negative_examples)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = itertools.chain.from_iterable(examples)
        images = [example.obs._frames[-1] for example in examples]
        return np.array(images).squeeze()  # TODO: Avoid np.array() to preserve memory

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

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma=self.gamma)

        downsampled_positive_feature_matrix = self.batched_downsample(positive_feature_matrix)
        self.pessimistic_classifier.fit(downsampled_positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(kernel="rbf", nu=nu/10., gamma=self.gamma)
        self.optimistic_classifier.fit(downsampled_positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))
        X = self.batched_downsample(X)

        kwargs = {"kernel": "rbf", "gamma": "auto"}
        
        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "auto", "class_weight": "balanced"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma=self.gamma)
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
        assert self.optimistic_classifier is not None
        assert self.pessimistic_classifier is not None

        x_positive = self.construct_feature_matrix(self.positive_examples)
        x_negative = self.construct_feature_matrix(self.negative_examples)

        x_positive = self.batched_downsample(x_positive)
        x_negative = self.batched_downsample(x_negative)

        optimistic_positive_predictions = self.optimistic_classifier.predict(x_positive) == 1
        pessimistic_positive_predictions = self.pessimistic_classifier.predict(x_positive) == 1

        optimistic_negative_predictions = self.optimistic_classifier.predict(x_negative) == 1
        pessimistic_negative_predictions = self.pessimistic_classifier.predict(x_negative) == 1

        positive_positions = self.extract_positions(self.positive_examples)
        negative_positions = self.extract_positions(self.negative_examples)

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

        plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option_name}_init_clf_episode_{episode}.png")
        plt.close()
