import ipdb
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from hrl.agent.dsc.datastructures import TrainingExample
from hrl.agent.dsc.classifier.conv_classifier import ConvClassifier
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier


class DoubleConvInitiationClassifier(InitiationClassifier):
    """ Initiation classifier where the optimistic and the pessimistic clfs are their own CNNs."""
    def __init__(self,
                 device, 
                 optimistic_threshold=0.50,
                 pessimistic_threshold=0.75,
                 n_input_channels=1,
                 pessimistic_relabel=False,
                 maxlen=1000):
        self.device = device
        self.n_input_channels = n_input_channels
        self.pessimistic_relabel = pessimistic_relabel
        self.optimistic_threshold = optimistic_threshold
        self.pessimistic_threshold = pessimistic_threshold
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)
        optimistic_classifier = ConvClassifier(device, optimistic_threshold, n_input_channels)
        pessimistic_classifier = ConvClassifier(device, pessimistic_threshold, n_input_channels)
        
        super().__init__(optimistic_classifier, pessimistic_classifier)

    def is_initialized(self):
        return self.optimistic_classifier.is_trained and \
               self.pessimistic_classifier.is_trained

    @torch.no_grad()
    def batched_optimistic_predict(self, X):
        assert isinstance(self.optimistic_classifier, ConvClassifier)
        assert self.optimistic_classifier.is_trained
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).float().to(self.device)
        optimistic_predictions = self.optimistic_classifier.predict(X)
        return optimistic_predictions.cpu().numpy()

    @torch.no_grad()
    def batched_pessimistic_predict(self, X):
        assert isinstance(self.pessimistic_classifier, ConvClassifier)
        assert self.pessimistic_classifier.is_trained
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).float().to(self.device)
        pessimistic_predictions = self.pessimistic_classifier.predict(X)
        return pessimistic_predictions.cpu().numpy()

    def optimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert isinstance(self.optimistic_classifier, ConvClassifier)
        assert self.optimistic_classifier.is_trained
        features = torch.as_tensor(state).float().to(self.device)
        label = self.optimistic_classifier.predict(features) == 1
        return label.cpu().numpy()

    def pessimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert isinstance(self.pessimistic_classifier, ConvClassifier)
        assert self.pessimistic_classifier.is_trained
        features = torch.as_tensor(state).float().to(self.device)
        label = self.pessimistic_classifier.predict(features) == 1
        return label.cpu().numpy()

    def get_false_positive_rate(self):
        """ Fraction of the negative data that is classified as positive. """ 

        negative_examples = self.construct_feature_matrix(self.negative_examples)
        
        if len(negative_examples) > 0:
            optimistic_pred = self.optimistic_classifier.predict(negative_examples).cpu().numpy()
            pessimistic_pred = self.pessimistic_classifier.predict(negative_examples).cpu().numpy()

            return np.array(
                optimistic_pred.mean(), pessimistic_pred.mean()
            )

        return np.array([1., 1.])
    
    def add_positive_examples(self, observations, infos):
        assert len(observations) == len(infos)

        positive_examples = [TrainingExample(img, info) for img, info in zip(observations, infos)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, observations, infos):
        assert len(observations) == len(infos)

        negative_examples = [TrainingExample(img, info) for img, info in zip(observations, infos)]
        self.negative_examples.append(negative_examples)

    def construct_feature_matrix(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
        observations = np.array([example.obs._frames[-1] for example in examples])
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

        if self.optimistic_classifier.should_train(Y):
            self.optimistic_classifier = ConvClassifier(device=self.device,
                                                        threshold=self.optimistic_threshold,
                                                        n_input_channels=self.n_input_channels)
            self.optimistic_classifier.fit(X, Y)
            
            # When pessimistic_relabel is True, we train the pessimistic classifier on the 
            # training predictions of the optimistic classifier. When false, we use the default labels.
            if self.pessimistic_relabel:
                training_predictions = self.optimistic_classifier.predict(X)
                positive_training_examples = X[training_predictions == 1]
                negative_training_examples = X[training_predictions != 1]
                
                X_pessimistic = torch.cat(
                    (positive_training_examples,
                    negative_training_examples), 
                    dim=0
                ).unsqueeze(1)
                Y_pessimistic = torch.cat(
                    (torch.ones(positive_training_examples.shape[0],),
                    torch.zeros(negative_training_examples.shape[0],)),
                    dim=0
                )
            else:
                X_pessimistic = X
                Y_pessimistic = Y

            if self.pessimistic_classifier.should_train(Y_pessimistic):
                self.pessimistic_classifier = ConvClassifier(device=self.device,
                                                            threshold=self.pessimistic_threshold,
                                                            n_input_channels=self.n_input_channels)

                self.pessimistic_classifier.fit(X_pessimistic, Y_pessimistic)

    def plot_training_predictions(self, option_name, episode, experiment_name, seed, goal=None):
        """ Plot the predictions on the traininng data. """
        if not self.is_initialized():
            return
        
        x_positive = self.construct_feature_matrix(self.positive_examples)
        x_negative = self.construct_feature_matrix(self.negative_examples)

        optimistic_positive_predictions = self.optimistic_classifier.predict(x_positive) == 1
        pessimistic_positive_predictions = self.pessimistic_classifier.predict(x_positive) == 1

        optimistic_negative_predictions = self.optimistic_classifier.predict(x_negative) == 1
        pessimistic_negative_predictions = self.pessimistic_classifier.predict(x_negative) == 1

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

        plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option_name}_init_clf_episode_{episode}.png")
        plt.close()
