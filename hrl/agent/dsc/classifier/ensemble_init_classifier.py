import ipdb
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt


from collections import deque
from .ensemble_classifier import EnsembleClassifier
from .init_classifier import InitiationClassifier
from hrl.agent.dsc.datastructures import TrainingExample


class EnsembleInitiationClassifier(InitiationClassifier):
    def __init__(self, ensemble_size, device, maxlen=100):
        self.device = device
        self.ensemble_size = ensemble_size

        self.classifier = EnsembleClassifier(ensemble_size, device)
        optimistic_classifier = self.classifier
        pessimistic_classifier = self.classifier

        # TODO: How to initialize this?
        self.variance_threshold = 0.2

        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)

        super().__init__(optimistic_classifier, pessimistic_classifier)

    def is_initialized(self):
        return self.classifier.is_trained

    @torch.no_grad()
    def ensemble_classify(self, state):
        """ Extract the ensemble decision and disagreement for the given state input. """
        assert isinstance(state, (np.ndarray, torch.Tensor))
        predicted_classes = self.classifier.predict(state).squeeze()
        assert predicted_classes.shape == (self.ensemble_size,), predicted_classes.shape
        predicted_label = np.median(predicted_classes)
        prediction_variance = np.std(predicted_classes) ** 2
        assert predicted_label in (0., 0.5, 1.), predicted_label
        return predicted_label, prediction_variance

    def optimistic_predict(self, state):
        """ State is inside optimistic classifier if
            (a) Aleatoric uncertainty is high OR
            (b) Epistemic uncertainty is high OR
            (c) Majority of classifiers classify positive
        """
        assert isinstance(state, np.ndarray)
        assert isinstance(self.classifier, EnsembleClassifier)
        assert self.classifier.is_trained
        label, variance = self.ensemble_classify(state)
        return (label > 0) or (variance > self.variance_threshold)

    def pessimistic_predict(self, state):
        """ State is inside pessimistic classifier if
            (a) Aleatoric uncertainty is low AND
            (b) Epistemic uncertainty is low AND
            (c) Majority of classifiers classify positive
        """
        assert isinstance(state, np.ndarray)
        assert isinstance(self.classifier, EnsembleClassifier)
        assert self.classifier.is_trained
        label, variance = self.ensemble_classify(state)
        return (label == 1) and (variance < self.variance_threshold)

    def get_false_positive_rate(self):
        """ Fraction of the negative data that is classified as positive. """ 

        negative_examples = self.construct_feature_matrix(self.negative_examples)
        
        if len(negative_examples) > 0:
            optimistic_preds = [self.optimistic_predict(x) for x in negative_examples]
            pessimistic_preds = [self.pessimistic_predict(x) for x in negative_examples]

            return np.array(
                np.mean(optimistic_preds), np.mean(pessimistic_preds)
            )

        return 1.0

    def add_positive_examples(self, images, infos):
        assert len(images) == len(infos)
        
        positive_examples = [TrainingExample(img, info) for img, info in zip(images, infos)]
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
        if len(self.positive_examples) > 0 and \
           len(self.negative_examples) > 0:
            self.train_two_class_classifier()
            self.determine_variance_threshold()
        else:
            print("No data to fit initiation classifier")

    def train_two_class_classifier(self):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = torch.ones((positive_feature_matrix.shape[0],), device=self.device)
        negative_labels = torch.zeros((negative_feature_matrix.shape[0],), device=self.device)

        X = torch.cat((positive_feature_matrix, negative_feature_matrix))
        Y = torch.cat((positive_labels, negative_labels))

        if self.classifier.should_train(Y):
            self.classifier = EnsembleClassifier(self.ensemble_size, self.device)
            self.classifier.fit(X, Y)

    def determine_variance_threshold(self):
        if self.classifier.is_trained:
            X = self.construct_feature_matrix(
                list(self.positive_examples) + list(self.negative_examples)
            )
            predicted_variances = [self.ensemble_classify(x)[1] for x in X] 
            variance_threshold = np.quantile(predicted_variances, 0.9)
            if variance_threshold > self.variance_threshold:
                self.variance_threshold = variance_threshold
                print(f"Increased ensemble classifier variance threshold to {variance_threshold}")

    def plot_training_predictions(self, option_name, episode, experiment_name, seed):
        """ Plot the predictions on the traininng data. """
        if not self.is_initialized():
            return
        
        x_positive = self.construct_feature_matrix(self.positive_examples).cpu().numpy()
        x_negative = self.construct_feature_matrix(self.negative_examples).cpu().numpy()

        optimistic_positive_predictions = np.array([self.optimistic_predict(x) for x in x_positive])
        pessimistic_positive_predictions = np.array([self.pessimistic_predict(x) for x in x_positive])

        optimistic_negative_predictions = np.array([self.optimistic_predict(x) for x in x_negative])
        pessimistic_negative_predictions = np.array([self.pessimistic_predict(x) for x in x_negative])

        positive_variance_predictions = np.array([self.ensemble_classify(x)[1] for x in x_positive])
        negative_variance_predictions = np.array([self.ensemble_classify(x)[1] for x in x_negative])
        max_variance = max(positive_variance_predictions.max(), negative_variance_predictions.max())

        positive_positions = self.extract_positions(self.positive_examples)
        negative_positions = self.extract_positions(self.negative_examples)

        plt.figure(figsize=(16, 10))

        plt.subplot(1, 3, 1)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=optimistic_positive_predictions, marker="+", label="positive data")
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1],
                    c=optimistic_negative_predictions, marker="o", label="negative data")
        plt.clim(0, 1)
        plt.colorbar()
        plt.legend()
        plt.title("Optimistic classifier")

        plt.subplot(1, 3, 2)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=pessimistic_positive_predictions, marker="+", label="positive data")
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1], 
                    c=pessimistic_negative_predictions, marker="o", label="negative data")
        plt.clim(0, 1)
        plt.colorbar()
        plt.legend()
        plt.title("Pessimistic classifier")

        plt.subplot(1, 3, 3)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=positive_variance_predictions, marker="+", label="positive data")
        plt.clim(0, max_variance)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1], 
                    c=negative_variance_predictions, marker="o", label="negative data")
        plt.clim(0, max_variance)
        plt.colorbar()
        plt.legend()
        plt.title("Variance predictions")

        plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option_name}_init_clf_episode_{episode}.png")
        plt.close()

    def plot_variance_predictions(self, examples, option_name, episode, experiment_name, seed):
        if not self.is_initialized():
            return
        
        features = self.construct_feature_matrix(examples).cpu().numpy()
        variance_predictions = np.array([self.ensemble_classify(x)[1] for x in features])
        positions = self.extract_positions(examples)

        plt.scatter(positions[:, 0], positions[:, 1], c=variance_predictions)
        plt.colorbar()
        plt.title(f"Variance (thresh={np.round(self.variance_threshold, decimals=3)})")

        plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option_name}_init_clf_variance_episode_{episode}.png")
        plt.close()
