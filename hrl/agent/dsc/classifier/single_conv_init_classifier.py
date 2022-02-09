import ipdb
import torch
import itertools
import numpy as np
from collections import deque
from sklearn.svm import OneClassSVM
from hrl.agent.dsc.datastructures import TrainingExample
from hrl.agent.dsc.classifier.conv_classifier import ConvClassifier
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier


class SingleConvInitiationClassifier(InitiationClassifier):
    def __init__(self, device, n_input_channels=1, maxlen=100):
        self.device = device
        self.n_input_channels = n_input_channels
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)
        optimistic_classifier = ConvClassifier(device, n_input_channels)
        pessimistic_classifier = OneClassSVM()
        
        super().__init__(optimistic_classifier, pessimistic_classifier)

    def is_initialized(self):
        return self.optimistic_classifier.is_trained and \
               self.pessimistic_classifier.fit_status == 0

    def optimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert isinstance(self.optimistic_classifier, ConvClassifier)
        features = torch.as_tensor(state).float().to(self.device)
        return self.optimistic_classifier.predict(features) == 1

    def pessimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert isinstance(self.pessimistic_classifier, OneClassSVM)
        state_tensor = torch.as_tensor(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        features = self.optimistic_classifier.model.extract_features(state_tensor)
        return self.pessimistic_classifier.predict([features]) == 1
    
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
        if observations.shape[1:] != (1, 84, 84):
            ipdb.set_trace()
        obs_tensor = torch.as_tensor(observations).float().to(self.device)
        features = self.optimistic_classifier.model.extract_features(obs_tensor)
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
    
    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        self.optimistic_classifier = ConvClassifier(self.device, self.n_input_channels)
        self.optimistic_classifier.fit(X, Y)
        print(f"Fitting optimistic clasifier on input shape {X.shape}")

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            print(f"Fitting pessimistic clasifier on input shape {positive_training_examples.shape}")
            self.pessimistic_classifier.fit(
                positive_training_examples.cpu().numpy()
            )
