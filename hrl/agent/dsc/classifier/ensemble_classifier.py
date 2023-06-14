import torch
import random
import itertools
import numpy as np
from collections import deque
from hrl.agent.dsc.classifier.conv_classifier import ConvClassifier
from hrl.agent.dsc.datastructures import TrainingExample


class EnsembleClassifier(object):
    def __init__(self, ensemble_size, device):
        """ An ensemble of binary convolutional classifiers. """ 

        self.device = device
        self.is_trained = False
        self.ensemble_size = ensemble_size
        
        self.members = [ConvClassifier(device) for _ in range(ensemble_size)]
        self.positive_examples = [deque([], maxlen=100) for _ in range(ensemble_size)]
        self.negative_examples = [deque([], maxlen=100) for _ in range(ensemble_size)]

    @torch.no_grad()
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        predicted_classes = np.vstack(
            [member.predict(X).cpu().numpy() for member in self.members]
        )
        return predicted_classes

    def should_train(self):
        assert len(self.members) > 0
        for i, member in enumerate(self.members):
            Y = self.prepare_training_data(i)[1]
            if not member.should_train(Y):
                return False
        return True

    def prepare_training_data(self, member_idx):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples[member_idx])
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples[member_idx])
        positive_labels = torch.ones((positive_feature_matrix.shape[0],), device=self.device)
        negative_labels = torch.zeros((negative_feature_matrix.shape[0],), device=self.device)

        X = torch.cat((positive_feature_matrix, negative_feature_matrix))
        Y = torch.cat((positive_labels, negative_labels))

        return X, Y

    def construct_feature_matrix(self, examples):
        examples = list(itertools.chain.from_iterable(examples))
        observations = np.array([example.obs._frames[-1] for example in examples])
        obs_tensor = torch.as_tensor(observations).float().to(self.device)
        return obs_tensor

    def fit(self):
        if self.should_train():
            for i in range(self.ensemble_size):
                X_member, Y_member = self.prepare_training_data(i)
                self.members[i].fit(
                    X_member, Y_member
                )
            
            self.is_trained = True

    def _subsample_trajectory(self, egs):
        subsampled_trajectory = []

        for eg in egs:
            assert isinstance(eg, TrainingExample), type(eg)
            if random.random() > 0.5:
                subsampled_trajectory.append(eg)
        
        return subsampled_trajectory

    def add_positive_trajectory(self, positive_egs):
        for i in range(self.ensemble_size):
            subsampled_trajectory = self._subsample_trajectory(positive_egs)
            self.positive_examples[i].append(subsampled_trajectory)
    
    def add_negative_trajectory(self, negative_egs):
        for i in range(self.ensemble_size):
            subsampled_trajectory = self._subsample_trajectory(negative_egs)
            self.negative_examples[i].append(subsampled_trajectory)
