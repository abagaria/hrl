import math
import torch
import numpy as np
from hrl.agent.dsc.classifier.conv_classifier import ConvClassifier


class EnsembleClassifier(object):
    def __init__(self, ensemble_size, device):
        """ An ensemble of binary convolutional classifiers. """ 

        self.device = device
        self.is_trained = False
        self.ensemble_size = ensemble_size
        self.members = [ConvClassifier(device) for _ in range(ensemble_size)]

    @torch.no_grad()
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        predicted_classes = np.vstack(
            [member.predict(X).cpu().numpy() for member in self.members]
        )
        return predicted_classes

    def should_train(self, y):
        assert len(self.members) > 0
        return all([member.should_train(y) for member in self.members])

    def fit(self, X, y):
        if self.should_train(y):
            for i in range(self.ensemble_size):
                n_samples = math.ceil(len(y) / 2)
                rows = np.random.randint(X.shape[0], size=n_samples)
                X_member, y_member = X[rows, :], y[rows]
                self.members[i].fit(X_member, y_member)
            
            self.is_trained = True
    
    