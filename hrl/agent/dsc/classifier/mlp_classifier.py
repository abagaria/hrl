import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from hrl.agent.dsc.classifier.utils import ObsClassifierMLP


class BinaryMLPClassifier:
    """" Generic binary neural network classifier. """
    def __init__(self,
                obs_dim,
                device,
                threshold=0.5,
                batch_size=128):
        
        self.device = device
        self.is_trained = False
        self.threshold = threshold
        self.batch_size = batch_size

        self.model = ObsClassifierMLP(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Debug variables
        self.losses = []

    @torch.no_grad()
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).to(self.device).float()
        logits = self.model(X)
        probabilities = torch.sigmoid(logits)
        return probabilities
        
    @torch.no_grad()
    def predict(self, X, threshold=None):
        logits = self.model(X)
        probabilities = torch.sigmoid(logits)
        threshold = self.threshold if threshold is None else threshold
        return probabilities > threshold

    def determine_pos_weight(self, y):
        n_negatives = len(y[y != 1])
        n_positives = len(y[y == 1])
        if n_positives > 0:
            pos_weight = (1. * n_negatives) / n_positives
            return torch.as_tensor(pos_weight).float()

    def should_train(self, y):
        enough_data = len(y) > self.batch_size
        has_positives = len(y[y == 1]) > 0
        has_negatives = len(y[y != 1]) > 0
        return enough_data and has_positives and has_negatives

    def fit(self, X, y, W=None, n_epochs=5):
        dataset = ClassifierDataset(X, y, W)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.should_train(y):
            losses = []

            for _ in range(n_epochs):
                epoch_loss = self._train(dataloader)                
                losses.append(epoch_loss)
            
            self.is_trained = True

            mean_loss = np.mean(losses)
            print(mean_loss)
            self.losses.append(mean_loss)

    def _train(self, loader):
        """ Single epoch of training. """
        batch_losses = []

        for sample in loader:
            observations, labels, weights = self._extract_sample(sample)

            pos_weight = self.determine_pos_weight(labels)

            if not pos_weight:
                continue

            logits = self.model(observations)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(),
                                                      labels,
                                                      pos_weight=pos_weight,
                                                      weight=weights) 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())
        
        return np.mean(batch_losses)
    
    @staticmethod
    def _extract_sample(sample):
        weights = None

        if len(sample) == 3:
            observations, labels, weights = sample
        else:
            observations, labels = sample

        return observations, labels, weights


class ClassifierDataset(Dataset):
    def __init__(self, states, labels, weights=None):
        self.states = states
        self.labels = labels
        self.weights = weights
        super().__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.weights:
            return self.states[i], self.labels[i], self.weights[i]

        return self.states[i], self.labels[i]
