import torch
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from hrl.agent.dsc.classifier.ensemble_models import ImageCNN


class ConvClassifier:
    """" Generic binary convolutional classifier. """
    def __init__(self,
                device,
                threshold=0.5,
                n_input_channels=1,
                batch_size=32):
        
        self.device = device
        self.is_trained = False
        self.threshold = threshold
        self.batch_size = batch_size

        self.model = ImageCNN(device, n_input_channels)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Debug variables
        self.losses = []

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

    def sample(self, X, y):
        idx = random.sample(range(len(X)), k=self.batch_size)
        input_samples = X[idx, :]
        label_samples = y[idx]
        return torch.as_tensor(input_samples).to(self.device),\
               torch.as_tensor(label_samples).to(self.device)

    def fit(self, X, y, n_epochs=3):
        dataset = ClassifierDataset(X, y)
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

        for sampled_inputs, sampled_labels in loader:
            pos_weight = self.determine_pos_weight(sampled_labels)

            if not pos_weight:
                continue

            logits = self.model(sampled_inputs)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(),
                                                      sampled_labels,
                                                      pos_weight=pos_weight) 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())
        
        return np.mean(batch_losses)


class ClassifierDataset(Dataset):
    def __init__(self, states, labels):
        self.states = states
        self.labels = labels
        super().__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.states[i], self.labels[i]
