import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
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
    def predict(self, X):
        logits = self.model(X)
        probabilities = F.sigmoid(logits)
        return probabilities > self.threshold

    def determine_pos_weight(self, y):
        n_negatives = len(y[y != 1])
        n_positives = len(y[y == 1])
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

    def fit(self, X, y):
        if self.should_train(y):
            losses = []
            pos_weight = self.determine_pos_weight(y)
            n_gradient_steps = len(X) // self.batch_size

            for _ in tqdm(range(n_gradient_steps), desc="Training CNN classifier"):
                sampled_inputs, sampled_labels = self.sample(X, y)

                logits = self.model(sampled_inputs)
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(),
                                                          sampled_labels,
                                                          pos_weight=pos_weight)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            
            self.is_trained = True

            mean_loss = np.mean(losses)
            self.losses.append(mean_loss)
            print("Mean loss: ", mean_loss)
