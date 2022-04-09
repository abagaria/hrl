import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pfrl import nn as pnn
from pfrl.initializers import init_chainer_default

from .classifier import Classifier
from utils.plotting import plot_SVM

def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias

class SmallAtariCNN(nn.Module):
    """Small CNN module proposed for DQN in NeurIPS DL Workshop, 2013.
    See: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self, n_input_channels=4, n_output_channels=256, activation=F.relu, bias=0.1
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 16, 8, stride=4),
                nn.Conv2d(16, 32, 4, stride=2),
            ]
        )
        self.output = nn.Linear(2592, n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.reshape(h.size(0), -1)
        return self.activation(self.output(h_flat))

class ImageCNN(torch.nn.Module):
    def __init__(self, device, n_input_channels=1, n_classes=2):
        super().__init__()

        self.device = device

        # SmallAtariCNN has 256 output channels
        self.feature_extractor = SmallAtariCNN(
            n_input_channels=n_input_channels
        )

        self.model = nn.Sequential(
            self.feature_extractor,
            nn.ReLU(),
            nn.Linear(self.feature_extractor.n_output_channels, n_classes)
        )

        self.to(device)

    def forward(self, images):

        # Add batch and channel dimensions to lone inputs
        if images[0].shape == (84, 84):
            images = images.unsqueeze(0).unsqueeze(0)
            
        return self.model(images)

    @torch.no_grad()
    def predict(self, inputs):
        logits = self.forward(inputs)
        probabilities = F.softmax(logits, dim=1)
        classes = torch.argmax(probabilities, dim=1)
        return classes.float().cpu().numpy().squeeze()

    @torch.no_grad()
    def extract_features(self, image_tensor):
        feature_tensor = self.forward(image_tensor)
        return feature_tensor.view(
            feature_tensor.size(0), -1
        ).cpu().numpy()

class FullCNN(Classifier):
    def __init__(self, device, n_input_channels=1, n_classes=2, batch_size=32):
        '''
        This is an end-to-end classifier that, unlike OneClassSVM and TwoClassSVM, does
        not need a feature extractor.

        Args:
            feature_extractor: obj that extracts features by calling extract_features()
        '''
        self.device = device
        self.is_trained = False
        self.batch_size = batch_size

        self.model = ImageCNN(device, n_input_channels, n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def determine_class_weights(self, Y):
        weights = torch.as_tensor(len(Y) / np.bincount(Y)).float().cuda()
        print(np.bincount(Y))
        print(weights)
        return weights

    def train(self, X, Y):
        '''
        Train classifier using X and Y. 

        Note that there might be more than 2 classes, such as when using the TransductiveExtractor.

        Args:
            states (list(np.array)): list of np.array
            X (list(np.array or MonteRAMState)): list of states
            Y (list(int)): class labels for states in X

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        X, Y = np.array(X), np.array(Y)
        idxs = np.arange(len(Y))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]

        X = torch.as_tensor(X).float().cuda()
        X = X.permute(0, 3, 1, 2)
        class_weights = self.determine_class_weights(Y)
        Y = torch.as_tensor(Y).cuda()

        for batch_start in range(0, len(X), self.batch_size):
            batch_end = batch_start + self.batch_size
            batch_x = X[batch_start:batch_end]
            batch_y = Y[batch_start:batch_end]

            logits = self.model(batch_x)
            loss = F.cross_entropy(logits, batch_y, weight=class_weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.is_trained = True

    @torch.no_grad()
    def predict(self, states):
        '''
        Predict whether state is in the term set.

        Note that when using the TransductiveExtractor, there are 3 classes: 
            1. Positive (1)
            2. Negative, in subgoal traj (0)
            3. Negative, all states outside subgoal traj (2)

        When using the PositiveAugmentExtractor, there is an additional class:
            4. Positive, states outside subgoal traj that is above cos similarity threshold (3)

        Args:
            states (list(np.array or MonteRAMState)): list of states to predict on

        Returns:
            (list(bool): whether states are in the term set
        '''
        states = torch.as_tensor(np.array(states)).float().cuda()
        states = states.permute(0, 3, 1, 2)
        logits = self.model(states)
        probabilities = F.softmax(logits, dim=1)
        predict = torch.argmax(probabilities, dim=1)
        return list(torch.logical_or(predict == 1, predict == 3))

    @torch.no_grad()
    def predict_raw(self, states):
        '''
        Predict class label of states. For the TransductiveExtractor, the labels are 0, 1 and 2. 
        For other label extractors, the labels are 0 and 1.

        Args:
            states (list(np.array or MonteRAMState)): list of states to predict on

        Returns:
            (list(int)): predicted class label of states
        '''
        logits = self.model(states)
        probabilities = F.softmax(logits, dim=1)
        return torch.argmax(probabilities, dim=1)
