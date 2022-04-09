import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchsummary import summary

from utils.monte_preprocessing import parse_ram
from .feature_extractor import FeatureExtractor

class CNN(FeatureExtractor):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        '''
        # 1 Conv Layer
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=10,
                stride=5),
        )
        '''

        # 2 Conv Layers
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
        )
        self.model.cuda()
        summary(self.model, (1, 84, 84))

        for p in self.model:
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, states):
        '''
        Extract RND features from raw images

        Args:
            states (list(np.array)): list of np.array

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        output = []
        for i in range(0, len(states), self.batch_size):
            batch_states = states[i:i+self.batch_size]
            batch_states = np.stack(batch_states, axis=0)
            batch_states = batch_states.transpose(0, 3, 1, 2)

            batch_output = self.model(torch.from_numpy(batch_states).float().to("cuda:0"))
            batch_output = batch_output.cpu().numpy()
            batch_output = [np.squeeze(x) for x in np.split(batch_output, np.size(batch_output, 0))]

            output.extend(batch_output)
        return output
