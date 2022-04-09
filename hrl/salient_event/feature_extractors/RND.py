import numpy as np
import torch

from rnd.model import RNDModel

from utils.monte_preprocessing import parse_ram
from .feature_extractor import FeatureExtractor

class RND(FeatureExtractor):
    def __init__(self, predictor_path, batch_size=32):
        self.batch_size = batch_size
        self.rnd = RNDModel((4, 84, 84), 18)
        self.rnd.predictor.load_state_dict(torch.load(predictor_path))
        self.rnd.predictor.cuda()

        def get_features(model, input, output):
            self.features = output.detach()

        self.rnd.predictor[4].register_forward_hook(get_features)

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
            self.rnd.predictor(torch.from_numpy(batch_states).float().to("cuda:0"))

            batch_output = self.features.cpu().numpy()
            batch_output = [np.squeeze(x) for x in np.split(batch_output, np.size(batch_output, 0))]
            output.extend(batch_output)
        return output
