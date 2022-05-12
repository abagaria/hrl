import numpy as np


class TrainingExample:
    def __init__(self, obs, info):

        self.obs = obs
        self.info = self._construct_info(info)

    def _construct_info(self, x):
        return x if isinstance(x, dict) else dict(player_x=x[0], player_y=x[1])

    @property
    def pos(self):
        pos = self.info['player_x'], self.info['player_y']
        return np.array(pos)


class StepThresholder:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def __call__(self, steps):
        return steps < self.threshold


class ValueThresholder:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, value):
        return value > self.threshold
