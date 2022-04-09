class Classifier:
    def __init__(self, term_set, feature_extractor):
        raise NotImplementedError

    def train(self, X, Y):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError
