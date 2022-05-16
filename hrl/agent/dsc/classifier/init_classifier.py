class InitiationClassifier:
    def __init__(self, optimistic_classifier, pessimistic_classifier):
        self.optimistic_classifier = optimistic_classifier
        self.pessimistic_classifier = pessimistic_classifier

    def optimistic_predict(self, state):
        pass

    def pessimistic_predict(self, state):
        pass

    def optimistic_predict_proba(self, state):
        pass

    def pessimistic_predict_proba(self, state):
        pass

    def is_initialized(self):
        pass

    @staticmethod
    def construct_feature_matrix(examples):
        pass

    def add_positive_examples(self, states, positions):
        pass

    def add_negative_examples(self, states, positions):
        pass

    def fit_initiation_classifier(self):
        pass

    def sample(self):
        pass

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        pass
