class InitiationClassifier:
    def __init__(self, optimistic_classifier, pessimistic_classifier):
        self.optimistic_classifier = optimistic_classifier
        self.pessimistic_classifier = pessimistic_classifier

    def optimistic_predict(self, state):
        pass

    def pessimistic_predict(self, state):
        pass

    def add_positive_examples(self, trajectory):
        pass

    def add_negative_examples(self, trajectory):
        pass

    def sample(self):
        pass
