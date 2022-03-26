import ipdb
import random
import itertools
import numpy as np
from collections import deque
from pfrl.wrappers import atari_wrappers
from sklearn.svm import OneClassSVM, SVC
from .init_classifier import InitiationClassifier
from hrl.agent.dsc.datastructures import TrainingExample



class PositionInitiationClassifier(InitiationClassifier):
    def __init__(self, maxlen=100):
        optimistic_classifier = None
        pessimistic_classifier = None
        self.positive_examples = deque([], maxlen=maxlen)
        self.negative_examples = deque([], maxlen=maxlen)
        super().__init__(optimistic_classifier, pessimistic_classifier)

    def optimistic_predict(self, state):
        assert isinstance(self.optimistic_classifier, (OneClassSVM, SVC))
        return self.optimistic_classifier.predict([state])[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(self.pessimistic_classifier, (OneClassSVM, SVC))
        return self.pessimistic_classifier.predict([state])[0] == 1
    
    def is_initialized(self):
        return self.optimistic_classifier is not None and \
            self.pessimistic_classifier is not None

    def get_false_positive_rate(self):  # TODO: Implement this
        return np.array([0., 0.])

    def add_positive_examples(self, images, positions):
        assert len(images) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, images, positions):
        assert len(images) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.negative_examples.append(negative_examples)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        positions = [example.pos for example in examples]
        return np.array(positions)

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_svm()

    def train_one_class_svm(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
        self.pessimistic_classifier.fit(positive_feature_matrix)

        self.optimistic_classifier = OneClassSVM(kernel="rbf", nu=nu/10., gamma="scale")
        self.optimistic_classifier.fit(positive_feature_matrix)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            self.pessimistic_classifier.fit(positive_training_examples)

    def sample(self, epsilon=2.):
        """ Epsilon-safe sampling from the pessimistic initiation classifier. """
        
        def compile_states(s):
            """ Get positions that lie in an epsilon-box around s.pos. """
            pos0 = s.pos
            pos1 = np.copy(pos0)
            pos2 = np.copy(pos0)
            pos3 = np.copy(pos0)
            pos4 = np.copy(pos0)
            pos1[0] -= epsilon
            pos2[0] += epsilon
            pos3[1] -= epsilon
            pos4[1] += epsilon
            return pos0, pos1, pos2, pos3, pos4

        idxs = list(range(len(self.positive_examples)))
        random.shuffle(idxs)

        for idx in idxs:
            sampled_trajectory = self.positive_examples[idx]
            
            positions = []
            for s in sampled_trajectory:
                positions.extend(compile_states(s))

            position_matrix = np.vstack(positions)
            predictions = self.pessimistic_classifier.predict(position_matrix) == 1
            predictions = np.reshape(predictions, (-1, 5))
            valid = np.all(predictions, axis=1)
            indices = np.argwhere(valid == True)
            
            if len(indices) > 0:
                return sampled_trajectory[indices[0][0]]

        return self.sample_from_initiation_region()

    def sample_from_initiation_region(self):
        """ Sample from the pessimistic initiation classifier. """
        num_tries = 0
        sampled_state = None
        while sampled_state is None and num_tries < 200:
            num_tries = num_tries + 1
            sampled_trajectory_idx = random.choice(range(len(self.positive_examples)))
            sampled_trajectory = self.positive_examples[sampled_trajectory_idx]
            sampled_state = self.get_first_state_in_classifier(sampled_trajectory)
        return sampled_state

    def get_first_state_in_classifier(self, trajectory):
        """ Extract the first state in the trajectory that is inside the initiation classifier. """

        for state in trajectory:
            assert isinstance(state, TrainingExample)
            if self.pessimistic_predict(state.pos):
                return state

    def get_states_inside_pessimistic_classifier_region(self):
        def get_observations(idx):
            positive_examples = itertools.chain.from_iterable(self.positive_examples)
            return [positive_examples[i].obs for i in idx]

        if self.pessimistic_classifier is not None:
            point_array = self.construct_feature_matrix(self.positive_examples)
            point_array_predictions = self.pessimistic_classifier.predict(point_array)
            ipdb.set_trace()  # TODO: Test if np.where works here
            positive_indices = np.where(point_array_predictions==1)
            positive_observations = get_observations(positive_indices)
            return positive_observations
        return []
