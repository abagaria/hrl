import os
import ipdb
import scipy
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
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
        features = self.extract_position(state)
        return self.optimistic_classifier.predict([features])[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(self.pessimistic_classifier, (OneClassSVM, SVC))
        features = self.extract_position(state)
        return self.pessimistic_classifier.predict([features])[0] == 1
    
    def is_initialized(self):
        return self.optimistic_classifier is not None and \
            self.pessimistic_classifier is not None

    def get_false_positive_rate(self):  # TODO: Implement this
        return np.array([0., 0.])

    def add_positive_examples(self, states, positions):
        assert len(states) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(states, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, states, positions):
        assert len(states) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(states, positions)]
        self.negative_examples.append(negative_examples)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        positions = [example.pos for example in examples]
        return np.array(positions)

    @staticmethod
    def extract_position(state):
        assert isinstance(state, np.ndarray), state
        if len(state.shape) == 1:
            return state[:2]
        assert len(state.shape) == 2, state.shape
        return state[:, :2]

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

    def sample(self, epsilon=0.6):
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
            positive_examples = list(itertools.chain.from_iterable(self.positive_examples))
            return [positive_examples[i].pos for i in idx]

        if self.pessimistic_classifier is not None:
            point_array = self.construct_feature_matrix(self.positive_examples)
            point_array_predictions = self.pessimistic_classifier.predict(point_array)
            positive_indices = np.where(point_array_predictions==1)[0]
            positive_observations = get_observations(positive_indices)
            return positive_observations
        return []

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            return xx, yy

        def get_grid_states(mdp):
            ss = []
            x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
            x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
            for x in np.arange(x_low_lim, x_high_lim+1, 1):
                for y in np.arange(y_low_lim, y_high_lim+1, 1):
                    ss.append(np.array((x, y)))
            return ss

        def get_initiation_set_values(mdp):
            values = []
            x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
            x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
            for x in np.arange(x_low_lim, x_high_lim+1, 1):
                for y in np.arange(y_low_lim, y_high_lim+1, 1):
                    pos = np.array((x, y))
                    init = self.optimistic_predict(pos)
                    # if hasattr(mdp.env, 'env'):
                    #     init = init and not mdp.env.env._is_in_collision(pos)
                    values.append(init)
            return values

        def plot_one_class_initiation_classifier():
            colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

            X = self.construct_feature_matrix(self.positive_examples)
            X0, X1 = X[:, 0], X[:, 1]
            xx, yy = make_meshgrid(X0, X1)
            Z1 = self.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z1 = Z1.reshape(xx.shape)

            color = random.choice(colors)
            plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

        def plot_two_class_classifier(mdp, episode, experiment_name, plot_examples=True, seed=0):
            states = get_grid_states(mdp)
            values = get_initiation_set_values(mdp)

            x = np.array([state[0] for state in states])
            y = np.array([state[1] for state in states])
            xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
            xx, yy = np.meshgrid(xi, yi)
            rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
            zz = rbf(xx, yy)
            plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
            plt.colorbar()

            # Plot trajectories
            positive_examples = self.construct_feature_matrix(self.positive_examples)
            negative_examples = self.construct_feature_matrix(self.negative_examples)

            if positive_examples.shape[0] > 0 and plot_examples:
                plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=10)

            if negative_examples.shape[0] > 0 and plot_examples:
                plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=10)

            if self.pessimistic_classifier is not None:
                plot_one_class_initiation_classifier()

            # background_image = imageio.imread("four_room_domain.png")
            # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

            name = option_name if episode is None else option_name + f"_{experiment_name}_{episode}"
            plt.title(f"{option_name} Initiation Set")
            saving_path = os.path.join('results', experiment_name, 'initiation_set_plots', f'{name}_initiation_classifier_{seed}.png')
            plt.savefig(saving_path)
            plt.close()

        plot_two_class_classifier(env, episode, experiment_name, True, seed)
