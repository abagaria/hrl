import os
import gzip
import random
import pickle
import argparse
import itertools
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from hrl.utils import create_log_dir
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier


class TrainingExample:
    def __init__(self, obs, pos):
        assert isinstance(obs, np.ndarray)
        assert isinstance(pos, (tuple, np.ndarray))

        self.obs = obs
        self.pos = pos



class BaggingInitiationClassifier(InitiationClassifier):
    def __init__(self, buffer_size=100):
        optimistic_classifier = BaggingClassifier()
        pessimistic_classifier = BaggingClassifier()

        self.positive_examples = deque([], maxlen=buffer_size)
        self.negative_examples = deque([], maxlen=buffer_size)
        
        super().__init__(optimistic_classifier, pessimistic_classifier)
    
    def is_initialized(self):
        pass

    def optimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        return self.optimistic_classifier.predict([state])[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (84, 84)
        if self.pessimistic_type == 'one_class_svm':
            return self.pessimistic_classifier.predict([state])[0] == 1 and self.extra_pessimistic_classifier.predict([state])[0] == -1
        else:
            return self.pessimistic_classifier.predict([state])[0] == 1

    def add_positive_examples(self, images, positions):
        assert len(images) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, images, positions):
        assert len(images) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(images, positions)]
        self.negative_examples.append(negative_examples)

    @staticmethod
    def extract_positions(examples):
        examples = itertools.chain.from_iterable(examples)
        positions = [example.pos for example in examples]
        return np.array(positions)

    @staticmethod
    def construct_image_list(examples):
        """
        examples is either self.positive_examples or self.nagetive_examples
        """
        images = [example.obs.squeeze() for trajectory in examples for example in trajectory]
        return images
        # examples = itertools.chain.from_iterable(examples)
        # images = [example.obs._frames[-1] for example in examples]
        # return images

    def fit_initiation_classifier(self):
        if len(self.negative_examples) > 0 and len(self.positive_examples) > 0:
            self.train_two_class_classifier()
        elif len(self.positive_examples) > 0:
            self.train_one_class_classifier()

    def train_one_class_classifier(self):
        # data
        positive_examples = self.construct_image_list(self.positive_examples)
        positive_labels = [1 for _ in positive_examples]
        # pessimistic
        self.pessimistic_classifier.fit(positive_examples, positive_labels, svm_type='one_class_svm', nu=0.1)
        # optimistic
        self.optimistic_classifier.fit(positive_examples, positive_labels, svm_type='one_class_svm', nu=0.01)

    def train_two_class_classifier(self):
        # data
        positive_examples = self.construct_image_list(self.positive_examples)
        positive_labels = [1 for _ in positive_examples]
        nagative_examples = self.construct_image_list(self.negative_examples)
        negative_labels = [0 for _ in nagative_examples]
        X = np.array(positive_examples + nagative_examples)
        Y = np.array(positive_labels + negative_labels)
        # optimistic
        self.optimistic_classifier.fit(X, Y, svm_type='svc')
        # pessimistic
        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]
        negative_training_examples = X[training_predictions == 0]
        if len(positive_training_examples) > 0:
            if self.pessimistic_type == 'one_class_svm':
                self.pessimistic_classifier.fit(positive_training_examples, Y[training_predictions == 1], 
                                                svm_type='one_class_svm', gamma=self.gamma, nu=self.nu)
                self.extra_pessimistic_classifier.fit(negative_training_examples, [1 for _ in negative_training_examples], 
                                                        svm_type='one_class_svm', gamma=self.gamma, nu=self.nu)
            else:
            # two-class-svm pessimistic
                training_positive_predictions = self.optimistic_classifier.predict(positive_examples)
                positive_examples = np.array(positive_examples)
                positive_training_examples = positive_examples[training_positive_predictions == 1]
                negative_training_examples = positive_examples[training_positive_predictions == 0]
                self.pessimistic_classifier.fit(
                    X=np.concatenate((positive_training_examples, negative_training_examples, nagative_examples), axis=0),
                    Y=[1] * len(positive_training_examples) + [0] * (len(negative_training_examples) + len(nagative_examples)),
                    svm_type='svc',
                )

    def sample(self):
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
            frame = state.obs._frames[-1].squeeze()
            if self.pessimistic_predict(frame):
                return state

    def plot_training_predictions(self, plot_path):
        """ Plot the predictions on the traininng data. """
        if not self.is_initialized():
            return
        
        x_positive = self.construct_image_list(self.positive_examples)
        x_negative = self.construct_image_list(self.negative_examples)

        optimistic_positive_predictions = self.optimistic_classifier.predict(x_positive) == 1
        pessimistic_positive_predictions = [self.pessimistic_predict(x) for x in x_positive]

        optimistic_negative_predictions = self.optimistic_classifier.predict(x_negative) == 1
        pessimistic_negative_predictions = [self.pessimistic_predict(x) for x in x_negative]

        positive_positions = self.extract_positions(self.positive_examples)
        negative_positions = self.extract_positions(self.negative_examples)

        plt.subplot(1, 2, 1)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=optimistic_positive_predictions, marker="+", label="positive data")
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1],
                    c=optimistic_negative_predictions, marker="o", label="negative data")
        plt.clim(0, 1)
        plt.xlim(15, 140)
        plt.ylim(140, 260)
        plt.colorbar()
        plt.legend()
        plt.title("Optimistic classifier")

        plt.subplot(1, 2, 2)
        plt.scatter(positive_positions[:, 0], positive_positions[:, 1],
                    c=pessimistic_positive_predictions, marker="+", label="positive data")
        plt.clim(0, 1)
        plt.scatter(negative_positions[:, 0], negative_positions[:, 1], 
                    c=pessimistic_negative_predictions, marker="o", label="negative data")
        plt.clim(0, 1)
        plt.xlim(15, 140)
        plt.ylim(140, 260)
        plt.colorbar()
        plt.legend()
        plt.title("Pessimistic classifier")

        plt.savefig(plot_path)
        plt.close()


def load_trajectories(path, skip=0):
    '''
    Returns a generator for getting states.
    Args:
        path (str): filepath of pkl file containing trajectories
        skip (int): number of trajectories to skip
    Returns:
        (generator): generator to be called for trajectories
    '''
    print(f"[+] Loading trajectories from file '{path}'")
    with gzip.open(path, 'rb') as f:
        for _ in range(skip):
            traj = pickle.load(f)

        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass


def _getIndex(address):
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row * 16 + col
def getByte(ram, address):
    # Return the byte at the specified emulator RAM location
    idx = _getIndex(address)
    return ram[idx]

def get_player_position(ram):
    """
    given the ram state, get the position of the player
    """
    # return the player position at a particular state
    x = int(getByte(ram, 'aa'))
    y = int(getByte(ram, 'ab'))
    return x, y


def get_skull_position(ram):
    """
    given the ram state, get the position of the skull
    """
    x = int(getByte(ram, 'af'))
    return x


def load_training_and_testing_data(
        goal_position=(133, 148), 
        state_type='ram',
        select_training_points=lambda x, y: x >= 90
    ):
    """
    load training and testing data either as RAM states or as images
    """

    assert state_type in ['ram', 'image']

    if os.path.exists("logs/trajectories/training_data.pkl"):
        with open("logs/trajectories/training_data.pkl", "rb") as f:
            positive_training, positive_testing, negative_training, negative_testing = pickle.load(f)
        return positive_training, positive_testing, negative_training, negative_testing  
  
    import random
    traj_generator = load_trajectories("logs/trajectories/monte_rnd_full_trajectories.pkl.gz", skip=0)
    seen_pos = set()
    positive_training = deque(maxlen=80)
    negative_training = deque(maxlen=400)
    positive_testing = deque(maxlen=80)
    negative_testing = deque(maxlen=400)
    for traj in traj_generator:
        if len(positive_training) >= 80 and len(positive_testing) >= 80:
            break
        for state in traj:
            ram, obs = state
            training_state = ram if state_type == 'ram' else obs
            pos = get_player_position(ram)
            if 0 <= np.linalg.norm(np.array(pos) - np.array(goal_position)) < 30:
                # a positive example
                if pos not in seen_pos:
                    if select_training_points(*pos):
                        if random.random() < 0.5:
                            positive_training.append((training_state, pos))
                        else:
                            positive_testing.append((training_state, pos))
                    else:
                        positive_testing.append((training_state, pos))
                    seen_pos.add(pos)
            else:
                # a negative example
                if pos not in seen_pos:
                    if select_training_points(*pos):
                        if random.random() < 0.5:
                            negative_training.append((training_state, pos))
                        else:
                            negative_testing.append((training_state, pos))
                    else:
                        negative_testing.append((training_state, pos))
                    seen_pos.add(pos)
    print(len(positive_training), len(positive_testing), len(negative_training), len(negative_testing))

    with open("logs/trajectories/training_data.pkl", "wb") as f:
        pickle.dump((positive_training, positive_testing, negative_training, negative_testing), f)

    return positive_training, positive_testing, negative_training, negative_testing


def load_data_from_same_position(position=(133, 148), state_type='ram',):
    if os.path.exists("logs/trajectories/same_pos_data.pkl"):
        with open("logs/trajectories/same_pos_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    traj_generator = load_trajectories("logs/trajectories/monte_rnd_full_trajectories.pkl.gz", skip=0)
    data = deque(maxlen=200)
    largest_skull_pos = 0
    for traj in traj_generator:
        if len(data) >= 200 and largest_skull_pos >= 0:
            break
        for state in traj:
            ram, obs = state
            training_state = ram if state_type == 'ram' else obs
            pos = get_player_position(ram)
            skull_pos = get_skull_position(ram)
            largest_skull_pos = max(largest_skull_pos, skull_pos)
            if np.array_equal(pos, position):
                data.append((training_state, (skull_pos, 148)))
    print(f"Loaded {len(data)} data points from position {position}")

    with open("logs/trajectories/same_pos_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return data



def test_bagging_classifier(positive_train, positive_test, negative_train, 
                            negative_test, same_pos_data, plot_dir):
    """
    simply test the bagging classifier
    """
    base_classifier = MLPClassifier(hidden_layer_sizes=(100,))
    clf = BaggingClassifier(base_estimator=base_classifier, n_estimators=4, random_state=0)

    # train
    training_samples = [state for state, _ in positive_train] + [state for state, _ in negative_train]
    clf.fit(training_samples, [1] * len(positive_train) + [0] * len(negative_train))

    # test
    from sklearn.metrics import accuracy_score
    testing_samples = [state for state, _ in positive_test] + [state for state, _ in negative_test]
    prediction = clf.predict(testing_samples)
    accuracy = accuracy_score(prediction, [1] * len(positive_test) + [0] * len(negative_test))
    print(f"Accuracy: {accuracy}")

    # plot the results
    plt.subplot(1, 2, 1)
    positive_positions = [pos for _, pos in positive_train]
    negative_positions = [pos for _, pos in negative_train]
    positive_predictions = clf.predict([state for state, _ in positive_train])
    negative_predictions = clf.predict([state for state, _ in negative_train])
    plt.scatter([x for x, _ in positive_positions], [y for _, y in positive_positions],
                c=positive_predictions, marker="+", label="positive data")
    plt.clim(0, 1)
    plt.scatter([x for x, _ in negative_positions], [y for _, y in negative_positions],
                c=negative_predictions, marker="o", label="negative data")
    plt.clim(0, 1)
    plt.xlim(15, 140)
    plt.ylim(140, 260)
    plt.colorbar()
    plt.legend()
    plt.title("Training")

    plt.subplot(1, 2, 2)
    positive_positions = [pos for _, pos in positive_test]
    negative_positions = [pos for _, pos in negative_test]
    positive_predictions = clf.predict([state for state, _ in positive_test])
    negative_predictions = clf.predict([state for state, _ in negative_test])
    plt.scatter([x for x, _ in positive_positions], [y for _, y in positive_positions],
                c=positive_predictions, marker="+", label="positive data")
    plt.clim(0, 1)
    plt.scatter([x for x, _ in negative_positions], [y for _, y in negative_positions],
                c=negative_predictions, marker="o", label="negative data")
    plt.clim(0, 1)
    plt.xlim(15, 140)
    plt.ylim(140, 260)
    plt.colorbar()
    plt.legend()
    plt.title("Testing")

    plt.savefig(f"{plot_dir}/bagging_classifier.png")
    plt.close()


# def test_classifier(positive_train, positive_test, negative_train, negative_test, 
#                     same_pos_data, pessimistic_type, plot_dir,
#                     gamma, nu, num_kp, num_clusters):
#     svc_classifier = BaggingInitiationClassifier(
#         pessimistic_type=pessimistic_type,
#         gamma=gamma,
#         nu=nu,
#         num_sift_keypoints=num_kp,
#         num_clusters=num_clusters,
#     )
#     svc_classifier.add_positive_examples(*zip(*positive_train))
#     svc_classifier.add_negative_examples(*zip(*negative_train))
#     svc_classifier.fit_initiation_classifier()

#     # training plot
#     plot_path = os.path.join(plot_dir, f"{pessimistic_type}_classifier_train.png")
#     svc_classifier.plot_training_predictions(plot_path)

#     # testing plot
#     svc_classifier.positive_examples = []
#     svc_classifier.negative_examples = []
#     svc_classifier.add_positive_examples(*zip(*positive_test))
#     svc_classifier.add_negative_examples(*zip(*negative_test))
#     plot_path = os.path.join(plot_dir, f"{pessimistic_type}_classifier_test.png")
#     svc_classifier.plot_training_predictions(plot_path)

#     # same position plot
#     svc_classifier.positive_examples = []
#     svc_classifier.negative_examples = [[svc_classifier.negative_examples[0][0]]]
#     assert len(svc_classifier.negative_examples) == 1
#     svc_classifier.add_positive_examples(*zip(*same_pos_data))
#     plot_path = os.path.join(plot_dir, f"{pessimistic_type}_classifier_same_pos.png")
#     svc_classifier.plot_training_predictions(plot_path)


if __name__ == "__main__":
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.05)
    args = parser.parse_args()

    plot_dir = f"plots/bagging"
    create_log_dir("plots")
    create_log_dir(plot_dir)

    positive_train, positive_test, negative_train, negative_test = load_training_and_testing_data()
    assert len(positive_train) > 0
    assert len(negative_train) > 0
    assert len(positive_test) > 0
    assert len(negative_test) > 0
    same_pos_data = load_data_from_same_position()
    assert len(same_pos_data) > 0
    test_bagging_classifier(positive_train, positive_test, negative_train, negative_test, same_pos_data, plot_dir)
