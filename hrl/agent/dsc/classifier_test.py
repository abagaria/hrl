import os
import cv2
import gzip
import random
import pickle
import argparse
import itertools
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, svm

from hrl.utils import create_log_dir
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier


class TrainingExample:
    def __init__(self, obs, pos):
        assert isinstance(obs, np.ndarray)
        assert isinstance(pos, (tuple, np.ndarray))

        self.obs = obs
        self.pos = pos
    

class BOVWClassifier:
    """
    bag of visual words (BOVW) classifier
    used for image classification
    args:
        num_clusters: number of clusters to use for kmeans clustering
    """
    def __init__(self, num_clusters=50, num_sift_keypoints=None):
        self.num_clusters = num_clusters
        self.kmeans_cluster = None
        self.svm_classifier = None
        if num_sift_keypoints is not None:
            self.sift_detector = cv2.SIFT_create(nfeatures=num_sift_keypoints)
        else:
            self.sift_detector = cv2.SIFT_create()
    
    def is_initialized(self):
        return self.kmeans_cluster is not None and self.svm_classifier is not None

    def fit(self, X=None, Y=None, svm_type='svc', gamma='scale', nu=0.1):
        """
        train the classifier in 3 steps
        1. train the kmeans classifier based on the SIFT features of images
        2. extract the histogram of the SIFT features
        3. train an SVM classifier based on the histogram features
        args:
            x: a list of images
            y: a list of labels (str)
        """
        assert svm_type in ['svc', 'one_class_svm']
        X = list(X)  # in case X, Y are numpy arrays
        Y = list(Y)

        # get sift features
        sift_features = self.get_sift_features(images=X)
        # train kmeans
        self.train_kmeans(sift_features=sift_features)
        # get histogram
        hist_features = self.histogram_from_sift(sift_features=sift_features)
        # train svm
        if svm_type == 'svc':
            class_weight = 'balanced' if len(X) > 10 else None
            self.svm_classifier = svm.SVC(class_weight=class_weight)
        elif svm_type == 'one_class_svm':
            self.svm_classifier = svm.OneClassSVM(gamma=gamma, nu=nu)
        self.svm_classifier.fit(hist_features, Y)

    def predict(self, X):
        """
        test the classifier
        args:
            X: a list of images
        """
        assert self.svm_classifier is not None
        X = list(X)  # in case X is numpy array

        # preprocess the images
        sift_features = self.get_sift_features(images=X)  # get sift features
        hist_features = self.histogram_from_sift(sift_features=sift_features)  # get histogram

        return self.svm_classifier.predict(hist_features)
    
    def save(self, save_path):
        """
        save the classifier to disk
        """
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, load_path):
        """
        init a classifier by loading it from disk
        """
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    
    def get_sift_features(self, images):
        """
        extract the SIFT features of a list of images
        args:
            images: a list of RGB/GrayScale images
        return:
            a list of SIFT features
        """
        images = [cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) for img in images]
        keypoints = self.sift_detector.detect(images)
        keypoints, descriptors = self.sift_detector.compute(images, keypoints)
        return descriptors  # type: tuple
    
    def train_kmeans(self, sift_features):
        """
        train the kmeans classifier using the SIFT features
        args:
            sift_features: a list of SIFT features
        """
        # reshape the data
        # each image has a different number of descriptors, we should gather 
        # them together to train the clustering
        sift_features=np.array(sift_features, dtype=object)
        sift_features=np.concatenate(sift_features, axis=0).astype(np.float32)

        # train the kmeans classifier
        if self.kmeans_cluster is None:
            self.kmeans_cluster = cluster.MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0).fit(sift_features)
        else:
            self.kmeans_cluster.partial_fit(sift_features)
    
    def histogram_from_sift(self, sift_features):
        """
        transform the sift features by putting the sift features into self.num_clusters 
        bins in a histogram. The counting result is a new feature space, which
        is used directly by the SVM for classification
        args:
            sift_features: a list of SIFT features
        
        this code is optimized for speed, but it's doing essentially the following:
        hist_features = []
        for sift in sift_features:
            # classification of all descriptors in the model
            # sift.shape == (n_descriptors, 128)
            predicted_cluster = self.kmeans_cluster.predict(sift)  # (n_descriptors,)
            # calculates the histogram
            # hist, bin_edges = np.histogram(predicted_cluster, bins=self.num_clusters)  # (num_clusters,)
            hist = np.bincount(predicted_cluster, minlength=self.num_clusters)  # (num_clusters,)
            # histogram is the feature vector
            hist_features.append(hist)

        hist_features = np.asarray(hist_features)
        return hist_features
        """
        assert self.kmeans_cluster is not None, "kmeans classifier not trained"

        n_descriptors_per_image = [len(sift) for sift in sift_features]
        idx_num_descriptors = list(itertools.accumulate(n_descriptors_per_image))
        sift_features_of_all_images = np.concatenate(sift_features, axis=0).astype(np.float32)

        predicted_cluster_of_all_images = self.kmeans_cluster.predict(sift_features_of_all_images)  # (num_examples,)
        predicted_clusters = np.split(predicted_cluster_of_all_images, indices_or_sections=idx_num_descriptors)
        predicted_clusters.pop()  # remove the last element, which is empty due to np.split
        
        hist_features = np.array([np.bincount(predicted_cluster, minlength=self.num_clusters) for predicted_cluster in predicted_clusters])
        return hist_features


class SiftInitiationClassifier(InitiationClassifier):
    def __init__(self, num_clusters=50, num_sift_keypoints=None, gamma='scale', nu=0.1, class_weight='balanced', buffer_size=100, pessimistic_type='one_class_svm'):
        optimistic_classifier = BOVWClassifier(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)
        pessimistic_classifier = BOVWClassifier(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)

        self.gamma = gamma
        self.nu = nu
        self.class_weight = class_weight

        self.positive_examples = deque([], maxlen=buffer_size)
        self.negative_examples = deque([], maxlen=buffer_size)

        assert pessimistic_type in ['one_class_svm', 'svc']
        self.pessimistic_type = pessimistic_type
        if pessimistic_type == 'one_class_svm':
            self.extra_pessimistic_classifier = BOVWClassifier(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)
        
        super().__init__(optimistic_classifier, pessimistic_classifier)
    
    def is_initialized(self):
        if self.pessimistic_type == 'one_class_svm':
            return self.optimistic_classifier.is_initialized() and self.pessimistic_classifier.is_initialized() \
                and self.extra_pessimistic_classifier.is_initialized()
        else:
            return self.optimistic_classifier.is_initialized() and self.pessimistic_classifier.is_initialized()

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


def load_training_and_testing_data(goal_position=(133, 148), select_training_points=lambda x, y: x >= 90):
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
            pos = get_player_position(ram)
            if 0 <= np.linalg.norm(np.array(pos) - np.array(goal_position)) < 30:
                # a positive example
                if pos not in seen_pos:
                    if select_training_points(*pos):
                        if random.random() < 0.5:
                            positive_training.append((obs, pos))
                        else:
                            positive_testing.append((obs, pos))
                    else:
                        positive_testing.append((obs, pos))
                    seen_pos.add(pos)
            else:
                # a negative example
                if pos not in seen_pos:
                    if select_training_points(*pos):
                        if random.random() < 0.5:
                            negative_training.append((obs, pos))
                        else:
                            negative_testing.append((obs, pos))
                    else:
                        negative_testing.append((obs, pos))
                    seen_pos.add(pos)
    print(len(positive_training), len(positive_testing), len(negative_training), len(negative_testing))

    with open("logs/trajectories/training_data.pkl", "wb") as f:
        pickle.dump((positive_training, positive_testing, negative_training, negative_testing), f)

    return positive_training, positive_testing, negative_training, negative_testing

    # traj_generator = load_trajectories("logs/trajectories/monte_rnd_full_trajectories.pkl.gz", skip=0)
    # positive_data = deque(maxlen=100)
    # negative_data = deque(maxlen=500)
    # positive_pos = set()
    # negative_pos = set()
    # for traj in traj_generator:
    #     if len(positive_data) >= 100:
    #         break
    #     for state in traj:
    #         ram, obs = state
    #         pos = get_player_position(ram)
    #         if 0 <= np.linalg.norm(np.array(pos) - np.array(goal_position)) < 30:
    #             if pos not in positive_pos:
    #                 positive_data.append((obs, pos))
    #                 positive_pos.add(pos)
    #         else:
    #             if pos not in negative_pos:
    #                 negative_data.append((obs, pos))
    #                 negative_pos.add(pos)
    # print(len(positive_data), len(negative_data))
    # return positive_data, negative_data


def load_data_from_same_position(position=(133, 148)):
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
            pos = get_player_position(ram)
            skull_pos = get_skull_position(ram)
            largest_skull_pos = max(largest_skull_pos, skull_pos)
            if np.array_equal(pos, position):
                data.append((obs, (skull_pos, 148)))
    print(f"Loaded {len(data)} data points from position {position}")

    with open("logs/trajectories/same_pos_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return data


def train_test_split(positive, negative, test_size=0.2):
    """
    input is two lists
    """
    from sklearn.model_selection import train_test_split
    positive_train, positive_test = train_test_split(positive, test_size=test_size)
    negative_train, negative_test = train_test_split(negative, test_size=test_size)
    return positive_train, positive_test, negative_train, negative_test


def test_classifier(positive_train, positive_test, negative_train, negative_test, 
                    same_pos_data, pessimistic_type, plot_dir,
                    gamma, nu, class_weight, num_kp, num_clusters):
    svc_classifier = SiftInitiationClassifier(
        pessimistic_type=pessimistic_type,
        gamma=gamma,
        nu=nu,
        num_sift_keypoints=num_kp,
        num_clusters=num_clusters,
    )
    svc_classifier.add_positive_examples(*zip(*positive_train))
    svc_classifier.add_negative_examples(*zip(*negative_train))
    svc_classifier.fit_initiation_classifier()

    # training plot
    plot_path = os.path.join(plot_dir, f"{pessimistic_type}_classifier_train.png")
    svc_classifier.plot_training_predictions(plot_path)

    # testing plot
    svc_classifier.positive_examples = []
    svc_classifier.negative_examples = []
    svc_classifier.add_positive_examples(*zip(*positive_test))
    svc_classifier.add_negative_examples(*zip(*negative_test))
    plot_path = os.path.join(plot_dir, f"{pessimistic_type}_classifier_test.png")
    svc_classifier.plot_training_predictions(plot_path)

    # same position plot
    svc_classifier.positive_examples = []
    svc_classifier.negative_examples = [[svc_classifier.negative_examples[0][0]]]
    assert len(svc_classifier.negative_examples) == 1
    svc_classifier.add_positive_examples(*zip(*same_pos_data))
    plot_path = os.path.join(plot_dir, f"{pessimistic_type}_classifier_same_pos.png")
    svc_classifier.plot_training_predictions(plot_path)


if __name__ == "__main__":
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.075)
    parser.add_argument("--num_kp", type=int, default=30)
    parser.add_argument("--num_cluster", type=int, default=99)
    args = parser.parse_args()

    plot_dir = f"plots/gamma_{args.gamma}_nu_{args.nu}_num_kp_{args.num_kp}_num_cluster_{args.num_cluster}"
    create_log_dir("plots")
    create_log_dir(plot_dir)

    positive_train, positive_test, negative_train, negative_test = load_training_and_testing_data()
    assert len(positive_train) > 0
    assert len(negative_train) > 0
    assert len(positive_test) > 0
    assert len(negative_test) > 0
    same_pos_data = load_data_from_same_position()
    assert len(same_pos_data) > 0
    # test_classifier(positive_train, positive_test, negative_train, negative_test, 
    #                 same_pos_data, pessimistic_type='svc', plot_dir=plot_dir,
    #                 gamma=args.gamma, nu=args.nu, class_weight=args.class_weight, num_kp=args.num_kp)
    test_classifier(positive_train, positive_test, negative_train, negative_test, 
                    same_pos_data, pessimistic_type='one_class_svm', plot_dir=plot_dir,
                    gamma=args.gamma, nu=args.nu, class_weight=None, num_kp=args.num_kp, num_clusters=args.num_cluster)
