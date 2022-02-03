import os
import cv2
import gzip
import random
import pickle
import argparse
import itertools
from collections import deque

import numpy as np
import cudasift
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
        self.data = cudasift.PySiftData(1000)
    
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
        # keypoints = self.sift_detector.detect(images)
        # keypoints, descriptors = self.sift_detector.compute(images, keypoints)

        descriptors = []
        for stacked_image in images: 
            # each image is a stacked image, so we need to extract sift features for each
            stacked_descriptors = []
            for image in stacked_image:
                cudasift.ExtractKeypoints(image, self.data, thresh=7)
                df, descriptor = self.data.to_data_frame()
                stacked_descriptors.append(descriptor[:len(df), :])
            stacked_descriptors = np.concatenate(stacked_descriptors, axis=0)
            descriptors.append(stacked_descriptors)

        return descriptors
    
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
    def __init__(self, num_clusters=55, num_sift_keypoints=None, gamma=0.15, nu=0.1, buffer_size=100):
        optimistic_classifier = BOVWClassifier(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)
        pessimistic_classifier = BOVWClassifier(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)

        self.gamma = gamma
        self.nu = nu

        self.positive_examples = deque([], maxlen=buffer_size)
        self.negative_examples = deque([], maxlen=buffer_size)

        self.extra_pessimistic_classifier = BOVWClassifier(num_clusters=num_clusters, num_sift_keypoints=num_sift_keypoints)
        
        super().__init__(optimistic_classifier, pessimistic_classifier)
    
    def is_initialized(self):
        return self.optimistic_classifier.is_initialized() and self.pessimistic_classifier.is_initialized() \
            and self.extra_pessimistic_classifier.is_initialized()

    def optimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (4, 84, 84)
        return self.optimistic_classifier.predict([state])[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (4, 84, 84)
        return self.pessimistic_classifier.predict([state])[0] == 1 and self.extra_pessimistic_classifier.predict([state])[0] == -1

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
            self.pessimistic_classifier.fit(positive_training_examples, Y[training_predictions == 1], 
                                            svm_type='one_class_svm', gamma=self.gamma, nu=self.nu)
            self.extra_pessimistic_classifier.fit(negative_training_examples, [1 for _ in negative_training_examples], 
                                                    svm_type='one_class_svm', gamma=self.gamma, nu=self.nu)


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


def load_training_data(goal_pos=(109, 148), goal_epsilon=7):
    """
    return positive_train, negative_train
    all the state are framestacks of 4 
    """
    positive_training = []
    negative_training = []
    with open("logs/trajectories/positive.pkl", "rb") as f:
        positive_trajectories = pickle.load(f)
    with open("logs/trajectories/negative.pkl", "rb") as f:
        negative_trajectories = pickle.load(f)
    for traj in positive_trajectories:
        for state, ram in traj:
            pos = get_player_position(ram)
            if np.linalg.norm(np.array(pos) - np.array(goal_pos)) < goal_epsilon:
                positive_training.append((state, pos))
            else:
                negative_training.append((state, pos))
    negative_training += [(state, get_player_position(ram)) for traj in negative_trajectories for state, ram in traj]
    return positive_training, negative_training


def load_testing_data():
    """
    return testing_data
    all states are framestacks of 4
    """
    testing_data = []
    with open("logs/trajectories/random_traj.pkl", "rb") as f:
        testing_trajectories = pickle.load(f)
    for traj in testing_trajectories:
        for state, ram in traj:
            testing_data.append((state, get_player_position(ram)))
    return testing_data


def train_classifier(clf, positive_train, negative_train, plot_dir):
    # add examples
    clf.add_positive_examples(*zip(*positive_train))
    clf.add_negative_examples(*zip(*negative_train))

    # train classifier
    clf.fit_initiation_classifier()

    # plot
    plot_path = os.path.join(plot_dir, f"classifier_train.png")
    clf.plot_training_predictions(plot_path)


def test_classifier(clf, test_data, plot_dir):
    """
    test the clf by just crusing the agent around randomly
    """
    optimistic_positive_pos = []
    optimistic_negative_pos = []
    pessimistic_positive_pos = []
    pessimistic_negative_pos = []
    positive_states = []
    for test_point in test_data:
        # each test point is (state, pos)
        state, pos = test_point
        optimisitic_prediction = clf.optimistic_predict(state)
        pessimistic_prediction = clf.pessimistic_predict(state)

        if optimisitic_prediction:
            optimistic_positive_pos.append(pos)
            positive_states.append(state)
        else:
            optimistic_negative_pos.append(pos)

        if pessimistic_prediction:
            pessimistic_positive_pos.append(pos)
            positive_states.append(state)
        else:
            pessimistic_negative_pos.append(pos)

    # optimistic predictions
    if len(optimistic_positive_pos) > 0:
        plt.scatter(*zip(*optimistic_positive_pos), marker='+', alpha=0.3)
    if len(optimistic_positive_pos) > 0:
        plt.scatter(*zip(*optimistic_negative_pos), marker='o', alpha=0.3)
    plt.clim(0, 1)
    plt.xlim(15, 140)
    plt.ylim(140, 260)
    plt.title('player positions: optimistic clf')
    plt.show()
    plt.savefig(os.path.join(plot_dir, "optimistic.png"))
    plt.close()

    # pessimistic predictions
    if len(pessimistic_positive_pos):
        plt.scatter(*zip(*pessimistic_positive_pos), marker='+', alpha=0.3)
    if len(pessimistic_negative_pos):
        plt.scatter(*zip(*pessimistic_negative_pos), marker='o', alpha=0.3)
    plt.clim(0, 1)
    plt.xlim(15, 140)
    plt.ylim(140, 260)
    plt.title('player positions: pessimistic clf')
    plt.show()
    plt.savefig(os.path.join(plot_dir, "pessimistic.png"))
    plt.close()

    # positive predictions
    for i, stacked_state in enumerate(positive_states):
        sub_dir = os.path.join(plot_dir, f"{i}")
        create_log_dir(sub_dir)
        for idx_s, state in enumerate(stacked_state):
            plt.imshow(state)
            plt.savefig(os.path.join(sub_dir, f"{idx_s}.png"))
            plt.close()


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

    plot_dir = f"plots/stacked_classifier"
    create_log_dir("plots")
    create_log_dir(plot_dir)

    positive_train, negative_train = load_training_data()
    assert len(positive_train) > 0
    assert len(negative_train) > 0
    testing_data = load_testing_data()
    assert len(testing_data) > 0

    # init classifier
    classifier = SiftInitiationClassifier()

    train_classifier(classifier, positive_train, negative_train, plot_dir)
    test_classifier(classifier, testing_data, plot_dir)
