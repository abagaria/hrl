import math
import random
import pickle
import itertools
from collections import deque

# import cudasift
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, svm
from pfrl.wrappers import atari_wrappers
from hrl.agent.dsc.datastructures import TrainingExample
from hrl.agent.dsc.classifier.init_classifier import InitiationClassifier
    

class BOVWClassifier:
    """
    bag of visual words (BOVW) classifier
    used for image classification
    args:
        num_clusters: number of clusters to use for kmeans clustering
    """
    def __init__(self, num_clusters=55, sift_threshold=7):
        self.num_clusters = num_clusters
        self.kmeans_cluster = None
        self.svm_classifier = None
        self.sift_data = cudasift.PySiftData(100)
        self.sift_threshold = sift_threshold
    
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
            self.svm_classifier = svm.SVC(class_weight=class_weight, random_state=0)
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
        descriptors = []
        for image in images:   
            cudasift.ExtractKeypoints(image, self.sift_data, thresh=7)
            df, descriptor = self.sift_data.to_data_frame()
            descriptors.append(descriptor[:len(df), :])
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
            if len(sift_features) < self.num_clusters:
                # can't train kmeans with less example than clusters
                # just duplicate the examples
                num_duplicate = math.ceil(self.num_clusters / len(sift_features))
                sift_features = np.tile(sift_features, (num_duplicate, 1))
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
    def __init__(self, num_clusters=55, sift_threshold=7, gamma=0.15, nu=0.05, buffer_size=100):
        optimistic_classifier = BOVWClassifier(num_clusters=num_clusters, sift_threshold=sift_threshold)
        self.pessimistic_classifier_for_pos_data = BOVWClassifier(num_clusters=num_clusters, sift_threshold=sift_threshold)
        self.pessimistic_classifier_for_neg_data = BOVWClassifier(num_clusters=num_clusters, sift_threshold=sift_threshold)

        self.gamma = gamma
        self.nu = nu

        self.positive_examples = deque([], maxlen=buffer_size)
        self.negative_examples = deque([], maxlen=buffer_size)
        
        super().__init__(optimistic_classifier, self.pessimistic_classifier_for_pos_data)
    
    def is_initialized(self):
        return (self.optimistic_classifier.is_initialized() 
            and self.pessimistic_classifier_for_pos_data.is_initialized()
            and self.pessimistic_classifier_for_neg_data.is_initialized())

    def optimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        return self.optimistic_classifier.predict([state])[0] == 1

    def pessimistic_predict(self, state):
        assert isinstance(state, np.ndarray)
        assert state.shape == (84, 84)
        
        def pos_predict(s, clf):
            return clf.is_initialized() and clf.predict([s]) == 1

        def neg_predict(s, clf):
            return clf.is_initialized() and clf.predict([s]) != 1

        pos_label = pos_predict(state, self.pessimistic_classifier_for_pos_data)
        neg_label = neg_predict(state, self.pessimistic_classifier_for_neg_data)

        return pos_label and neg_label

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
        images = [example.obs._frames[-1].squeeze() for trajectory in examples for example in trajectory]
        return images

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
            self.pessimistic_classifier_for_pos_data.fit(positive_training_examples, Y[training_predictions == 1], 
                                    svm_type='one_class_svm', gamma=self.gamma, nu=self.nu)
        if len(negative_training_examples) > 0:
            self.pessimistic_classifier_for_neg_data.fit(negative_training_examples, [1 for _ in negative_training_examples],
                                    svm_type='one_class_svm', gamma=self.gamma, nu=self.nu)

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

    def plot_training_predictions(self, option_name, episode, experiment_name, seed):
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
        plt.colorbar()
        plt.legend()
        plt.title("Pessimistic classifier")

        plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option_name}_init_clf_episode_{episode}.png")
        plt.close()
