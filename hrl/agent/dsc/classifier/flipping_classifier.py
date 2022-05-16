import ipdb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from hrl.utils import flatten
from hrl.agent.dsc.datastructures import TrainingExample
from hrl.agent.dsc.classifier.mlp_classifier import BinaryMLPClassifier


class FlippingClassifier:
    """ Classifier that takes TrainingExamples and predicts their probability of changing init labels. """
    def __init__(self, obs_dim, device, classifier_type, feature_extractor_type):
        assert classifier_type in ("svm", "nn"), classifier_type
        assert feature_extractor_type in ("pos", "obs", "augmented_obs")

        self.device = device
        self.obs_dim = obs_dim

        self.classifier = None
        self.classifier_type = classifier_type
        self.feature_extractor_type = feature_extractor_type

    def __call__(self, states):
        """ Given states, predict the probability of flipping their labels. """
        assert isinstance(states, list), states
        assert isinstance(states[0], TrainingExample), states[0]

        if self.classifier is None:
            return np.zeros((len(states),))
        
        X = self.extract_features(states)
        if self.classifier_type == "svm":
            return self._svm_predict(X)
        return self._nn_predict(X)

    def fit(self, examples, assigned_labels, vf_old_labels, vf_new_labels):
        """ Train a classifier to predict the probability an init label getting flipped.
        
        Args:
          - examples: training examples collected by running the option policy
          - assigned_labels: init label assigned when the option was executed
          - vf_old_labels: init label that would have been assigned by the VF during execution
          - vf_new_labels: init label that would be assigned by the VF *now*

        """
        assert isinstance(examples, list)
        assert isinstance(examples[0], TrainingExample)
        assert isinstance(assigned_labels, np.ndarray)
        assert isinstance(vf_old_labels, np.ndarray)
        assert isinstance(vf_new_labels, np.ndarray)

        X = self.extract_features(examples)
        Y = self.extract_labels(assigned_labels, vf_old_labels, vf_new_labels)

        # If there are no flips in the data, we cannot fit a classifier
        if (Y == 0).all():
            print("Not fitting a classifier because no flips observed yet")
            return

        if self.classifier_type == "svm":
            self._fit_svm_classifier(X, Y)

        elif self.classifier_type == "nn":
            self._fit_nn_classifier(X, Y)

        else:
            raise NotImplementedError(self.classifier_type)
    
    def extract_features(self, examples):
        if self.feature_extractor_type == "pos":
            return np.array([eg.pos for eg in examples])
        if self.feature_extractor_type == "obs":
            return np.array([eg.obs for eg in examples])
        raise NotImplementedError(self.feature_extractor_type)

    def extract_labels(self, assigned_labels, vf_old_labels, vf_new_labels):
        """ Rules for label extraction: 
          - Let L0 = assigned_labels, Y1 = vf_old_labels, Y2 = vf_new_labels
          - When L0 == Y1, its a flip when Y1 != Y2
          - When L0 != Y1, its a flip when Y1 == Y2
          - All other labels are non-flips
        """
        assert len(assigned_labels) == len(vf_old_labels) == len(vf_new_labels)

        initially_agree_idx = np.where(assigned_labels == vf_old_labels)[0]
        initially_disagree_idx = np.where(assigned_labels != vf_old_labels)[0]

        Y = np.zeros((assigned_labels.shape[0],))

        Y[initially_agree_idx] = vf_old_labels[initially_agree_idx] != vf_new_labels[initially_agree_idx]
        Y[initially_disagree_idx] = vf_old_labels[initially_disagree_idx] == vf_new_labels[initially_disagree_idx]

        return Y

    def _svm_predict(self, X):
        assert isinstance(X, np.ndarray), X
        assert len(X.shape) == 2, X.shape
        probs = self.classifier.predict_proba(X)
        flipping_probs = probs[:, 1]
        return flipping_probs

    def _nn_predict(self, X):
        assert isinstance(X, np.ndarray), X
        assert len(X.shape) == 2, X.shape
        assert isinstance(self.classifier, BinaryMLPClassifier)
        flipping_probs = self.classifier.predict_proba(X)
        return flipping_probs

    def _fit_svm_classifier(self, X, Y):
        assert len(X.shape) == 2, X.shape
        assert len(Y.shape) == 1, Y.shape
        assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)

        self.classifier = SVC(
            gamma="scale",
            probability=True,
            class_weight="balanced"
        )

        self.classifier.fit(X, Y)

    def _fit_nn_classifier(self, X, Y):
        assert len(X.shape) == 2, X.shape
        assert len(Y.shape) == 1, Y.shape
        assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)

        self.classifier = BinaryMLPClassifier(
            self.obs_dim,
            self.device,
        )

        self.classifier.fit(X, Y, W=None)
