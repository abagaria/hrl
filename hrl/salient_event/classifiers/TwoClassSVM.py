import numpy as np
from sklearn.svm import SVC

from .classifier import Classifier
from utils.plotting import plot_SVM

class TwoClassSVMClassifier(Classifier):
    def __init__(self, feature_extractor, window_sz=1, gamma='scale'):
        '''
        Args:
            feature_extractor: obj that extracts features by calling extract_features()
        '''
        self.feature_extractor = feature_extractor
        self.window_sz = window_sz
        self.gamma = gamma

    def train(self, X, Y):
        '''
        Train classifier using X and Y. 

        Note that there might be more than 2 classes, such as when using the TransductiveExtractor.
        As the SVC is a multi-class classifier, this will still work.

        Args:
            states (list(np.array)): list of np.array
            X (list(np.array or MonteRAMState)): list of states
            Y (list(int)): class labels for states in X

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        self.X = X
        self.Y = Y
        self.__update_term_classifier()

    def __update_term_classifier(self):
        states_features = self.feature_extractor.extract_features(self.X)
        feature_matrix = self.__construct_feature_matrix(states_features)

        self.term_classifier = SVC(kernel='rbf', gamma=self.gamma, class_weight='balanced')
        self.term_classifier.fit(feature_matrix, self.Y)

    def __construct_feature_matrix(self, states_features):
        return (np.array([np.reshape(state_features, (-1,)) for state_features in states_features]))

    def predict(self, states):
        '''
        Predict whether state is in the term set.

        Note that when using the TransductiveExtractor, there are 3 classes: 
            1. Positive (1)
            2. Negative, in subgoal traj (0)
            3. Negative, all states outside subgoal traj (2)

        When using the PositiveAugmentExtractor, there is an additional class:
            4. Positive, states outside subgoal traj that is above cos similarity threshold (3)

        Args:
            states (list(np.array or MonteRAMState)): list of states to predict on

        Returns:
            (list(bool): whether states are in the term set
        '''
        states_features = self.feature_extractor.extract_features(states)
        features = np.array([np.reshape(state_features, (-1,)) for state_features in states_features])
        predict = self.term_classifier.predict(features)
        return [pred == 1 or pred == 3 for pred in predict]

    def predict_raw(self, states):
        '''
        Predict class label of states. For the TransductiveExtractor, the labels are 0, 1 and 2. 
        For other label extractors, the labels are 0 and 1.

        Args:
            states (list(np.array or MonteRAMState)): list of states to predict on

        Returns:
            (list(int)): predicted class label of states
        '''
        features = np.array(self.feature_extractor.extract_features(states))
        return self.term_classifier.predict(features)
