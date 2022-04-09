import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import itertools
import cv2
from sklearn import cluster

from .feature_extractor import FeatureExtractor

'''
Reference to Paul Zhou's repo (https://github.com/zhouzypaul/bovw-classifier)
when implementing BOVW
'''
class BOVW_stack(FeatureExtractor):
    def __init__(self, num_clusters=100, num_sift_keypoints=None):
        '''
	Bag of visual words (BOVW) classifier used for image classification

	Args:
	    num_clusters: number of clusters to use for kmeans clustering
	    num_sift_keypoints: number of SIFT keypoints to use for SIFT extraction,
				if None, use cv2 will auto pick a number of keypoints
	'''
        self.num_clusters = num_clusters
        self.num_keypoints = num_sift_keypoints
        self.kmeans_cluster = None

        if num_sift_keypoints is not None:
            self.sift_detector = cv2.SIFT_create(nfeatures=num_sift_keypoints)
        else:
            self.sift_detector = cv2.SIFT_create()


    def train(self, states):
        '''
        Train the BOVW feature extractor by extracting sift features and
        training the kmeans classifier

        Args:
            states (list(list(np.array))): frame stack
        '''
        # get sift features
        sift_feats = self.get_sift_features(states)

        # train kmeans
        self.train_kmeans(sift_feats)

    def extract_features(self, states):
        '''
	Extract features using Bag of Visual Words (BOVW)

        Args:
            states (list(list(np.array))): frame stack

        Returns:
            (list(np.array)): list of histograms of extracted features
        '''
        # get sift features
        sift_feats = self.get_sift_features(states)

        # get histogram
        hist_feats = self.histogram_from_sift(sift_feats)

        return hist_feats

    def get_sift_features(self, states):
        '''
        Extract the SIFT features of a list of frame stacks

        Args:
            states (list(list(np.array))): frame stack

        Returns:
            a list of SIFT features
        '''
        states = [[cv2.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U).squeeze() for frame in state] for state in states]

        descriptors = []
        for frame_stack in states:
            keypoints = self.sift_detector.detect(frame_stack)
            keypoints, stacked_descriptors = self.sift_detector.compute(frame_stack, keypoints)
            stacked_descriptors = [desc[:self.num_keypoints] for desc in stacked_descriptors]
            stacked_descriptors = np.concatenate(stacked_descriptors, axis=1)
            descriptors.append(stacked_descriptors)
        return descriptors

    def train_kmeans(self, sift_feats):
        '''
        Train the kmeans classifier using the SIFT features

        Args:
            sift_feats (list(list(np.array))): list of list of SIFT features
        '''
        # reshape the data
        # each image has a different number of descriptors, we should gather 
        # them together to train the clustering
        sift_feats=np.array(sift_feats, dtype=object)
        sift_feats=np.concatenate(sift_feats, axis=0).astype(np.float32)

        # train the kmeans classifier
        if self.kmeans_cluster is None:
            self.kmeans_cluster = cluster.MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0).fit(sift_feats)
        else:
            self.kmeans_cluster.partial_fit(sift_feats)
    
    def histogram_from_sift(self, sift_feats):
        '''
        Transform the sift features by putting the sift features into self.num_clusters 
        bins in a histogram. The counting result is a new feature space, which
        is used directly by the SVM for classification

        Args:
            sift_feats (list(np.array)): a list of SIFT features

        Returns:
            (list(np.array)): list of np.array of extracted features
        '''
        assert self.kmeans_cluster is not None, "kmeans classifier not trained"

        n_descriptors_per_image = [len(sift) for sift in sift_feats]
        idx_num_descriptors = list(itertools.accumulate(n_descriptors_per_image))
        sift_feats_of_all_images = np.concatenate(sift_feats, axis=0).astype(np.float32)

        predicted_cluster_of_all_images = self.kmeans_cluster.predict(sift_feats_of_all_images)  # (num_examples,)
        predicted_clusters = np.split(predicted_cluster_of_all_images, indices_or_sections=idx_num_descriptors)
        predicted_clusters.pop()  # remove the last element, which is empty due to np.split
        
        hist_features = [np.bincount(predicted_cluster, minlength=self.num_clusters) for predicted_cluster in predicted_clusters]
        return hist_features

    def visualize_sift_feats(self, states, save_dir):
        states = [cv2.normalize(src=state, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U).squeeze() for state in states]
        for i, state in enumerate(states):
            state = cv2.cvtColor(state, cv2.COLOR_GRAY2BGR)
            keypoints, descriptors = self.sift_detector.detectAndCompute(state, None)
            sift_image = cv2.drawKeypoints(state, keypoints, state)
            cv2.imwrite(save_dir + f"/sifts={self.num_keypoints}_clusters={self.num_clusters}_{i}_sift.png", sift_image)
