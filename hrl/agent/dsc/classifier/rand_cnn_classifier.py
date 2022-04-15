import math
import random
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from sklearn import svm
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class RandCNNClassifier:
    """
    Randomly-initialized CNN classifier
    used for image classification
    args:
        batch_size: batch size used for feature extraction
    """
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

	# Initialize conv layers
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=3),
            nn.MaxPool2d(
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.MaxPool2d(
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(6400, 512)
        )
        self.model.cuda()

        for p in self.model:
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.model.parameters():
            param.requires_grad = False        
    
    def is_initialized(self):
        return self.model is not None

    def fit(self, X=None, Y=None, svm_type='svc', gamma='scale', nu=0.1):
        """
	Train the classifier
        args:
            x: a list of images
            y: a list of labels (str)
        """
        assert svm_type in ['svc', 'one_class_svm']
        X = list(X)  # in case X, Y are numpy arrays
        Y = list(Y)

        # Extract features from images
        feats = self.extract_features(X)
        feat_matrix = np.array([np.reshape(feat, (-1,)) for feat in feats])

        # train svm
        if svm_type == 'svc':
            self.svm_classifier = svm.SVC(kernel='rbf', gamma=gamma, class_weight='balanced')
        elif svm_type == 'one_class_svm':
            self.svm_classifier = svm.OneClassSVM(gamma=gamma, nu=nu)
        self.svm_classifier.fit(feat_matrix, Y)

    def predict(self, X):
        """
        test the classifier
        args:
            X: a list of images
        """
        assert self.svm_classifier is not None
        X = list(X)  # in case X is numpy array

        # preprocess the images
        feats = self.extract_features(X)
        feat_matrix = np.array([np.reshape(feat, (-1,)) for feat in feats])

        return self.svm_classifier.predict(feat_matrix)
    
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
    
    def extract_features(self, images):
        features = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            batch_sz = len(batch_images)
            flattened_images = [frame for frame_stack in batch_images for frame in frame_stack]
            flattened_images = np.stack(flattened_images, axis=0) # flatten list of list of frames into list of single frames

            batch_features = self.model(torch.from_numpy(flattened_images).float().to("cuda:0"))
            batch_features = batch_features.cpu().numpy()
            batch_features = np.split(batch_features, indices_or_sections=batch_sz) # split list of single frames into list of frame stacks
            #batch_features = [np.squeeze(x) for x in np.split(batch_features, np.size(batch_features, 0))]
            features.extend(batch_features)
        return features
