import ipdb
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from hrl.agent.dsc.classifier.utils import ObsClassifierMLP
from hrl.agent.dsc.classifier.initiation_gvf import GoalConditionedInitiationGVF


class BinaryMLPClassifier:
    """" Generic binary neural network classifier. """
    def __init__(self,
                obs_dim,
                device,
                threshold=0.5,
                batch_size=128):
        
        self.device = device
        self.is_trained = False
        self.threshold = threshold
        self.batch_size = batch_size

        self.model = ObsClassifierMLP(obs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Debug variables
        self.losses = []

    @torch.no_grad()
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).to(self.device).float()
        logits = self.model(X)
        probabilities = torch.sigmoid(logits)
        return probabilities
        
    @torch.no_grad()
    def predict(self, X, threshold=None):
        logits = self.model(X)
        probabilities = torch.sigmoid(logits)
        threshold = self.threshold if threshold is None else threshold
        return probabilities > threshold

    def determine_pos_weight(self, y):
        n_negatives = len(y[y != 1])
        n_positives = len(y[y == 1])
        if n_positives > 0:
            pos_weight = (1. * n_negatives) / n_positives
            return torch.as_tensor(pos_weight).float()

    def should_train(self, y):
        enough_data = len(y) > self.batch_size
        has_positives = len(y[y == 1]) > 0
        has_negatives = len(y[y != 1]) > 0
        return enough_data and has_positives and has_negatives
    
    @torch.no_grad()
    def determine_instance_weights(
        self,
        states: np.ndarray,
        labels: torch.Tensor,
        init_gvf: GoalConditionedInitiationGVF,
        goal: np.ndarray  # goal that was pursued in the last option rollout
    ):
        goal = goal.astype(np.float32, copy=False)
        goals = np.repeat(goal[np.newaxis, ...], repeats=len(states), axis=0)
        assert isinstance(states, np.ndarray), 'Conversion done in TD(0)'
        assert isinstance(goal, np.ndarray), 'Conversion done in TD(0)'
        assert states.dtype == goals.dtype == np.float32, (states.dtype, goals.dtype)
        values = init_gvf.get_values(states, goals)
        values = torch.as_tensor(values).float().to(self.device)  # TODO(ab): keep these on GPU
        # weights = values.clip(0., 1.) # no need for clipping b/c of sigmoid in GVF
        values[labels == 0] = 1. - values[labels == 0]
        return values

    def fit(self, X, y, initiation_gvf=None, goal=None, n_epochs=5):
        dataset = ClassifierDataset(X, y)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        if self.should_train(y):
            losses = []

            for _ in range(n_epochs):
                epoch_loss = self._train(dataloader, initiation_gvf, goal)                
                losses.append(epoch_loss)
            
            self.is_trained = True

            mean_loss = np.mean(losses)
            print(mean_loss)
            self.losses.append(mean_loss)

    def _train(self, loader, initiation_gvf=None, goal=None):
        """ Single epoch of training. """
        batch_losses = []

        for sample in loader:
            observations, labels, weights = self._extract_sample(sample)

            pos_weight = self.determine_pos_weight(labels)

            if pos_weight is None or pos_weight.nelement == 0:
                continue

            if initiation_gvf is not None:
                weights = self.determine_instance_weights(
                    sample[0].detach().cpu().numpy(), # DataLoader converts to tensor, undoing that here
                    labels,  # This is a tensor on the GPU
                    initiation_gvf,
                    goal
                )

            logits = self.model(observations)

            loss = F.binary_cross_entropy_with_logits(logits.squeeze(1),
                                                      labels,
                                                      pos_weight=pos_weight,
                                                      weight=weights) 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())
        
        return np.mean(batch_losses)
    
    def _extract_sample(self, sample):
        weights = None

        if len(sample) == 3:
            observations, labels, weights = sample
            weights = weights.to(self.device).float().squeeze()
        else:
            observations, labels = sample

        observations = observations.to(self.device).float()
        labels = labels.to(self.device)

        return observations, labels, weights


class ClassifierDataset(Dataset):
    def __init__(self, states, labels, weights=None):
        self.states = states
        self.labels = labels
        self.weights = weights
        super().__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.weights is not None:
            return self.states[i], self.labels[i], self.weights[i]

        return self.states[i], self.labels[i]
