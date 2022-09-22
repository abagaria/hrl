import os
import ipdb
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from tqdm import tqdm
from hrl.utils import flatten
from sklearn.svm import OneClassSVM, SVC
from .flipping_classifier import FlippingClassifier
from .critic_classifier import CriticInitiationClassifier
from .position_classifier import PositionInitiationClassifier
import warnings
from hrl.agent.dsc.classifier.mlp_classifier import BinaryMLPClassifier

warnings.filterwarnings("ignore")

class DistributionalCriticClassifier(PositionInitiationClassifier):

    def __init__(
        self,
        agent,
        use_position,
        goal_sampler,
        augment_func,
        optimistic_threshold,
        pessimistic_threshold,
        option_name,
        maxlen=100,
        resample_goals=False,
        threshold=None,
        device=None
    ):
        self.agent = agent
        self.use_position = use_position
        self.device = device
        self.goal_sampler = goal_sampler

        self.critic_classifier = CriticInitiationClassifier(
            agent,
            goal_sampler,
            augment_func,
            optimistic_threshold,
            pessimistic_threshold
        )

        self.option_name = option_name
        self.resample_goals = resample_goals
        self.threshold = threshold

        self.optimistic_predict_thresh = 0.4
        self.pessimistic_predict_thresh = 0.6

        super().__init__(maxlen)
        
    @torch.no_grad()
    def get_weights(self, states, labels): 
        '''
        Given state, labels compute the weights to associate to each sample based on its uncertainty. 

        Args:
          states (np.ndarray): num states, state_dim
          labels (np.ndarray): num states, 

        '''
        states = states.to(self.device).to(torch.float32)
        best_actions = self.critic_classifier.agent.actor.get_best_qvalue_and_action(states.to(torch.float32))[1]
        best_actions = best_actions.to(self.device).to(torch.float32)
        value_distribution = self.critic_classifier.agent.actor.forward(states, best_actions).cpu().numpy()

        # how should we determine the proper threshold? Well one way is to look at states where our policy 
        # was able to reach the target position. We can look at those value distributions and take mean of the 85% 
        # percentiles. This seems to be a good strategy. 

        threshold = np.mean(np.quantile(value_distribution[labels == 1], q=0.85, axis=1))

        states_sans_goals = states[:, :-2]

        base_proba_rate_no_weights = BinaryMLPClassifier(
            states_sans_goals.shape[1],
            self.device,
        )

        base_proba_rate_no_weights.fit(states_sans_goals, labels, W=None)
        
        P_Dp = base_proba_rate_no_weights.predict_proba(states_sans_goals).detach().cpu().numpy().squeeze()


        pi_plus_1 = self.unbatched_compute_flipping_prob_value_distribution(value_distribution, threshold, flip_from=1)
        pi_minus_1 = self.unbatched_compute_flipping_prob_value_distribution(value_distribution, threshold, flip_from=-1)
        pi_minus_y = self.unbatched_compute_flipping_prob_value_distribution(value_distribution, threshold, flip_from='-y', labels=labels)
        weights = (((1 - pi_minus_1 - pi_plus_1)*P_Dp)/P_Dp)

        return weights


    '''
    Computes the flipping probability given the value distribution. 

    distributions: passed in distributions [batch_size x 200]
    threshold: float
    flip_from: if +1, then probability of -1 given a +1 prediction ie, P(Y = -1 | Y = +1)
               if -1, then probability of +1 given a -1 prediction ie, P(Y = +1 | Y = -1)
               if '-y', then probability of +1 given a -1 prediction if label is -1 and a -1 given a +1 if label is +1. 
    '''
    def unbatched_compute_flipping_prob_value_distribution(self, distributions, threshold, flip_from, labels=[]):
        if flip_from == '-y' and not (len(labels) == distributions.shape[0]):
            raise Exception("if you want to compute pi_{-y} you must pass in a labels array with same as batch of distributions.")
        proba = []
        for i, distr in enumerate(distributions): 
            index_arr = np.argmax(distr > threshold)
            if flip_from == '-y': 
                if (labels[i] == 1):
                    proba.append((index_arr)*(1/distr.shape[0]))
                elif (labels[i] == -1):
                    proba.append((distr.shape[0]-index_arr)*(1/distr.shape[0]))
            else: 
                if flip_from == 1: 
                    proba.append((index_arr)*(1/distr.shape[0]))
                elif flip_from == -1:
                    proba.append((distr.shape[0]-index_arr)*(1/distr.shape[0]))
        return np.array(proba)

    def _compute_weights_unbatched(self, states, labels, values):
        n_states = states.shape[0]
        weights = np.zeros((n_states,))
        for i in range(n_states):
            label = labels[i]
            state_value = values[i]
            if label == 1:  # These signs are assuming that we are thresholding *steps*, not values.
                flip_mass = state_value[state_value > self.threshold].sum()
            else:
                flip_mass = state_value[state_value < self.threshold].sum()
            weights[i] = flip_mass / state_value.sum()
        return weights

    def _compute_weights_batched(self, states, labels, values):  # TODO
        pass

    def add_positive_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(sg) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_positive_examples(states, infos)
    
    def add_negative_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(s) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_negative_examples(states, infos)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        positions = [example.pos for example in examples]
        return np.array(positions)

    def train_two_class_classifier(self, nu=0.1):
        positive_feature_matrix = self.construct_feature_matrix(self.positive_examples)
        negative_feature_matrix = self.construct_feature_matrix(self.negative_examples)
        positive_labels = [1] * positive_feature_matrix.shape[0]
        negative_labels = [0] * negative_feature_matrix.shape[0]

        X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
        Y = np.concatenate((positive_labels, negative_labels))
        W = self.get_sample_weights(plot=True)

        if negative_feature_matrix.shape[0] >= 10:
            kwargs = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        else:
            kwargs = {"kernel": "rbf", "gamma": "scale"}

        self.optimistic_classifier = SVC(**kwargs)
        self.optimistic_classifier.fit(X, Y, sample_weight=W)

        training_predictions = self.optimistic_classifier.predict(X)
        positive_training_examples = X[training_predictions == 1]

        if positive_training_examples.shape[0] > 0:
            self.pessimistic_classifier = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
            self.pessimistic_classifier.fit(positive_training_examples)

    def get_sample_weights(self, plot=False):

        pos_egs = flatten(self.positive_examples)
        neg_egs = flatten(self.negative_examples)
        examples = pos_egs + neg_egs

        assigned_labels = np.concatenate((
            +1 * np.ones((len(pos_egs),)),
            -1 * np.ones((len(neg_egs),))
        ))

        # Extract what labels the current VF would have assigned
        augmented_states = np.array([eg.info["augmented_state"] for eg in examples])

        if self.resample_goals:
            observations = augmented_states[:, :-2]
            new_goal = self.critic_classifier.goal_sampler()[np.newaxis, ...]
            new_goals = np.repeat(new_goal, axis=0, repeats=observations.shape[0])
            augmented_states = np.concatenate((observations, new_goals), axis=1)

        # Compute the weights based on the probability that the samples will flip
        weights = self.get_weights(torch.from_numpy(augmented_states), assigned_labels)

        if plot:
            # ipdb.set_trace()
            x = [eg.info["player_x"] for eg in examples]
            y = [eg.info["player_y"] for eg in examples]
            c = assigned_labels.tolist()
            s = (1. * weights).tolist()
            plt.subplot(1, 2, 1)
            plt.scatter(x[:len(pos_egs)], y[:len(pos_egs)], c=s[:len(pos_egs)])
            plt.colorbar()
            plt.clim((0, 10))
            plt.subplot(1, 2, 2)
            plt.scatter(x[len(pos_egs):], y[len(pos_egs):], c=s[len(pos_egs):])
            plt.colorbar()
            plt.clim((0, 10))
            plt.savefig(f"results/weight_plots_{self.option_name}.png")
            plt.close()

        return weights

    @staticmethod
    def value2steps(value):
        """ Assuming -1 step reward, convert a value prediction to a n_step prediction. """
        def _clip(v):
            if isinstance(v, np.ndarray):
                v[v>0] = 0
                return v
            return v if v <= 0 else 0

        gamma = .99
        clipped_value = _clip(value)
        numerator = np.log(1 + ((1-gamma) * np.abs(clipped_value)))
        denominator = np.log(gamma)
        return np.abs(numerator / denominator)

    def plot_initiation_classifier(self, goal, option_name, episode, experiment_name, seed):
        print(f"Plotting Critic Initiation Set Classifier for {option_name}")

        chunk_size = 1000
        replay_buffer = self.agent.actor.buffer_object.storage

        # Take out the original goal
        trans = replay_buffer.get_all_transitions()
        states = trans['obs']
        states = [state[:-2] for state in states]

        if len(states) > 100_000:
            print(f"Subsampling {len(states)} s-a pairs to 100,000")
            idx = np.random.randint(0, len(states), size=100_000)
            states = [states[i] for i in idx]

        print(f"preparing {len(states)} states")
        states = np.array(states)

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))

        if num_chunks == 0:
            return 0.

        print("chunking")
        state_chunks = np.array_split(states, num_chunks, axis=0)
        steps = np.zeros((states.shape[0],))
        
        optimistic_predictions = np.zeros((states.shape[0],))
        pessimistic_predictions = np.zeros((states.shape[0],))

        current_idx = 0

        for state_chunk in tqdm(state_chunks, desc="Plotting Critic Init Classifier"):
            goal = np.repeat([goal], repeats=len(state_chunk), axis=0)
            state_chunk = state_chunk[:, :2]
            current_chunk_size = len(state_chunk)

            optimistic_predictions[current_idx:current_idx + current_chunk_size] = self.optimistic_classifier.predict(state_chunk)
            pessimistic_predictions[current_idx:current_idx + current_chunk_size] = self.pessimistic_classifier.predict(state_chunk)

            current_idx += current_chunk_size

        print("plotting")
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 3, 1)
        plt.scatter(states[:, 0], states[:, 1], c=steps)
        plt.title(f"nSteps to termination region")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.scatter(states[:, 0], states[:, 1], c=optimistic_predictions)
        plt.title(f"Optimistic Classifier")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.scatter(states[:, 0], states[:, 1], c=pessimistic_predictions)
        plt.title(f"Pessimistic Classifier")
        plt.colorbar()

        plt.suptitle(f"{option_name}")
        file_name = f"{option_name}_critic_init_clf_{seed}_episode_{episode}"
        saving_path = os.path.join('results', experiment_name, 'initiation_set_plots', f'{file_name}.png')

        print("saving")
        plt.savefig(saving_path)
        plt.close()
