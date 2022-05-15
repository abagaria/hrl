import os
import ipdb
import scipy
import random
import numpy as np
import matplotlib.pyplot as plt

from hrl.utils import flatten
from sklearn.svm import OneClassSVM, SVC
from .flipping_classifier import FlippingClassifier
from .critic_classifier import CriticInitiationClassifier
from .position_classifier import PositionInitiationClassifier


class CriticBayesClassifier(PositionInitiationClassifier):
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
        resample_goals=False
    ):
        self.agent = agent
        self.use_position = use_position

        self.critic_classifier = CriticInitiationClassifier(
            agent,
            goal_sampler,
            augment_func,
            optimistic_threshold,
            pessimistic_threshold
        )

        self.flipping_classifier = FlippingClassifier(
            classifier_type="svm",
            feature_extractor_type="pos",
        )

        self.option_name = option_name
        self.resample_goals = resample_goals

        super().__init__(maxlen)

    def add_positive_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(sg) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_positive_examples(states, infos)
    
    def add_negative_examples(self, states, infos):
        assert all(["value" in info for info in infos]), "need V(s) for weights"
        assert all(["augmented_state" in info for info in infos]), "need sg to recompute V(sg)"
        return super().add_negative_examples(states, infos)

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
            np.ones((len(pos_egs),)),
            np.zeros((len(neg_egs),))
        ))

        # Extract what labels the old VF *would* have assigned
        old_values = np.array([eg.info["value"] for eg in examples]).squeeze()
        old_nsteps = self.critic_classifier.value2steps(old_values)
        old_critic_labels = self.critic_classifier.pessimistic_classifier(old_nsteps)

        # Extract what labels the current VF would have assigned
        augmented_states = np.array([eg.info["augmented_state"] for eg in examples])

        if self.resample_goals:
            observations = augmented_states[:, :-2]
            new_goal = self.critic_classifier.goal_sampler()[np.newaxis, ...]
            new_goals = np.repeat(new_goal, axis=0, repeats=observations.shape[0])
            augmented_states = np.concatenate((observations, new_goals), axis=1)

        new_values = self.agent.get_values(augmented_states).squeeze()
        new_nsteps = self.critic_classifier.value2steps(new_values)
        new_critic_labels = self.critic_classifier.optimistic_classifier(new_nsteps)

        # Train the flip predictor
        self.flipping_classifier.fit(
            examples,
            assigned_labels,
            old_critic_labels,
            new_critic_labels
        )
        
        # Predict the probability that the samples will flip
        probabilities = self.flipping_classifier(examples)
        weights = 1. / (probabilities + 1e-4)

        if plot:
            self.plot_flipping_labels(
                examples,
                assigned_labels, 
                old_critic_labels, 
                new_critic_labels
            )

            self.plot_flipping_probabilities(
                examples,
                assigned_labels,
                old_critic_labels,
                new_critic_labels,
                probabilities
            )

        return weights

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        def make_meshgrid(x, y, h=.02):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            return xx, yy

        def get_grid_states(mdp):
            ss = []
            x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
            x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
            for x in np.arange(x_low_lim, x_high_lim+1, 1):
                for y in np.arange(y_low_lim, y_high_lim+1, 1):
                    ss.append(np.array((x, y)))
            return ss

        def get_initiation_set_values(mdp):
            values = []
            x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
            x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
            for x in np.arange(x_low_lim, x_high_lim+1, 1):
                for y in np.arange(y_low_lim, y_high_lim+1, 1):
                    pos = np.array((x, y))
                    init = self.optimistic_predict(pos)
                    if hasattr(mdp.env, 'env'):
                        init = init and not mdp.env.env._is_in_collision(pos)
                    values.append(init)
            return values

        def plot_one_class_initiation_classifier():
            colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

            X = self.construct_feature_matrix(self.positive_examples)
            X0, X1 = X[:, 0], X[:, 1]
            xx, yy = make_meshgrid(X0, X1)
            Z1 = self.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z1 = Z1.reshape(xx.shape)

            color = random.choice(colors)
            plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

        def plot_two_class_classifier(mdp, episode, experiment_name, plot_examples=True, seed=0):
            states = get_grid_states(mdp)
            values = get_initiation_set_values(mdp)

            x = np.array([state[0] for state in states])
            y = np.array([state[1] for state in states])
            xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
            xx, yy = np.meshgrid(xi, yi)
            rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
            zz = rbf(xx, yy)
            plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
            plt.colorbar()

            # Plot trajectories
            positive_examples = self.construct_feature_matrix(self.positive_examples)
            negative_examples = self.construct_feature_matrix(self.negative_examples)
            
            weights = self.get_sample_weights()
            positive_weights = weights[:len(positive_examples)]
            negative_weights = weights[len(positive_examples):]

            if positive_examples.shape[0] > 0 and plot_examples:
                plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=positive_weights)

            if negative_examples.shape[0] > 0 and plot_examples:
                plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=negative_weights)

            if self.pessimistic_classifier is not None:
                plot_one_class_initiation_classifier()

            # background_image = imageio.imread("four_room_domain.png")
            # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

            name = option_name if episode is None else option_name + f"_{experiment_name}_{episode}"
            plt.title(f"{option_name} Initiation Set")
            saving_path = os.path.join('results', experiment_name, 'initiation_set_plots', f'{name}_initiation_classifier_{seed}.png')
            plt.savefig(saving_path)
            plt.close()

        plot_two_class_classifier(env, episode, experiment_name, True, seed)

    def plot_flipping_labels(self, examples, assigned_labels, old_vf_labels, new_vf_labels):
        x_positions = np.array([eg.pos[0] for eg in examples])
        y_positions = np.array([eg.pos[1] for eg in examples])

        pos_x_positions = x_positions[assigned_labels==1]
        pos_y_positions = y_positions[assigned_labels==1]

        neg_x_positions = x_positions[assigned_labels==0]
        neg_y_positions = y_positions[assigned_labels==0]

        flip_labels = self.flipping_classifier.extract_labels(
            assigned_labels,
            old_vf_labels,
            new_vf_labels
        )

        pos_flip_labels = flip_labels[assigned_labels==1]
        neg_flip_labels = flip_labels[assigned_labels==0]

        plt.subplot(1, 2, 1)
        plt.scatter(pos_x_positions, pos_y_positions, c=pos_flip_labels, s=20, cmap="Set1")
        plt.colorbar()
        plt.title("Originally Positive")

        plt.subplot(1, 2, 2)
        plt.scatter(neg_x_positions, neg_y_positions, c=neg_flip_labels, s=20, cmap="Set1")
        plt.colorbar()
        plt.title("Orginally Negative")

        plt.suptitle(f"{self.option_name} Flip Labels")
        plt.savefig(f"results/critic_clf_resample_goals/initiation_set_plots/{self.option_name}_flip_labels.png")
        plt.close()
    
    def plot_flipping_probabilities(self, examples, assigned_labels, old_vf_labels, new_vf_labels, probs):
        x_positions = np.array([eg.pos[0] for eg in examples])
        y_positions = np.array([eg.pos[1] for eg in examples])

        pos_x_positions = x_positions[assigned_labels==1]
        pos_y_positions = y_positions[assigned_labels==1]

        neg_x_positions = x_positions[assigned_labels==0]
        neg_y_positions = y_positions[assigned_labels==0]

        pos_old_vf_labels = old_vf_labels[assigned_labels==1]
        neg_old_vf_labels = old_vf_labels[assigned_labels==0]

        pos_new_vf_labels = new_vf_labels[assigned_labels==1]
        neg_new_vf_labels = new_vf_labels[assigned_labels==0]

        pos_probs = probs[assigned_labels==1]
        neg_probs = probs[assigned_labels==0]

        plt.figure(figsize=(16, 10))

        # Plot old vf labels
        plt.subplot(1, 3, 1)
        plt.scatter(pos_x_positions, pos_y_positions, c=pos_old_vf_labels, marker="+", s=20)
        plt.scatter(neg_x_positions, neg_y_positions, c=neg_old_vf_labels, marker="o", s=20)
        plt.colorbar()
        plt.title("Old VF labels")

        # Plot new vf labels
        plt.subplot(1, 3, 2)
        plt.scatter(pos_x_positions, pos_y_positions, c=pos_new_vf_labels, marker="+", s=20)
        plt.scatter(neg_x_positions, neg_y_positions, c=neg_new_vf_labels, marker="o", s=20)
        plt.colorbar()
        plt.title("New VF Labels")

        # Plot flipping probs
        plt.subplot(1, 3, 3)
        plt.scatter(pos_x_positions, pos_y_positions, c=pos_probs, marker="+", s=20); plt.colorbar()
        plt.scatter(neg_x_positions, neg_y_positions, c=neg_probs, marker="o", s=20); plt.colorbar()

        plt.title(f"Flipping Probs")
        plt.savefig(f"results/critic_clf_resample_goals/initiation_set_plots/{self.option_name}_flipper.png")
        plt.close()
