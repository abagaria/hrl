import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque
from .init_classifier import InitiationClassifier
from ..datastructures import TrainingExample, StepThresholder


class CriticInitiationClassifier(InitiationClassifier):
    """ Initiation Classifier that thresholds the current critic. """

    def __init__(self, agent, goal_sampler, augment_func, optimistic_threshold=40, pessimistic_threshold=20):
        self.agent = agent  # Actor-critc agent (eg, TD3, SAC, etc)
        self.goal_sampler = goal_sampler  # sample from option termination region
        self.get_augmented_state = augment_func  # to cat state and goal
        
        optimistic_classifier = StepThresholder(optimistic_threshold)
        pessimistic_classifier = StepThresholder(pessimistic_threshold)

        # We need these to sample from init region
        self.positive_examples = deque([], maxlen=100)

        # We don't really need these, but keeping them around for consistency
        self.negative_examples = deque([], maxlen=100)

        super().__init__(optimistic_classifier, pessimistic_classifier)

    def is_initialized(self):  # Always ready to go
        return True

    def value_function(self, states):
        """ Wrapper function to manage VF queries for single and batched states. """
        assert isinstance(states, np.ndarray)
        assert len(states.shape) in (1, 2), states.shape

        def _single_vf(s, g):
            sg = self.get_augmented_state(s, g)
            sg = sg[np.newaxis, ...]  # Add a batch dimension
            return self.agent.get_values(sg)[0]
        
        def _batch_vf(s, g):
            assert s.shape[0] > 1, s.shape
            g = np.repeat(g[np.newaxis, ...], repeats=len(s), axis=0)
            sg = np.concatenate((s, g), axis=1)
            return self.agent.get_values(sg)

        goal = self.goal_sampler()
        if len(states.shape) == 1:
            return _single_vf(states, goal)
        return _batch_vf(states, goal)

    def value2steps(self, value):
        """ Assuming -1 step reward, convert a value prediction to a n_step prediction. """
        def _clip(v):
            if isinstance(v, np.ndarray):
                v[v>0] = 0
                return v
            return v if v <= 0 else 0

        gamma = self.agent.gamma
        clipped_value = _clip(value)
        numerator = np.log(1 + ((1-gamma) * np.abs(clipped_value)))
        denominator = np.log(gamma)
        return np.abs(numerator / denominator)

    def optimistic_predict(self, state):
        value = self.value_function(state)
        steps = self.value2steps(value)
        return self.optimistic_classifier(steps)

    def pessimistic_predict(self, state):
        value = self.value_function(state)
        steps = self.value2steps(value)
        return self.pessimistic_classifier(steps)

    @staticmethod
    def construct_feature_matrix(examples):
        examples = list(itertools.chain.from_iterable(examples))
        observations = [example.obs for example in examples]
        return np.array(observations)

    def add_positive_examples(self, states, positions):
        assert len(states) == len(positions)

        positive_examples = [TrainingExample(img, pos) for img, pos in zip(states, positions)]
        self.positive_examples.append(positive_examples)

    def add_negative_examples(self, states, positions):
        assert len(states) == len(positions)

        negative_examples = [TrainingExample(img, pos) for img, pos in zip(states, positions)]
        self.negative_examples.append(negative_examples)

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
        observations = np.array([eg.obs for eg in trajectory])
        predictions = self.pessimistic_predict(observations)
        if predictions.any(): # grab the first positive obs in traj
            return observations[predictions.squeeze()][0]

    def get_states_inside_pessimistic_classifier_region(self):
        if self.pessimistic_classifier is not None:
            observations = self.construct_feature_matrix(self.positive_examples)
            predictions = self.pessimistic_predict(observations).squeeze()
            positive_observations = observations[predictions==1]
            return positive_observations
        return []

    def plot_initiation_classifier(self, env, replay_buffer, option_name, episode, experiment_name, seed):
        print(f"Plotting Critic Initiation Set Classifier for {option_name}")

        chunk_size = 1000

        # Take out the original goal
        states = [exp[0] for exp in replay_buffer]
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
            chunk_values = self.value_function(state_chunk)
            chunk_steps = self.value2steps(chunk_values).squeeze()
            current_chunk_size = len(state_chunk)

            steps[current_idx:current_idx + current_chunk_size] = chunk_steps
            optimistic_predictions[current_idx:current_idx + current_chunk_size] = self.optimistic_classifier(chunk_steps)
            pessimistic_predictions[current_idx:current_idx + current_chunk_size] = self.pessimistic_classifier(chunk_steps)

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
