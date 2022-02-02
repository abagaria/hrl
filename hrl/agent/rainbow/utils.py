from math import exp
import pfrl
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from .rainbow import Rainbow
from pfrl.wrappers import atari_wrappers


def make_chunked_value_function_plot(pfrl_agent, episode, seed, experiment_name, chunk_size=1000):
    assert isinstance(pfrl_agent, Rainbow)

    def get_states(rbuf):
        states = []
        infos = []
        for n_transitions in rbuf.memory.data:
            for transition in n_transitions:
                states.append(transition['next_state'])
                infos.append(transition["info"])
        return states, infos

    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    states, infos = get_states(pfrl_agent.rbuf)
    state_chunks = get_chunks(states, chunk_size)
    values = np.zeros((len(states),))
    current_idx = 0

    for state_chunk in tqdm(state_chunks, desc="Making VF plot"):
        chunk_values = pfrl_agent.value_function(state_chunk).cpu().numpy()
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values
        current_idx += current_chunk_size

    x = [info['player_x'] for info in infos]
    y = [info['player_y'] for info in infos]

    plt.scatter(x, y, c=values)
    plt.colorbar()
    file_name = f"rainbow_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"plots/{experiment_name}/{seed}/{file_name}.png")
    plt.close()

    return values.max()


def make_chunked_gc_value_function_plot(pfrl_agent, goal, goal_pos, episode, seed, experiment_name, chunk_size=1000):

    def get_states(rbuf):
        states = []
        infos = []
        for n_transitions in rbuf.memory.data:
            for transition in n_transitions:
                states.append(transition['next_state'])
                infos.append(transition["info"])
        return states, infos

    def augmenent_states(s, g):
        assert isinstance(g, (np.ndarray, atari_wrappers.LazyFrames)), type(g)
        g = g._frames[-1] if isinstance(g, atari_wrappers.LazyFrames) else g
        augmented_states = [atari_wrappers.LazyFrames(list(ss._frames[:-1])+[g], stack_axis=0) for ss in s]
        return augmented_states

    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    states, infos = get_states(pfrl_agent.rbuf)
    states = augmenent_states(states, goal)
    state_chunks = get_chunks(states, chunk_size)
    values = np.zeros((len(states),))
    current_idx = 0

    for state_chunk in tqdm(state_chunks, desc="Making VF plot"):
        chunk_values = pfrl_agent.value_function(state_chunk).cpu().numpy()
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values
        current_idx += current_chunk_size

    x = [info['player_x'] for info in infos]
    y = [info['player_y'] for info in infos]

    plt.scatter(x, y, c=values)
    plt.colorbar()
    plt.title(f"VF targeting {goal_pos}")
    file_name = f"vf_goal_{goal_pos}_seed_{seed}_episode_{episode}"
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_plots/{file_name}.png")
    plt.close()

    return values.max()


def plot_goals_in_replay_buffer(pfrl_agent, experiment_name, seed):
    def get_positive_states(rbuf):
        states = []
        infos = []
        for n_transitions in rbuf.memory.data:
            for transition in n_transitions:
                if transition['reward'] > 0:
                    assert transition['is_state_terminal']==1, transition['is_state_terminal']
                    states.append(transition['next_state'])
                    infos.append(transition['info'])
        return states, infos
    
    states, infos = get_positive_states(pfrl_agent.rbuf)
    idx = random.sample(range(len(states)), k=min(len(states), 1000))
    sampled_states = [states[i] for i in idx]

    for i, state in enumerate(sampled_states):
        img = Image.fromarray(state._frames[-1].squeeze())
        img.save(f"plots/{experiment_name}/{seed}/sampled_goals/{i}.png")

