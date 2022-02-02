import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_chunked_action_value_function_plot(solver, episode, seed, experiment_name, chunk_size=1000):
    states = np.array([exp[0]['state'] for exp in solver.replay_buffer.memory])
    actions = np.array([exp[0]['action'] for exp in solver.replay_buffer.memory])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for state_chunk, action_chunk in tqdm(zip(state_chunks, action_chunks), desc="Making Q-plot"):
        chunk_qvalues = solver.action_value_function(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    plt.savefig(f"plots/{experiment_name}/{seed}/q_function_episode_{episode}.png")
    plt.close()
    return qvalues.max()


def make_chunked_value_function_plot(solver, episode, seed, experiment_name, chunk_size=1000):
    states = np.array([exp[0]['state'] for exp in solver.replay_buffer.memory])
    actions = np.array([exp[0]['action'] for exp in solver.replay_buffer.memory])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for state_chunk in tqdm(state_chunks, desc="Making VF plot"):
        chunk_qvalues = solver.value_function(state_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_episode_{episode}.png")
    plt.close()

    return qvalues.max()
