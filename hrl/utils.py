import os
import torch
import itertools
import numpy as np


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def chunked_inference(states, f, chunk_size=1000):
    """" f must take in np arrays and return np arrays. """
    
    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    state_chunks = get_chunks(states, chunk_size)
    values = np.zeros((len(states),))
    current_idx = 0

    for state_chunk in state_chunks:
        chunk_values = f(state_chunk)
        current_chunk_size = len(state_chunk)
        if isinstance(chunk_values, torch.Tensor):
            chunk_values = chunk_values.cpu().numpy()
        values[current_idx:current_idx + current_chunk_size] = chunk_values.squeeze()
        current_idx += current_chunk_size

    return values

def flatten(x):
    return list(itertools.chain.from_iterable(x))
