import os
import numpy as np
from tqdm import tqdm


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
    
    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    state_chunks = get_chunks(states, chunk_size)
    values = np.zeros((len(states),))
    current_idx = 0

    for state_chunk in tqdm(state_chunks, desc="Making VF plot"):
        chunk_values = f(state_chunk)
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values
        current_idx += current_chunk_size

    return values
