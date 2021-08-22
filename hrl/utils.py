import os
import csv
import logging
from pydoc import locate
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print(f"Creation of the directory {path} failed")
    else:
        print(f"Successfully created the directory {path}")
    return path


def load_hyperparams(filepath):
    params = dict()
    with open(filepath, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for name, value, dtype in reader:
            if dtype == 'bool':
                params[name] = bool(strtobool(value))
            else:
                params[name] = locate(dtype)(value)
    return params


def save_hyperparams(filepath, params):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                bool: 'bool',
                int: 'int',
                str: 'str',
                float: 'float',
            })[type(value)] # yapf: disable
            if type_str is not None:
                writer.writerow((name, value, type_str))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def update_param(params, name, value):
    if name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    if type(params[name]) == bool:
        params[name] = bool(strtobool(value))
    else:
        params[name] = type(params[name])(value)


def check_is_antmaze(env_name):
    return env_name in ["antmaze-umaze-v0", "antmaze-medium-play-v0", "antmaze-large-play-v0"]


def check_is_atari(env_name):
    return 'NoFrameskip' in env_name


def every_n_times(n, count, callback, *args, final_count=None):
    if (count % n == 0) or (final_count is not None and (count == final_count)):
        callback(*args)


def augment_state(obs, goal):
    """
    make the state goal-conditioned by concating state with goal
    """
    assert len(obs.shape) == 1
    assert len(goal.shape) == 1
    aug = np.concatenate([obs, goal], axis=-1)
    assert aug.shape[0] == obs.shape[0] + goal.shape[0]
    try:  # make sure dtype if float for forward pass in nnet
        assert aug.dtype == np.float32
    except AssertionError:
        aug = aug.astype(np.float32)
    finally:
        assert aug.dtype == np.float32
    return aug


def make_chunked_value_function_plot(solver, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([exp[0] for exp in replay_buffer])
    actions = np.array([exp[1] for exp in replay_buffer])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for state_chunk, action_chunk in tqdm(zip(state_chunks, action_chunks), desc="Making VF plot"):
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    plt.savefig(f"value_function_plots/{experiment_name}/{file_name}.png")
    plt.close()

    return qvalues.max()
