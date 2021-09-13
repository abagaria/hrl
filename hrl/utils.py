import os
import csv
import logging
from pydoc import locate
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


class StopExecution:
    """
    the class is used as a token to signify that execution of some env in the VectorEnv should be stopped
    using this class instead of `None` becaus None blocks multiprocessing.Connection.recv()
    """
    pass


def determine_device(disable_gpu=False):
    """
    determine whether execute on CPU or GPU
    """
    if disable_gpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
    logging.info(f'training on device {device}')
    return device


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


def make_done_position_plot(solver, episode, saving_dir, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([exp[0] for exp in replay_buffer])
    dones = np.array([exp[4] for exp in replay_buffer])

    # squeeze into 2-dim arrays, so it's compatible with vector-env states, dones
    states = states.reshape((-1, states.shape[-1]))
    dones = dones.reshape((-1,))

    done_states = states[dones == True]
    plt.scatter(done_states[:, 0], done_states[:, 1])
    file_name = f"done_position_episode_{episode}.png"
    plt.savefig(os.path.join(saving_dir, file_name))
    plt.close()


def make_reward_plot(solver, episode, saving_dir, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([exp[0] for exp in replay_buffer])
    rewards = np.array([exp[2] for exp in replay_buffer])

    # squeeze into 2-dim arrays
    states = states.reshape((-1, states.shape[-1]))
    rewards = rewards.reshape((-1, rewards.shape[-1]))

    plt.scatter(states[:, 0], states[:, 1], c=rewards)
    plt.colorbar()
    file_name = f"reward_episode_{episode}.png"
    plt.savefig(os.path.join(saving_dir, file_name))
    plt.close()


def make_chunked_value_function_plot(solver, episode, saving_dir, goal, chunk_size=1000, replay_buffer=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer
    states = np.array([np.concatenate([exp[0], goal], axis=-1) for exp in replay_buffer])  # goal conditioned
    actions = np.array([exp[1] for exp in replay_buffer])

    # squeeze into 2-dim arrays
    states = states.reshape((-1, states.shape[-1]))
    actions = actions.reshape((-1, actions.shape[-1]))

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
        with torch.no_grad():
            chunk_qvalues = solver.get_qvalues(state_chunk, action_chunk).cpu().numpy().squeeze(-1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()
    file_name = f"value_function_episode_{episode}.png"
    plt.savefig(os.path.join(saving_dir, file_name))
    plt.close()

    return qvalues.max()
