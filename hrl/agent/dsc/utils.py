import os
import gzip
import torch
import scipy
import pickle
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple
from hrl.utils import chunked_inference
from pfrl.wrappers import atari_wrappers
from hrl.agent.rainbow.rainbow import Rainbow


def info_to_pos(info):
    if isinstance(info, np.ndarray):
        return info
    pos = info['player_x'], info['player_y']
    return np.array(pos)

def pos_to_info(pos):
    if isinstance(pos, dict):
        return pos
    return dict(player_x=pos[0], player_y=pos[1])

def default_pos_to_info(pos):
    return dict(
        player_x=pos[0],
        player_y=pos[1],
        has_key=False,
        room_number=1,
        dead=False,
        falling=False
    )

def flatten(lol):
    return list(
        itertools.chain.from_iterable(lol)
    )

def make_meshgrid(x, y, h=1.):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def get_grid_states(low, high, res):
    ss = []
    for x in np.arange(low[0], high[0]+res, res):
        for y in np.arange(low[1], high[1]+res, res):
            pos = np.array((x, y))
            ss.append(pos)
    return ss


def get_initiation_set_values(option, low, high, res):
    values = []
    for x in np.arange(low[0], high[0]+res, res):
        for y in np.arange(low[1], high[1]+res, res):
            pos = np.array((x, y))
            init = option.is_init_true(
                pos, 
                default_pos_to_info(pos)
            )
            values.append(init)
    return values

def plot_one_class_initiation_classifier(option):

    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    X = option.initiation_classifier.construct_feature_matrix(option.initiation_classifier.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.initiation_classifier.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

def plot_two_class_classifier(option, episode, experiment_name, plot_examples=True, seed=0):
    low = 0, 140
    high = 150, 250
    states = get_grid_states(low, high, res=10)
    values = get_initiation_set_values(option, low, high, res=10)

    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
    plt.colorbar()

    # Plot trajectories
    positive_examples = option.initiation_classifier.construct_feature_matrix(option.initiation_classifier.positive_examples)
    negative_examples = option.initiation_classifier.construct_feature_matrix(option.initiation_classifier.negative_examples)

    if positive_examples.shape[0] > 0 and plot_examples:
        plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=10)

    if negative_examples.shape[0] > 0 and plot_examples:
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=10)

    if option.initiation_classifier.pessimistic_classifier is not None:
        plot_one_class_initiation_classifier(option)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    plt.title(f"{option.name} targeting {option.target_salient_event}")
    plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option.name}_{episode}_initiation_classifier.png")
    plt.close()


def make_chunked_goal_conditioned_value_function_plot(solver, options,
                                                      goal_salient_event, episode, seed,
                                                      experiment_name, chunk_size=1000):
    assert isinstance(solver, Rainbow)

    goal = goal_salient_event.target_obs
    goal_pos = goal_salient_event.target_pos

    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    def _get_states2():
        training_egs = []
        for option in options:
            pos_egs = list(itertools.chain.from_iterable(option.initiation_classifier.positive_examples))
            neg_egs = list(itertools.chain.from_iterable(option.initiation_classifier.negative_examples))
            training_egs.extend(pos_egs + neg_egs + list(option.effect_set))
        states = [eg.obs for eg in training_egs]
        positions = np.array([eg.pos for eg in training_egs])
        return states, positions

    def cat(s, g):
        assert isinstance(s, atari_wrappers.LazyFrames)
        g = g._frames[-1] if isinstance(g, atari_wrappers.LazyFrames) else g
        return atari_wrappers.LazyFrames(list(s._frames)[:4] + [g], stack_axis=0)

    def _get_gc_states():
        states, positions = _get_states2()
        return [cat(s, goal) for s in states], positions

    # Take out the original goal and append the new goal
    states, positions = _get_gc_states()

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(len(states) / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = get_chunks(states, num_chunks)
    values = torch.zeros((len(states),)).to(solver.device)
    current_idx = 0

    for state_chunk in tqdm(state_chunks, desc="Making VF plot"):
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = solver.value_function(state_chunk)
        current_idx += current_chunk_size

    plt.scatter(positions[:, 0], positions[:, 1], c=values.cpu().numpy())
    plt.colorbar()
    plt.title(f"VF Targeting {np.round(goal_pos, 2)}")
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_plots/vf_goal_{goal_pos}_episode_{episode}.png")
    plt.close()

    return values.max()


MonteRAMState = namedtuple("MonteRAMState", ["player_x", "player_y", "screen", "has_key", "door_left_locked", "door_right_locked", "skull_x", "lives"])

def get_byte(ram: np.ndarray, address: int) -> int:
    """Return the byte at the specified emulator RAM location"""
    assert isinstance(ram, np.ndarray) and ram.dtype == np.uint8 and isinstance(address, int), ram
    return int(ram[address & 0x7f])

def parse_ram(ram: np.ndarray) -> MonteRAMState:
    """Get the current annotated Montezuma RAM state as a tuple

    See RAM annotations:
    https://docs.google.com/spreadsheets/d/1KU4KcPqUuhSZJ1N2IyhPW59yxsc4mI4oSiHDWA3HCK4
    """
    x = get_byte(ram, 0xaa)
    y = get_byte(ram, 0xab)
    screen = get_byte(ram, 0x83)

    inventory = get_byte(ram, 0xc1)
    key_mask = 0b00000010
    has_key = bool(inventory & key_mask)

    objects = get_byte(ram, 0xc2)
    door_left_locked  = bool(objects & 0b1000)
    door_right_locked = bool(objects & 0b0100)

    skull_offset = 33
    skull_x = get_byte(ram, 0xaf) + skull_offset
    #skull_x = 0
    lives = get_byte(ram, 0xba)

    return MonteRAMState(x, y, screen, has_key, door_left_locked, door_right_locked, skull_x, lives)


def load_saved_trajs(data_path, skip=0):
    '''
    Returns a generator for getting states.

    Args:
        skip (int): number of trajectories to skip

    Returns:
        (generator): generator to be called for trajectories
    '''
    print(f"[+] Loading trajectories from file '{data_path}'")
    with gzip.open(data_path, 'rb') as f:
        print(f"[+] Skipping {skip} trajectories...")
        for _ in tqdm(range(skip)):
            traj = pickle.load(f)

        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass


def get_saved_trajectories(data_path, n_trajectories):
    traj_generator = load_saved_trajs(data_path)

    raw_ram_trajs, ram_trajs, frame_trajs = [], [], []

    for i in range(n_trajectories):
        traj = next(traj_generator)

        raw_ram_traj, ram_traj, frame_traj = [], [], []
        for ram, frame in traj:
            raw_ram_traj.append(ram)
            ram_traj.append(parse_ram(ram))
            frame_traj.append(frame)
        raw_ram_trajs.append(raw_ram_traj)
        ram_trajs.append(ram_traj)
        frame_trajs.append(frame_traj)
    return raw_ram_trajs, ram_trajs, frame_trajs


def plot_classifier_predictions(option, states, rams, episode, seed, experiment_name):
   x_positions = [ram.player_x for ram in rams]
   y_positions = [ram.player_y for ram in rams]

   states = np.array(states).reshape(-1, 1, 84, 84)

   classifier = option.initiation_classifier
   f1 = lambda x: classifier.batched_optimistic_predict(x).squeeze()
   f2 = lambda x: classifier.batched_pessimistic_predict(x).squeeze()

   optimistic_predictions = chunked_inference(states, f1)
   pessimistic_predictions = chunked_inference(states, f2)
   
   plt.figure(figsize=(16, 10))

   plt.subplot(1, 2, 1)
   plt.scatter(x_positions, y_positions, c=optimistic_predictions)
   plt.colorbar()
   plt.title(f"Optimistic Predictions")

   plt.subplot(1, 2, 2)
   plt.scatter(x_positions, y_positions, c=pessimistic_predictions)
   plt.colorbar()
   plt.title("Pesssimistic Predictions")

   plt.suptitle(f"Subgoal: {option.target_salient_event}")
   plt.savefig(f"plots/{experiment_name}/{seed}/initiation_set_plots/{option}_init_clf_episode_{episode}.png")
   plt.close()
