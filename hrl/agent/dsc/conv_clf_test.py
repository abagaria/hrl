import os
import gzip
import random
import pickle
import argparse
from collections import deque

import numpy as np

from hrl.utils import create_log_dir
from hrl.agent.dsc.classifier.conv_classifier import FixedConvInitiationClassifier


def load_trajectories(path, skip=0):
    '''
    Returns a generator for getting states.
    Args:
        path (str): filepath of pkl file containing trajectories
        skip (int): number of trajectories to skip
    Returns:
        (generator): generator to be called for trajectories
    '''
    print(f"[+] Loading trajectories from file '{path}'")
    with gzip.open(path, 'rb') as f:
        for _ in range(skip):
            traj = pickle.load(f)

        try:
            while True:
                traj = pickle.load(f)
                yield traj
        except EOFError:
            pass


def _getIndex(address):
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row * 16 + col
def getByte(ram, address):
    # Return the byte at the specified emulator RAM location
    idx = _getIndex(address)
    return ram[idx]

def get_player_position(ram):
    """
    given the ram state, get the position of the player
    """
    # return the player position at a particular state
    x = int(getByte(ram, 'aa'))
    y = int(getByte(ram, 'ab'))
    return x, y


def get_skull_position(ram):
    """
    given the ram state, get the position of the skull
    """
    x = int(getByte(ram, 'af'))
    return x


def load_training_and_testing_data(goal_position=(133, 148), select_training_points=lambda x, y: x >= 90):
    if os.path.exists("logs/trajectories/training_data.pkl"):
        with open("logs/trajectories/training_data.pkl", "rb") as f:
            positive_training, positive_testing, negative_training, negative_testing = pickle.load(f)
        return positive_training, positive_testing, negative_training, negative_testing  
  
    import random
    traj_generator = load_trajectories("logs/trajectories/monte_rnd_full_trajectories.pkl.gz", skip=0)
    seen_pos = set()
    positive_training = deque(maxlen=80)
    negative_training = deque(maxlen=400)
    positive_testing = deque(maxlen=80)
    negative_testing = deque(maxlen=400)
    for traj in traj_generator:
        if len(positive_training) >= 80 and len(positive_testing) >= 80:
            break
        for state in traj:
            ram, obs = state
            pos = get_player_position(ram)
            if 0 <= np.linalg.norm(np.array(pos) - np.array(goal_position)) < 30:
                # a positive example
                if pos not in seen_pos:
                    if select_training_points(*pos):
                        if random.random() < 0.5:
                            positive_training.append((obs, pos))
                        else:
                            positive_testing.append((obs, pos))
                    else:
                        positive_testing.append((obs, pos))
                    seen_pos.add(pos)
            else:
                # a negative example
                if pos not in seen_pos:
                    if select_training_points(*pos):
                        if random.random() < 0.5:
                            negative_training.append((obs, pos))
                        else:
                            negative_testing.append((obs, pos))
                    else:
                        negative_testing.append((obs, pos))
                    seen_pos.add(pos)
    print(len(positive_training), len(positive_testing), len(negative_training), len(negative_testing))

    with open("logs/trajectories/training_data.pkl", "wb") as f:
        pickle.dump((positive_training, positive_testing, negative_training, negative_testing), f)

    return positive_training, positive_testing, negative_training, negative_testing


def load_data_from_same_position(position=(133, 148)):
    if os.path.exists("logs/trajectories/same_pos_data.pkl"):
        with open("logs/trajectories/same_pos_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    traj_generator = load_trajectories("logs/trajectories/monte_rnd_full_trajectories.pkl.gz", skip=0)
    data = deque(maxlen=200)
    largest_skull_pos = 0
    for traj in traj_generator:
        if len(data) >= 200 and largest_skull_pos >= 0:
            break
        for state in traj:
            ram, obs = state
            pos = get_player_position(ram)
            skull_pos = get_skull_position(ram)
            largest_skull_pos = max(largest_skull_pos, skull_pos)
            if np.array_equal(pos, position):
                data.append((obs, (skull_pos, 148)))
    print(f"Loaded {len(data)} data points from position {position}")

    with open("logs/trajectories/same_pos_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return data


def test_classifier(positive_train, positive_test, negative_train, negative_test, 
                    same_pos_data, plot_dir, gamma, nu):
    clf = FixedConvInitiationClassifier(device='cuda:0', gamma=gamma, nu=nu)
    clf.add_positive_examples(*zip(*positive_train))
    clf.add_negative_examples(*zip(*negative_train))
    clf.fit_initiation_classifier()

    # training plot
    plot_path = os.path.join(plot_dir, f"classifier_train.png")
    clf.plot_training_predictions(plot_path)

    # testing plot
    clf.positive_examples = []
    clf.negative_examples = []
    clf.add_positive_examples(*zip(*positive_test))
    clf.add_negative_examples(*zip(*negative_test))
    plot_path = os.path.join(plot_dir, f"classifier_test.png")
    clf.plot_training_predictions(plot_path)

    # same position plot
    clf.positive_examples = []
    clf.negative_examples = [[clf.negative_examples[0][0]]]
    assert len(clf.negative_examples) == 1
    clf.add_positive_examples(*zip(*same_pos_data))
    plot_path = os.path.join(plot_dir, f"classifier_same_pos.png")
    clf.plot_training_predictions(plot_path)


if __name__ == "__main__":
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.075)
    args = parser.parse_args()

    plot_dir = f"plots/gamma_{args.gamma}_nu_{args.nu}"
    create_log_dir("plots")
    create_log_dir(plot_dir)

    positive_train, positive_test, negative_train, negative_test = load_training_and_testing_data()
    assert len(positive_train) > 0
    assert len(negative_train) > 0
    assert len(positive_test) > 0
    assert len(negative_test) > 0
    same_pos_data = load_data_from_same_position()
    assert len(same_pos_data) > 0
    test_classifier(positive_train, positive_test, negative_train, negative_test, 
                    same_pos_data, plot_dir=plot_dir, gamma=args.gamma, nu=args.nu)
