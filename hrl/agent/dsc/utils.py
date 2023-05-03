import os

import torch
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from treelib import Tree, Node
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer
from hrl.utils import chunked_inference


class SkillTree(object):
    def __init__(self, options):
        self._tree = Tree()
        self.options = options

        if len(options) > 0:
            [self.add_node(option) for option in options]

    def add_node(self, option):
        if option.name not in self._tree:
            print(f"Adding {option} to the skill-tree")
            self.options.append(option)
            parent = option.parent.name if option.parent is not None else None
            self._tree.create_node(
                tag=option.name, identifier=option.name, data=option, parent=parent)

    def get_option(self, option_name):
        if option_name in self._tree.nodes:
            node = self._tree.nodes[option_name]
            return node.data

    def get_depth(self, option):
        return self._tree.depth(option.name)

    def get_children(self, option):
        return self._tree.children(option.name)

    def traverse(self):
        """ Breadth first search traversal of the skill-tree. """
        return list(self._tree.expand_tree(mode=self._tree.WIDTH))

    def show(self):
        """ Visualize the graph by printing it to the terminal. """
        self._tree.show()


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


def get_initiation_set_values(option):
    values = []
    x_low_lim, y_low_lim = option.overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = option.overall_mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            pos = np.array((x, y))
            init = option.is_init_true(pos)
            if hasattr(option.overall_mdp.env, 'env'):
                init = init and not option.overall_mdp.env.env._is_in_collision(
                    pos)
            values.append(init)
    return values


def plot_one_class_initiation_classifier(option):

    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    X = option.initiation_classifier.construct_feature_matrix(
        option.initiation_classifier.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.initiation_classifier.pessimistic_classifier.decision_function(
        np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])


def plot_two_class_classifier(option, episode, experiment_name, plot_examples=True, seed=0):
    states = get_grid_states(option.overall_mdp)
    values = get_initiation_set_values(option)

    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(
        y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(
    ), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
    plt.colorbar()

    # Plot trajectories
    positive_examples = option.initiation_classifier.construct_feature_matrix(
        option.initiation_classifier.positive_examples)
    negative_examples = option.initiation_classifier.construct_feature_matrix(
        option.initiation_classifier.negative_examples)

    if positive_examples.shape[0] > 0 and plot_examples:
        plt.scatter(positive_examples[:, 0], positive_examples[:, 1],
                    label="positive", c="black", alpha=0.3, s=10)

    if negative_examples.shape[0] > 0 and plot_examples:
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1],
                    label="negative", c="lime", alpha=1.0, s=10)

    if option.initiation_classifier.pessimistic_classifier is not None:
        plot_one_class_initiation_classifier(option)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    name = option.name if episode is None else option.name + \
        f"_{experiment_name}_{episode}"
    plt.title(f"{option.name} Initiation Set")
    saving_path = os.path.join(
        'results', experiment_name, 'initiation_set_plots', f'{name}_initiation_classifier_{seed}.png')
    plt.savefig(saving_path)
    plt.close()


def plot_initiation_distribution(option, mdp, episode, experiment_name, chunk_size=10000):
    assert option.initiation_distribution is not None
    data = mdp.dataset[:, :2]

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))
    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(data, num_chunks, axis=0)
    pvalues = np.zeros((data.shape[0],))
    current_idx = 0

    for chunk_number, state_chunk in tqdm(enumerate(state_chunks)):
        probabilities = np.exp(
            option.initiation_distribution.score_samples(state_chunk))
        pvalues[current_idx:current_idx + len(state_chunk)] = probabilities
        current_idx += len(state_chunk)

    plt.scatter(data[:, 0], data[:, 1], c=pvalues)
    plt.colorbar()
    plt.title("Density Estimator Fitted on Pessimistic Classifier")
    saving_path = os.path.join('results', experiment_name, 'initiation_set_plots',
                               f'{option.name}_initiation_distribution_{episode}.png')
    plt.savefig(saving_path)
    plt.close()


def make_chunked_goal_conditioned_value_function_plot(solver, goal, episode, seed, experiment_name, chunk_size=1000, replay_buffer=None, option_idx=None):
    replay_buffer = replay_buffer if replay_buffer is not None else solver.replay_buffer

    goal = goal[:2]  # Extracting the position from the goal vector

    # Take out the original goal and append the new goal
    states = [exp[0] for exp in replay_buffer]
    states = [state[:-2] for state in states]
    actions = [exp[1] for exp in replay_buffer]

    if len(states) > 100_000:
        print(f"Subsampling {len(states)} s-a pairs to 100,000")
        idx = np.random.randint(0, len(states), size=100_000)
        states = [states[i] for i in idx]
        actions = [actions[i] for i in idx]

    print(f"preparing {len(states)} states")
    states = np.array([np.concatenate((state, goal), axis=0)
                      for state in states])

    print(f"preparing {len(actions)} actions")
    actions = np.array(actions)

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    print("chunking")
    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for chunk_number, (state_chunk, action_chunk) in tqdm(enumerate(zip(state_chunks, action_chunks)), desc="Making VF plot"):  # type: (int, np.ndarray)
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        chunk_qvalues = solver.get_qvalues(
            state_chunk, action_chunk).cpu().numpy().squeeze(1)
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_qvalues
        current_idx += current_chunk_size

    print("plotting")
    plt.scatter(states[:, 0], states[:, 1], c=qvalues)
    plt.colorbar()

    if option_idx is None:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}"
    else:
        file_name = f"{solver.name}_value_function_seed_{seed}_episode_{episode}_option_{option_idx}"
    plt.title(f"VF Targeting {np.round(goal, 2)}")
    saving_path = os.path.join(
        'results', experiment_name, 'value_function_plots', f'{file_name}.png')

    print("saving")
    plt.savefig(saving_path)
    plt.close()

    return qvalues.max()


def parse_replay(replay, n_goal_dims=2):
    states = []
    memory = replay.memory.data if isinstance(
        replay, PrioritizedReplayBuffer) else replay.memory
    for transition in memory:
        transition = transition[-1]  # n-step to single transition
        states.append(transition['next_state'][:-n_goal_dims])
    return states


def visualize_initiation_gvf(
    initiation_gvf,
    target_policy,
    goal,
    episode,
    experiment_name,
    seed
):
    assert goal.shape == (2,), goal.shape
    replay = initiation_gvf.initiation_replay_buffer
    states = parse_replay(replay)
    states = np.array([np.concatenate((state, goal), axis=0)
                      for state in states])

    @torch.no_grad()
    def pi(states):
        action_tensor = target_policy(states)
        return action_tensor.cpu().numpy()

    '''
    GVF values.
    '''
    def f(x): return initiation_gvf.policy_evaluation_module.get_values(
        x, target_policy)
    '''
    The difference between the q1 and q1 values of the online network in the GVF. 
    '''
    def g(x): return initiation_gvf.get_value_and_uncertainty(x)[1]
    '''
    The difference between the online and target GVF
    '''
    def x(x): return initiation_gvf.get_value_and_uncertainty(x)[2]
    '''
    Return a pseudo count based uncertainty metric
    '''
    def h(x):
        pos_x_arr = np.array(x[:, 0]/0.6).astype(int)
        pos_y_arr = np.array(x[:, 1]/0.6).astype(int)
        goal_x_arr = np.array(x[:, -2]/0.6).astype(int)
        goal_y_arr = np.array(x[:, -1]/0.6).astype(int)

        count = np.finfo(np.float32).eps

        uncertainty = []

        # group each element of pos_x, pos_y, goal_x, goal_y into a tuple and iterate over them
        for pos_x, pos_y, goal_x, goal_y in zip(pos_x_arr, pos_y_arr, goal_x_arr, goal_y_arr):
            count = np.finfo(np.float32).eps
            if (pos_x, pos_y, goal_x, goal_y) in initiation_gvf.state_goal_count_dict:
                count = initiation_gvf.state_goal_count_dict[(pos_x,
                                                              pos_y, goal_x, goal_y)]
            uncertainty.append(1/np.sqrt(count))
        return np.array(uncertainty)

    values = chunked_inference(states, f, chunk_size=10_000)
    values_uncertainty_type_1 = chunked_inference(states, g, chunk_size=10_000)
    values_uncertainty_type_2 = chunked_inference(states, x, chunk_size=10_000)
    values_uncertainty_type_3 = chunked_inference(states, h, chunk_size=10_000)

    '''
    We have 3 methods of computing uncertainty
    1. 0.5*|V_1 - V_2|
    2. 0.5*|V(s) - V(s')|
    3. 1/sqrt(N(s, g))
    '''

    plt.scatter(states[:, 0], states[:, 1], c=values)
    plt.colorbar()
    g_str = np.round(goal, 2)
    file_name = f"init_gvf_seed_{seed}_episode_{episode}_goal_{g_str}"
    plt.title(f"GVF Targeting {g_str}")
    saving_path = os.path.join(
        'results', experiment_name, 'value_function_plots', f'{file_name}.png')

    plt.savefig(saving_path)
    plt.close()

    def plot_uncertainty_type(uncertainty_type, values_uncertainty):
        plt.figure()
        plt.scatter(states[:, 0], states[:, 1], c=values_uncertainty)
        plt.colorbar()
        g_str = np.round(goal, 2)
        file_name = f"init_gvf_seed_{seed}_episode_{episode}_goal_{g_str}"
        plt.title(f"GVF Targeting Uncertainty {uncertainty_type} Plot {g_str}")
        saving_path = os.path.join(
            'results', experiment_name, 'value_function_plots', f'{file_name}_uncertainty_{uncertainty_type}.png')

        plt.savefig(saving_path)
        plt.close()

    plot_uncertainty_type("twin_diff", values_uncertainty_type_1)
    plot_uncertainty_type("target_diff", values_uncertainty_type_2)
    plot_uncertainty_type("N_count_uncertainty", values_uncertainty_type_3)


def softmax(scores, temperature):
    from scipy import special
    assert temperature > 0, temperature
    return special.softmax(scores / temperature)
