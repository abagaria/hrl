import os
# import ipdb
import math
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt

from scipy import special
from hrl.salient_event.salient_event import SalientEvent


def visualize_graph_nodes_with_expansion_probabilities(planner,
                                                       episode,
                                                       experiment_name,
                                                       seed,
                                                       background_img_fname=None):

    def _get_candidate_nodes():
        return planner.get_candidate_nodes_for_expansion()

    def _get_node_probability(node, method):
        accumulator = "mean" if method == "vf" else "sample"
        score = planner.get_rnd_score(node, method=method, accumulator=accumulator)
        return score

    def _get_node_probability_normalization_factor(descendants):
        """ Sum up the selection scores of all the nodes in the graph. """
        scores = np.array([node.compute_intrinsic_reward_score(planner.exploration_agent) for node in descendants])
        return scores.sum()

    def _get_representative_point(node):
        if isinstance(node, SalientEvent):
            return _get_event_representative_point(node)
        return _get_option_representative_point(node)

    def _get_event_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.target_pos
        trigger_positions = [eg.pos for eg in event.effect_set]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0)

    def _get_option_representative_point(option):
        effect_positions = np.array([eg.pos for eg in option.effect_set])
        return np.median(effect_positions, axis=0)

    nodes = _get_candidate_nodes()
    points = [_get_representative_point(n) for n in nodes]
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    rf_probabilities = [_get_node_probability(n, method='rf') for n in nodes]

    nodes2 = planner.salient_events
    points2 = [_get_representative_point(n) for n in nodes2]
    x_coords2 = [point[0] for point in points2]
    y_coords2 = [point[1] for point in points2]
    rf_probabilities2 = [_get_node_probability(n, method='rf') for n in nodes2]

    # Expansion stats for salient events other than the start state salient event
    expansion_nodes = planner.salient_events[1:] if len(planner.salient_events) > 0 else []
    expansion_points = [_get_representative_point(n) for n in expansion_nodes]
    expansion_xcoords = [point[0] for point in expansion_points]
    expansion_ycoords = [point[1] for point in expansion_points]
    n_expansion_attempts = [n.n_expansion_attempts for n in expansion_nodes]
    n_expansions_completed = [n.n_expansions_completed for n in expansion_nodes]

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(expansion_xcoords, expansion_ycoords, c=n_expansion_attempts, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.ylabel("Candidate Nodes")
    plt.title("Number of Expansion Attempts")

    plt.subplot(2, 2, 2)
    plt.scatter(x_coords, y_coords, c=rf_probabilities, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Intrinsic Reward Function")

    plt.subplot(2, 2, 3)
    plt.scatter(expansion_xcoords, expansion_ycoords, c=n_expansions_completed, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.ylabel("All Salient Nodes")
    plt.title("Num Expansions Completed")

    plt.subplot(2, 2, 4)
    plt.scatter(x_coords2, y_coords2, c=rf_probabilities2, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Intrinsic Reward Function")

    if background_img_fname is not None:
        filename = os.path.join(os.getcwd(), f"{background_img_fname}.png")
        if os.path.isfile(filename):
            background_image = imageio.imread(filename)
            plt.imshow(background_image, zorder=0, alpha=0.5)

    prefix = "event_node_prob_graph"
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_plots/{prefix}_episode_{episode}.png")
    plt.close()


def visualize_consolidation_probabilities(dsg_trainer,
                                          episode,
                                          experiment_name,
                                          seed):

    def _get_event_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.target_pos
        trigger_positions = [eg.pos for eg in event.effect_set]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0)

    def get_probabilities(events):  # TODO: This is bad code replication
        scores = []
        window_size = 50

        if len(events) > 1:
    
            for event in events:
                key = tuple(event.target_pos)
                if key in dsg_trainer.dsg_agent.gc_successes and \
                    len(dsg_trainer.dsg_agent.gc_successes[key]) >= (2 * window_size):
                    n_recent_successes = sum(dsg_trainer.dsg_agent.gc_successes[key][-window_size:])
                    n_past_successes = sum(dsg_trainer.dsg_agent.gc_successes[key][-2*window_size:window_size])
                    score = abs(n_recent_successes - n_past_successes) / window_size
                else:  # Optimistic initialization
                    score = 1.
                
                scores.append(score)

            scores = np.array(scores)
            probabilities = special.softmax(scores / dsg_trainer.boltzmann_temperature)
            return probabilities
        
        else:
            scores = [0] * len(events)
            return np.array(scores)
    
    state = dsg_trainer.init_salient_event.target_obs
    info = dsg_trainer.init_salient_event.target_info
    connected_events = dsg_trainer._get_connected_events(state, info)
    unconnected_events = dsg_trainer._get_unconnected_events(state, info)

    connected_probabilities = get_probabilities(connected_events)
    unconnected_probabilities = get_probabilities(unconnected_events)

    connected_points = [_get_event_representative_point(e) for e in connected_events]
    connected_x_coords = [point[0] for point in connected_points]
    connected_y_coords = [point[1] for point in connected_points]

    unconnected_points = [_get_event_representative_point(e) for e in unconnected_events]
    unconnected_x_coords = [point[0] for point in unconnected_points]
    unconnected_y_coords = [point[1] for point in unconnected_points]

    plt.figure(figsize=(16, 10))

    plt.subplot(1, 2, 1)
    plt.scatter(connected_x_coords, connected_y_coords, c=connected_probabilities, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Connected Consolidation Probabilities")

    plt.subplot(1, 2, 2)
    plt.scatter(unconnected_x_coords, unconnected_y_coords, c=unconnected_probabilities, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Unconnected Consolidation Probabilities")
    
    prefix = "event_consolidation_probs"
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_plots/{prefix}_episode_{episode}.png")
    plt.close()


def visualize_all_events(dsg_trainer, episode, experiment_name, seed):
    def _get_event_representative_point(event):
        assert isinstance(event, SalientEvent)
        if event.get_target_position() is not None:
            return event.target_pos, event.target_info["room_number"]
        trigger_positions = [eg.pos for eg in event.effect_set]
        trigger_positions = np.array(trigger_positions)
        return trigger_positions.mean(axis=0), event.target_info["room_number"]

    state = dsg_trainer.init_salient_event.target_obs
    info = dsg_trainer.init_salient_event.target_info
    connected_events = dsg_trainer._get_connected_events(state, info)
    unconnected_events = dsg_trainer._get_unconnected_events(state, info)

    connected_points = [_get_event_representative_point(e) for e in connected_events]
    connected_x_coords = [point[0][0] for point in connected_points]
    connected_y_coords = [point[0][1] for point in connected_points]
    connected_room_numbers = [point[1] for point in connected_points]

    unconnected_points = [_get_event_representative_point(e) for e in unconnected_events]
    unconnected_x_coords = [point[0][0] for point in unconnected_points]
    unconnected_y_coords = [point[0][1] for point in unconnected_points]
    unconnected_room_numbers = [point[1] for point in unconnected_points]

    plt.figure(figsize=(16, 10))

    cmap = plt.cm.get_cmap("tab20")

    plt.subplot(1, 2, 1)
    plt.scatter(connected_x_coords, connected_y_coords, c=connected_room_numbers, s=100, cmap=cmap)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Connected Events")

    plt.subplot(1, 2, 2)
    plt.scatter(unconnected_x_coords, unconnected_y_coords, c=unconnected_room_numbers, s=100, cmap=cmap)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Unconnected Events")

    plt.suptitle(f"T: {dsg_trainer.env.T}")
    
    prefix = "graph_events"
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_plots/{prefix}_episode_{episode}.png")
    plt.close()


def plot_distance_table(node_distances, salient_events, episode, experiment_name, seed):
    def dict2matrix(distances, events):
        N = len(events)
        max_int = np.iinfo(np.int32).max
        matrix = np.ones((N, N), dtype=np.int32) * max_int
        for i, e1 in enumerate(events):
            for j, e2 in enumerate(events):
                matrix[i][j] = distances[e1][e2]
        return matrix

    distance_matrix = dict2matrix(node_distances, salient_events)
    labels = [f"{event.target_pos[0], event.target_pos[1]}" for event in salient_events] 
    fig, ax = plt.subplots(figsize=(30, 30))
    fig.patch.set_visible(False); ax.axis('off'); ax.axis('tight')

    ax.table(rowLabels=labels, colLabels=labels, cellText=distance_matrix, loc='center')
    fig.tight_layout()
    plt.savefig(f"plots/{experiment_name}/{seed}/value_function_plots/distance_table_episode_{episode}.png")
    plt.close()


def get_regions_in_first_screen():
    # region -> f(info) -> bool
    return dict(
        top_left=lambda info: info['player_x'] < 54 and info['player_y'] == 235,
        top_right=lambda info: info['player_x'] > 98 and info['player_y'] == 235,
        top_middle=lambda info: (89 > info['player_x'] > 63) and info['player_y'] == 235,
        middle_left=lambda info: info['player_x'] < 36 and (209 >= info['player_y'] >= 180),
        middle_middle=lambda info: (101 > info['player_x'] > 56) and info['player_y'] == 192,
        middle_right=lambda info: info['player_x'] >= 120 and info['player_y'] == 192,
        bottom_right=lambda info: info['player_x'] >= 123 and info['player_y'] == 148,
        bottom_left=lambda info: info['player_x'] <= 23 and info['player_y'] == 148,
    )

def get_lineant_regions_in_first_screen():
    return dict(
        top_left=lambda info: info['player_x'] < 54 and info['player_y'] >= 235,
        top_right=lambda info: info['player_x'] > 98 and info['player_y'] >= 235,
        top_middle=lambda info: (89 > info['player_x'] > 63) and info['player_y'] >= 235,
        middle_left=lambda info: info['player_x'] < 36 and (209 >= info['player_y'] >= 180),
        middle_middle=lambda info: (101 > info['player_x'] > 56) and (209 >= info['player_y'] >= 180),
        middle_right=lambda info: info['player_x'] >= 120 and (209 >= info['player_y'] >= 180),
        bottom_right=lambda info: info['player_x'] >= 123 and info['player_y'] <= 160,
        bottom_left=lambda info: info['player_x'] <= 23 and info['player_y'] <= 160,
    )
