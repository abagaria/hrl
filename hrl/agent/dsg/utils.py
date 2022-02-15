import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
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
    vf_probabilities = [_get_node_probability(n, method='vf') for n in nodes]
    rf_probabilities = [_get_node_probability(n, method='rf') for n in nodes]

    nodes2 = planner.salient_events
    points2 = [_get_representative_point(n) for n in nodes2]
    x_coords2 = [point[0] for point in points2]
    y_coords2 = [point[1] for point in points2]
    vf_probabilities2 = [_get_node_probability(n, method='vf') for n in nodes2]
    rf_probabilities2 = [_get_node_probability(n, method='rf') for n in nodes2]

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(x_coords, y_coords, c=vf_probabilities, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.ylabel("Candidate Nodes")
    plt.title("Intrinsic Value Function")

    plt.subplot(2, 2, 2)
    plt.scatter(x_coords, y_coords, c=rf_probabilities, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.title("Intrinsic Reward Function")

    plt.subplot(2, 2, 3)
    plt.scatter(x_coords2, y_coords2, c=vf_probabilities2, s=100)
    plt.colorbar()
    plt.xlim((0, 150))
    plt.ylim((140, 250))
    plt.ylabel("All Salient Nodes")
    plt.title("Intrinsic Value Function")

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
