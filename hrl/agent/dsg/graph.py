import itertools
import numpy as np  
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.shortest_paths as shortest_paths
from pfrl.wrappers import atari_wrappers
from ..dsc.option import ModelFreeOption
from ...salient_event.salient_event import SalientEvent

class PlanGraph:
    def __init__(self):
        self.plan_graph = nx.DiGraph()
        self.shortest_paths = {}
        self.option_nodes = []
        self.salient_nodes = []

    # ----------------------------------------------------------------------------
    # Methods for maintaining the graph
    # ----------------------------------------------------------------------------

    def add_node(self, node):
        if node not in self.plan_graph.nodes:
            self.plan_graph.add_node(node)

        if self._is_option_type(node) and (node not in self.option_nodes):
            self.option_nodes.append(node)
        elif isinstance(node, SalientEvent) and (node not in self.salient_nodes):
            self.salient_nodes.append(node)

    def add_edge(self, option1, option2, edge_weight=1.):
        self.plan_graph.add_edge(option1, option2)
        self.set_edge_weight(option1, option2, edge_weight)

    def set_edge_weight(self, option1, option2, weight):
        if self.plan_graph.has_edge(option1, option2):
            self.plan_graph[option1][option2]["weight"] = weight

    # ----------------------------------------------------------------------------
    # Methods for querying the graph
    # ----------------------------------------------------------------------------

    def does_path_exist(self, state, info, node):
        assert self._is_observation_type(state), f"{type(state)}"
        assert self._is_option_or_event_type(node), f"{type(node)}"
        
        start_nodes = self._get_available_options(state, info)
        does_exists = [self.does_path_exist_between_nodes(start, node) for start in start_nodes]

        return any(does_exists)

    def get_unconnected_nodes(self, state, info, nodes):
        """ Return the nodes for which there is no path from `state`. """
        assert self._is_observation_type(state), f"{type(state)}"
        assert isinstance(nodes, list), f"{type(nodes)}"

        start_nodes = self._get_available_options(state, info)

        # If there are no options from the current state, all nodes are unconnected
        if len(start_nodes) == 0:
            return nodes

        unconnected_nodes = []

        # A goal_node is unconnected if there is *no* start_node that connects to it
        for goal_node in nodes:
            if all([not self.does_path_exist_between_nodes(s, goal_node) for s in start_nodes]):
                unconnected_nodes.append(goal_node)

        return unconnected_nodes

    def get_path_to_execute(self, start_state, info, goal_node):
        assert self._is_observation_type(start_state)
        assert self._is_option_or_event_type(goal_node)

        paths, costs = self.get_shortest_paths(start_state, info, goal_node)

        if not paths:
            return paths

        # Sort all the paths in ascending order of path-costs
        min_cost = min(costs)
        paths_with_min_costs = [path for (path, cost) in zip(paths, costs) if cost == min_cost]

        # If there are multiple paths with the same cost, choose the one that has the fewest number of options
        paths_sorted_by_length = sorted(paths_with_min_costs, key=lambda x: len(x))
        path_to_execute = paths_sorted_by_length[0]

        # Filter out all the salient events from the path so that it is actually executable
        option_sequence_to_execute = list(filter(lambda node: self._is_option_type(node), path_to_execute))

        return option_sequence_to_execute

    def get_reachable_nodes_from_source_state(self, state, info):
        """ Get all the nodes in the graph you can reach from state. """
        assert self._is_observation_type(state), f"{type(state)}"

        start_nodes = self._get_available_options(state, info)
        reachable_nodes = [self.get_reachable_nodes_from_source_node(src) for src in start_nodes]
        reachable_nodes = list(itertools.chain.from_iterable(reachable_nodes))
        return reachable_nodes

    def get_reachable_nodes_from_source_state(self, state, info):
        """ Get all the nodes in the graph you can reach from state. """
        assert self._is_observation_type(state)

        start_nodes = self._get_available_options(state, info)
        reachable_nodes = [self.get_reachable_nodes_from_source_node(src) for src in start_nodes]
        reachable_nodes = list(itertools.chain.from_iterable(reachable_nodes))
        return reachable_nodes

    def get_reachable_nodes_from_source_node(self, source_node):
        assert self._is_option_or_event_type(source_node), f"{type(source_node)}"
        if source_node not in self.plan_graph.nodes:
            return set()
        return nx.algorithms.dag.descendants(self.plan_graph, source_node)

    def get_nodes_that_reach_target_node(self, target_node):
        """ Get all the nodes from which you can reach the `target_node`. """
        assert self._is_option_or_event_type(target_node), f"{type(target_node)}"
        if target_node in self.plan_graph.nodes:
            return nx.algorithms.dag.ancestors(self.plan_graph, target_node)
        return []

    # ----------------------------------------------------------------------------
    # Private Methods
    # ----------------------------------------------------------------------------

    def get_shortest_paths(self, state, info, goal_node):
        assert self._is_observation_type(state), f"{type(state)}"
        assert self._is_option_or_event_type(goal_node), f"{goal_node}"

        paths, path_costs = [], []

        start_nodes = self._get_available_options(state, info)

        for start_node in start_nodes:
            if self.does_path_exist_between_nodes(start_node, goal_node):
                path, cost = self.get_shortest_path_between_nodes(start_node, goal_node)
                paths.append(path)
                path_costs.append(cost)

        return paths, path_costs


    def does_path_exist_between_nodes(self, node1, node2):
        assert self._is_option_or_event_type(node1), f"{type(node1)}"
        assert self._is_option_or_event_type(node2), f"{type(node2)}"

        if node1 not in self.plan_graph.nodes or node2 not in self.plan_graph.nodes:
            return False

        return shortest_paths.has_path(self.plan_graph, node1, node2)

    def get_shortest_path_between_nodes(self, node1, node2):
        assert self._is_option_or_event_type(node1), f"{type(node1)}"
        assert self._is_option_or_event_type(node2), f"{type(node2)}"

        def _get_path_cost(n1, n2):
            return nx.dijkstra_path_length(self.plan_graph, n1, n2)

        def _get_path(n1, n2):
            return nx.dijkstra_path(self.plan_graph, n1, n2)

        path = _get_path(node1, node2)
        path_cost = _get_path_cost(node1, node2)

        # If we are targeting an option effect set, we have to add that option
        # to the path and account for the additional cost of executing that option
        if self._is_option_type(node2):
            path += [node2]
            neighboring_nodes = self.get_outgoing_nodes(node2)
            additional_costs = [_get_path_cost(node2, node3) for node3 in neighboring_nodes]
            path_cost += min(additional_costs)

        return path, path_cost

    def _get_available_options(self, state, info):
        assert self._is_observation_type(state), f"{type(state)}"
        return [option for option in self.option_nodes if option.is_init_true(state, info)]

    def get_outgoing_nodes(self, node):
        """ Get the nodes you can reach from `node` in the plan_graph. """
        edges = [edge for edge in self.plan_graph.edges if node == edge[0]]
        connected_nodes = itertools.chain.from_iterable(edges)
        neighboring_nodes = [n for n in connected_nodes if n != node]
        return neighboring_nodes

    def get_incoming_nodes(self, node):
        """ Get the nodes from which you can get to `node` in a single transition. """
        edges = [edge for edge in self.plan_graph.edges if node == edge[1]]
        connected_nodes = itertools.chain.from_iterable(edges)
        neighboring_nodes = [n for n in connected_nodes if n != node]
        return neighboring_nodes

    @staticmethod
    def _is_observation_type(x):
        return isinstance(x, (atari_wrappers.LazyFrames, np.ndarray))

    @staticmethod
    def _is_option_type(node):
        return isinstance(node, ModelFreeOption)

    @staticmethod
    def _is_option_or_event_type(node):
        return isinstance(node, (ModelFreeOption, SalientEvent))
