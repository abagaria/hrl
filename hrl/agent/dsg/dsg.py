import ipdb
import torch
import scipy
import random
import numpy as np
from scipy.special import softmax
from pfrl.wrappers import atari_wrappers
from hrl.agent.dsc.option import ModelFreeOption
from .graph import PlanGraph
from ...salient_event.salient_event import SalientEvent
from ..dsc.dsc import RobustDSC
from ..dsc.chain import SkillChain


class SkillGraphAgent:
    def __init__(self, dsc_agent, exploration_agent, distance_metric):
        assert isinstance(dsc_agent, RobustDSC)
        assert distance_metric in ("euclidean", "vf", "ucb"), distance_metric

        self.dsc_agent = dsc_agent
        self.distance_metric = distance_metric
        
        self.planner = PlanGraph()
        self.exploration_agent = exploration_agent

        self.salient_events = []
        self.max_reward_so_far = -np.inf

    # -----------------------------–––––––--------------
    # Control loop methods
    # -----------------------------–––––––--------------

    def act(self, state, info, goal_vertex, sampled_goal):
        
        def _pick_option_to_execute_from_plan(selected_plan):
            admissible_options = [o for o in selected_plan if o.is_init_true(state, info) and not o.is_term_true(state, info)]
            if len(admissible_options) > 0:
                print(f"Got plan {selected_plan} and chose to execute {admissible_options[0]}")
                assert isinstance(admissible_options[0], ModelFreeOption)
                return admissible_options[0]
        
        if self.planner.does_path_exist(state, info, goal_vertex):
            plan = self.planner.get_path_to_execute(state, info, goal_vertex)
            if len(plan) > 0:
                selected_option = _pick_option_to_execute_from_plan(plan)
                if selected_option is not None:
                    return selected_option
        
        # TODO: What is the type of sampled_goal?
        print(f"Reverting to the DSC policy over options targeting {sampled_goal}")
        option = self.dsc_agent.act(state, info, goal_pos=sampled_goal)

        return option

    def run_loop(self, *, state, info, goal_salient_event, episode, eval_mode):
        assert isinstance(state, atari_wrappers.LazyFrames)
        assert isinstance(goal_salient_event, SalientEvent)
        assert isinstance(info, dict)
        assert isinstance(episode, int)
        assert isinstance(eval_mode, bool)

        if self.is_state_inside_vertex(state, info, goal_salient_event):
            return state, info, False, False, True

        planner_goal_vertex, dsc_goal_vertex = self.get_goal_vertices_for_rollout(state, info, goal_salient_event)
        print(f"Planner goal: {planner_goal_vertex}, DSC goal: {dsc_goal_vertex} and Goal: {goal_salient_event}")

        state, info, done, reset = self.run_sub_loop(state, info, planner_goal_vertex, goal_salient_event, episode, eval_mode)

        if not done and not reset:
            state, info, done, reset = self.run_sub_loop(state, info, dsc_goal_vertex, goal_salient_event, episode, eval_mode)

        if not done and not reset:
            state, info, done, reset = self.run_sub_loop(state, info, goal_salient_event, goal_salient_event, episode, eval_mode)

        return state, info, done, reset, self.is_state_inside_vertex(state, info, goal_salient_event)

    def run_sub_loop(self, state, info, goal_vertex, goal_salient_event, episode, eval_mode):

        done = False
        reset = False
        
        planner_condition = lambda s, i, g: self.planner.does_path_exist(s, i, g) and \
                                            not self.is_state_inside_vertex(s, i, g)

        if planner_condition(state, info, goal_vertex) and not self.is_state_inside_vertex(state, info, goal_salient_event):
            print(f"[Planner] Rolling out from {info} targeting {goal_vertex}")
            state, info, done, reset = self.planner_rollout_inside_graph(state=state,
                                                                         info=info,
                                                                         goal_vertex=goal_vertex,
                                                                         goal_salient_event=goal_salient_event,
                                                                         episode_number=episode,
                                                                         eval_mode=eval_mode)

        elif not self.is_state_inside_vertex(state, info, goal_salient_event):
            state, info, done, reset = self.dsc_outside_graph(state, info,
                                                              episode=episode,
                                                              goal_salient_event=goal_salient_event,
                                                              dsc_goal_vertex=goal_vertex)

        return state, info, done, reset

    def planner_rollout_inside_graph(self, *, state, info, goal_vertex,
                                     goal_salient_event, episode_number, eval_mode):
        """ Control loop that drives the agent from `state` to `goal_vertex` or times out. """
        assert isinstance(state, atari_wrappers.LazyFrames)
        assert isinstance(info, dict)
        assert isinstance(goal_vertex, (SalientEvent, ModelFreeOption))
        assert isinstance(episode_number, int)
        assert isinstance(eval_mode, bool)

        # Note: I removed an assertion that goal_salient_event be a SalientEvent b/c in the absence of 
        #       a state sampler, goal_vertex and goal_salient_event are the same node and they could both be options

        goal_obs, goal_pos = self.sample_from_vertex(goal_vertex)
        
        in_node = lambda node, s, info: node(info) if isinstance(node, SalientEvent) else node.is_term_true(s, info)
        inside = lambda s, info: in_node(goal_vertex, s, info)
        outside = lambda s, info: in_node(goal_salient_event, s, info)

        done = False
        reset = False
        option = None
        reached = False

        while not reached and not done and not reset:
            option = self.act(state, info, goal_vertex, sampled_goal=goal_pos)

            # TODO: goal_salient_event or goal_vertex? Whats the difference?
            state, info, done, reset, _, _ = self.perform_option_rollout(state,
                                                                         info, 
                                                                         option,
                                                                         episode_number,
                                                                         eval_mode,
                                                                         goal_vertex)

            reached = inside(state, info) or outside(state, info)

        return state, info, done, reset

    def dsc_outside_graph(self, state, info, episode, goal_salient_event, dsc_goal_vertex):
        _, dsc_goal_pos = self.sample_from_vertex(dsc_goal_vertex)
        dsc_interrupt_handle = lambda s, i: self.is_state_inside_vertex(s, i, dsc_goal_vertex) or\
                                            self.is_state_inside_vertex(s, i, goal_salient_event)

        if dsc_interrupt_handle(state, info):
            print(f"Not rolling out DSC because {info} triggered interrupt handle")
            return state, info, False, False

        print(f"Rolling out DSC with goal vertex {dsc_goal_vertex} and goal state {dsc_goal_pos}")

        # Deliberately did not add `eval_mode` to the DSC rollout b/c we usually do this when we
        # want to fall off the graph, and it is always good to do some exploration when outside the graph
        state, info, done, reset, new_options = self.dsc_agent.dsc_rollout(state, info, 
                                                                           dsc_goal_vertex,
                                                                           episode, eval_mode=False,
                                                                           interrupt_handle=dsc_interrupt_handle)

        for new_option in new_options:
            self.add_newly_created_option_to_plan_graph(new_option)

        return state, info, done, reset

    # -----------------------------–––––––--------------
    # Graph Expansion
    # -----------------------------–––––––--------------

    def model_free_extrapolation(self, env, state, info, episode):
        """ Single episodic rollout of the exploration policy to extend the graph. """
        
        starting_nodes = [event for event in self.salient_events if event(info)]

        print(f"Performing model-free extrapolation from {env.get_current_position()} and {starting_nodes}")

        state_reward_pairs, reward, length, max_reward_so_far = self.exploration_agent.rollout(env, 
                                                                                               state,
                                                                                               episode, 
                                                                                               self.max_reward_so_far)

        return self.create_target_state(state_reward_pairs)

    def create_target_state(self, state_reward_pairs):

        best_state = None
        max_intrinsic_reward = -np.inf

        for state, intrinsic_reward in state_reward_pairs:
            
            if intrinsic_reward > max_intrinsic_reward:
                best_state = state
                max_intrinsic_reward = intrinsic_reward

        return best_state, max_intrinsic_reward

    def get_node_to_expand(self, method="rf", accumulator="sample"):
        """ Use the RND agent to find the graph node to expand. """
        
        nodes = self.get_candidate_nodes_for_expansion()

        if len(nodes) > 0:
            scores = [self.get_rnd_score(node, method, accumulator) for node in nodes]
            sampled_node = self.pick_expansion_node(nodes, scores)
            return sampled_node
        
        assert len(nodes) == 0, nodes
        return self.dsc_agent.init_salient_event

    def pick_expansion_node(self, nodes, scores, choice_type="deterministic"):
        assert choice_type in ("deterministic", "stochastic"), choice_type

        def _deterministic_pick_node(nodes, scores):
            idx = np.argmax(scores)
            node = nodes[random.choice(idx) if isinstance(idx, np.ndarray) else idx]
            return node

        def _stochastic_pick_node(nodes, scores, temperature=1.):
            probabilities = softmax(scores / temperature)

            assert all(probabilities.tolist()) <= 1., probabilities
            np.testing.assert_almost_equal(probabilities.sum(), 1., err_msg=f"{probabilities}", decimal=3)

            sampled_node = np.random.choice(nodes, size=1, p=probabilities)[0]
            print(f"[Temp={temperature}] | Descendants: {nodes} | Probs: {probabilities} | Chose: {sampled_node}")
            return sampled_node

        if choice_type == "deterministic":
            return _deterministic_pick_node(nodes, scores)
        
        return _stochastic_pick_node(nodes, scores)

    def get_candidate_nodes_for_expansion(self, min_number_of_points=3):
        """ There are two possibilities: First, we only consider nodes to which there is a path, but that is too
         conservative. Second, we consider all the events ever discovered, but that could be too aggressive since
         we may not have enough data to correctly estimate its closest node in the graph. Therefore, we try to be
         aggressive, but temper it with the requirement that we have seen that event a small number of times. """
        nodes_with_enough_data = [node for node in self.salient_events if len(node.effect_set) >= min_number_of_points]
        return nodes_with_enough_data

    def get_rnd_score(self, node, method="vf", accumulator="mean"):
        assert method in ("vf", "rf"), method
        assert accumulator in ("mean", "sample"), accumulator
        
        def lz_to_np(lz):
            return np.array(lz).squeeze().transpose(1, 2, 0)

        def sample_vf_score(n):
            s = lz_to_np(random.choice(n.effect_set))
            return self.exploration_agent.value_function(s)[0]

        def average_vf_score(n):
            states = [lz_to_np(eg.obs) for eg in n.effect_set]
            return self.exploration_agent.value_function(states).mean()

        def sample_rint_score(n):
            s = lz_to_np(random.choice(n.effect_set).obs)
            s = s[:, :, -1]  # Extract last frame of LazyFrames
            return self.exploration_agent.reward_function([s])[0]

        def average_rint_score(n):
            states = [lz_to_np(eg.obs)[:, :, -1] for eg in n.effect_set]
            return np.array(
                [self.exploration_agent.reward_function(state) for state in states]
            ).mean()

        if method == "vf":
            if accumulator == "mean":
                return average_vf_score(node)
            return sample_vf_score(node)
        
        if accumulator == "mean":
            return average_rint_score(node)

        return sample_rint_score(node)

    # -----------------------------–––––––--------------
    # Graph Consolidation
    # -----------------------------–––––––--------------

    def perform_option_rollout(self, state, info, option, episode, eval_mode, goal_salient_event):
        assert isinstance(option, ModelFreeOption)

        state, done, reset, visited_positions, goal_pos, info = option.rollout(state,
                                                                               info,
                                                                               goal_salient_event,
                                                                               eval_mode)
        
        finished_learning = self.dsc_agent.manage_chain_after_option_rollout(option, episode)

        if finished_learning:
            self.add_newly_created_option_to_plan_graph(option)

        # Remove edges that no longer make sense for the executed option
        self.modify_node_connections(executed_option=option)

        # Modify the edge weight associated with the executed option
        self.modify_edge_weight(executed_option=option, final_state=state, final_info=info)

        return state, info, done, reset, visited_positions, goal_pos

    # -----------------------------–––––––--------------
    # Distance functions
    # -----------------------------–––––––--------------

    def closest_node_lut(self, target_event):
        spos  = (77, 235)
        gpos0 = (123, 148)
        gpos1 = (132, 192)
        gpos2 = (24, 235)
        gpos3 = (130, 235)
        gpos4 = (77, 192)
        gpos5 = (23, 148)

        # maps destination -> source
        lut = {
            spos : gpos4,
            gpos0: gpos1,
            gpos1: random.choice([spos, gpos4]),
            gpos2: spos,
            gpos3: spos,
            gpos4: spos,
            gpos5: gpos0,
        }

        target_pos = lut[tuple(target_event.target_pos)]

        for beta in self.salient_events:
            if tuple(beta.target_pos) == target_pos:
                return beta
        
        ipdb.set_trace()

    def choose_closest_source_target_vertex_pair(self, state, info, goal_salient_event, choose_among_events):
        candidate_vertices_to_fall_from = self.planner.get_reachable_nodes_from_source_state(state, info)
        candidate_vertices_to_fall_from = list(candidate_vertices_to_fall_from) + self.get_corresponding_events(state, info)

        candidate_vertices_to_jump_to = self.planner.get_nodes_that_reach_target_node(goal_salient_event)
        candidate_vertices_to_jump_to = list(candidate_vertices_to_jump_to) + [goal_salient_event]

        if choose_among_events:
            candidate_vertices_to_fall_from = [v for v in candidate_vertices_to_fall_from if isinstance(v, SalientEvent)]
            candidate_vertices_to_jump_to = [v for v in candidate_vertices_to_jump_to if isinstance(v, SalientEvent)]

        candidate_vertices_to_fall_from = list(set(candidate_vertices_to_fall_from))
        candidate_vertices_to_jump_to = list(set(candidate_vertices_to_jump_to))

        return self.get_closest_pair_of_vertices(candidate_vertices_to_fall_from,
                                                 candidate_vertices_to_jump_to,
                                                 metric=self.distance_metric)

    def get_goal_vertices_for_rollout(self, state, info, goal_salient_event):
        # Revise the goal_salient_event if it cannot be reached from the current state
        if not self.planner.does_path_exist(state, info, goal_salient_event):
            closest_vertex_pair = self.choose_closest_source_target_vertex_pair(state, info, goal_salient_event, True)  # TODO: choose_among
            if closest_vertex_pair is not None:
                planner_goal_vertex, dsc_goal_vertex = closest_vertex_pair
                print(f"Revised planner goal vertex to {planner_goal_vertex} and dsc goal vertex to {dsc_goal_vertex}")
                return planner_goal_vertex, dsc_goal_vertex
        return goal_salient_event, goal_salient_event

    def get_closest_pair_of_vertices(self, src_vertices, dest_vertices, metric):
        def sample(A, num_rows):
            return A[np.random.randint(A.shape[0], size=num_rows), :].squeeze()

        if len(src_vertices) > 0 and len(dest_vertices) > 0:
            distance_matrix = self.get_distance_matrix(src_vertices,
                                                       dest_vertices,
                                                       metric=metric)
            min_array = np.argwhere(distance_matrix == np.min(distance_matrix)).squeeze()

            if len(min_array.shape) > 1 and min_array.shape[0] > 1:
                min_array = sample(min_array, 1)

            return src_vertices[min_array[0]], dest_vertices[min_array[1]]

    def get_distance_matrix(self, src_vertices, dest_vertices, metric="euclidean"):
        assert metric in ("euclidean", "vf", "ucb"), metric

        def sample(vertex, key):
            assert key in ('obs', 'pos')
            x = random.choice(vertex.effect_set)
            return x.obs if key == 'obs' else x.pos

        def value_to_distance(v, n):
            pos_value = v.squeeze().abs()
            if metric == "ucb":
                _n = torch.as_tensor(n).float().to(
                    self.dsc_agent.global_option.solver.device
                )
                pos_value = pos_value + (1. / torch.sqrt(_n))
            return 1. / pos_value

        def batched_pairwise_distances(observationsA, observationsB, vf, n_points):
            distance_matrix = torch.zeros((len(observationsA), len(observationsB)))
            distance_matrix = distance_matrix.to(self.dsc_agent.global_option.solver.device)

            for i, obsA in enumerate(observationsA):
                assert isinstance(obsA, atari_wrappers.LazyFrames), type(obsA)
                with torch.no_grad():
                    value = vf(obsA, observationsB)
                    distance_matrix[i, :] = value_to_distance(value, n_points)

            return distance_matrix.cpu().numpy()

        def euclidean_distances(positionsA, positionsB):
            return scipy.spatial.distance.cdist(positionsA, positionsB)

        attribute = 'pos' if metric == "euclidean" else 'obs'
        src_observations = [sample(v, attribute) for v in src_vertices]
        dst_observations = [sample(v, attribute) for v in dest_vertices]
        n_dst_observations = [len(v.effect_set) for v in dest_vertices]

        if metric == "euclidean":
            return euclidean_distances(src_observations, dst_observations)

        return batched_pairwise_distances(src_observations,
                                          dst_observations,
                                          self.dsc_agent.global_option.value_function,
                                          n_dst_observations)

    # -----------------------------–––––––--------------
    # Maintaining the graph
    # -----------------------------–––––––--------------

    def modify_edge_weight(self, executed_option, final_state, final_info):
        """
        The edge weight in the graph represents the cost of executing an option policy.
        Remember that each option could be connected to a bunch of other options (if the effect
        set of the first option is inside the inside the initiation set of the other options).
        So, we must check if option execution successfully landed up in any of those initiation sets,
        and decrement the cost if that is the case. If we didn't reach the initiation set of some
        option that we are connected to, we must increment the edge cost.

        Args:
            executed_option (Option)
            final_state (LazyFrames)
            final_info (dict)

        """
        def modify(node, success):
            edge_weight = self.planner.plan_graph[executed_option][node]["weight"]
            new_weight = (0.95 ** success) * edge_weight
            self.planner.set_edge_weight(executed_option, node, new_weight)

        def inside_vertex(node, obs, info):
            if isinstance(node, ModelFreeOption):
                return node.is_init_true(obs, info)
            return node(info)

        outgoing_edges = [edge for edge in self.planner.plan_graph.edges if edge[0] == executed_option]
        neighboring_vertices = [edge[1] for edge in outgoing_edges]
        successfully_reached_vertices = [vertex for vertex in neighboring_vertices if inside_vertex(vertex, final_state, final_info)]
        failed_reaching_vertices = [vertex for vertex in neighboring_vertices if not inside_vertex(vertex, final_state, final_info)]

        for vertex in successfully_reached_vertices:
            modify(vertex, +1)

        for vertex in failed_reaching_vertices:
            modify(vertex, -1)

    def _delete_outgoing_edges_if_needed(self, executed_option):
        assert isinstance(executed_option, ModelFreeOption)

        outgoing_nodes = self.planner.get_outgoing_nodes(executed_option)

        for node in outgoing_nodes:
            assert isinstance(node, (ModelFreeOption, SalientEvent))

            if isinstance(node, ModelFreeOption):
                should_remove = not SkillChain.should_exist_edge_between_options(executed_option, node) \
                                and len(node.get_effective_effect_set()) > 0 \
                                and executed_option.parent != node
            else:
                assert isinstance(node, SalientEvent)
                should_remove = not SkillChain.should_exist_edge_from_option_to_event(executed_option, node)

            if should_remove:
                print(f"Deleting edge from {executed_option} to {node}")
                self.planner.plan_graph.remove_edge(executed_option, node)
    
    def _delete_incoming_edges_if_needed(self, executed_option):
        assert isinstance(executed_option, ModelFreeOption)

        incoming_nodes = self.planner.get_incoming_nodes(executed_option)

        for node in incoming_nodes:
            assert isinstance(node, (ModelFreeOption, SalientEvent))

            if isinstance(node, ModelFreeOption):
                should_remove = not SkillChain.should_exist_edge_between_options(node, executed_option) \
                                and len(node.get_effective_effect_set()) > 0 \
                                and node.parent != executed_option
            else:
                assert isinstance(node, SalientEvent)
                should_remove = not SkillChain.should_exist_edge_from_event_to_option(node, executed_option)

            if should_remove:
                print(f"Deleting edge from {node} to {executed_option}")
                self.planner.plan_graph.remove_edge(node, executed_option)

    def add_potential_edges(self, option):

        for node in self.planner.plan_graph.nodes:

            if node != option and not self.planner.plan_graph.has_edge(option, node):
                if isinstance(node, SalientEvent) and SkillChain.should_exist_edge_from_option_to_event(option, node):
                    print(f"Adding edge from {option} to {node}")
                    self.planner.add_edge(option, node)
                if isinstance(node, ModelFreeOption) and SkillChain.should_exist_edge_between_options(option, node):
                    print(f"Adding edge from {option} to {node}")
                    self.planner.add_edge(option, node)

            if node != option and not self.planner.plan_graph.has_edge(node, option):
                if isinstance(node, SalientEvent) and SkillChain.should_exist_edge_from_event_to_option(node, option):
                    print(f"Adding edge from {node} to {option}")
                    self.planner.add_edge(node, option)
                if isinstance(node, ModelFreeOption) and SkillChain.should_exist_edge_between_options(node, option):
                    print(f"Adding edge from {node} to {option}")
                    self.planner.add_edge(node, option)
    
    def modify_node_connections(self, executed_option):
        self._delete_outgoing_edges_if_needed(executed_option)
        self._delete_incoming_edges_if_needed(executed_option)

    def add_newly_created_option_to_plan_graph(self, newly_created_option):

        assert newly_created_option.get_training_phase() == "initiation_done", \
            f"{newly_created_option} in {newly_created_option.get_training_phase()}"

        # If the new option isn't already in the graph, add it
        self.planner.add_node(newly_created_option)

        # Skill Chain corresponding to the newly created option
        chain = self.dsc_agent.chains[newly_created_option.chain_id - 1]  # type: SkillChain
        is_leaf_node = newly_created_option in chain.get_leaf_nodes_from_skill_chain()
        init_salient_event = chain.init_salient_event
        target_salient_event = chain.target_salient_event

        # If the target_salient_event is not in the plan-graph, add it
        if target_salient_event not in self.planner.plan_graph:
            self.planner.add_node(target_salient_event)

        # If the init salient event is not in the plan-graph, add it
        if is_leaf_node and (init_salient_event not in self.planner.plan_graph):
            self.planner.add_node(init_salient_event)

        # For adding edges, there are 3 possible cases:
        # 1. Nominal case - you are neither a root nor a leaf - just connect to your parent option
        # 2. Leaf Option - if you are a leaf option, you should add a 0 weight edge from the init
        #                  salient event to the leaf option
        # 3. Root option - `if you are a root option, you should add an optimistic node from the
        #                  root option to the target salient event

        # Case 3: Add edge from the first salient event to the first backward option
        if newly_created_option.parent is None:
            print(f"Case 3: Adding edge from {newly_created_option} to {target_salient_event}")
            self.planner.add_edge(newly_created_option, target_salient_event, edge_weight=1.)

        # Case 2: Leaf option # TODO: Need to check intersection with the init_salient_event as well
        if chain.is_chain_completed() \
                and is_leaf_node \
                and chain.should_exist_edge_from_event_to_option(init_salient_event, newly_created_option):

            print(f"Case 2: Adding edge from {init_salient_event} to {newly_created_option}")
            self.planner.add_edge(init_salient_event, newly_created_option, edge_weight=0.)

        # Case 1: Nominal case
        if newly_created_option.parent is not None:
            print(f"Case 1: Adding edge from {newly_created_option} to {newly_created_option.parent}")
            self.planner.add_edge(newly_created_option, newly_created_option.parent, edge_weight=1.)

        # Case 4: I did not consider this case before. But, you could also intersect with any other
        # event in the MDP -- you might not rewire your chain because that event is not "completed/chained-to",
        # but that still means that there should be an edge from it to the newly learned option

        # 1. Get intersecting events
        # 2. Add an edge from each intersecting event to the newly_created_option
        events = [event for event in self.salient_events if event != chain.target_salient_event]
        intersecting_events = [event for event in events
                               if chain.should_exist_edge_from_event_to_option(event, newly_created_option)]
        for event in intersecting_events:  # type: SalientEvent
            assert isinstance(event, SalientEvent)
            print(f"Adding edge from {event} to {newly_created_option}")
            self.planner.add_edge(event, newly_created_option, edge_weight=0.)

        # 1. Get intersecting options
        # 2. Add an each from each intersecting option to the newly_created_option
        other_options = [o for o in self.planner.option_nodes if o != newly_created_option]
        
        for other_option in other_options:  # type: ModelBasedOption
            if SkillChain.should_exist_edge_between_options(other_option, newly_created_option):
                print(f"Adding edge from {other_option} to {newly_created_option}")
                self.planner.add_edge(other_option, newly_created_option)
            if SkillChain.should_exist_edge_between_options(newly_created_option, other_option):
                print(f"Adding edge from {newly_created_option} to {other_option}")
                self.planner.add_edge(newly_created_option, other_option)

        self.update_chain_init_descendants()

        # TODO: Expansion classifier
        # if chain.should_expand_initiation_classifier(newly_created_option):
        #     newly_created_option.expand_initiation_classifier(chain.init_salient_event)

        if chain.should_complete_chain(newly_created_option):
            chain.set_chain_completed()

    def update_chain_init_descendants(self):
        for chain in self.dsc_agent.chains:  # type: SkillChain
            assert isinstance(chain, SkillChain)
            descendants = self.planner.get_reachable_nodes_from_source_node(chain.init_salient_event)
            ancestors = self.planner.get_nodes_that_reach_target_node(chain.init_salient_event)
            chain.set_init_descendants(descendants)
            chain.set_init_ancestors(ancestors)

    # -----------------------------–––––––--------------
    # Utility functions
    # -----------------------------–––––––--------------

    @staticmethod
    def sample_from_vertex(vertex):
        assert isinstance(vertex, (ModelFreeOption, SalientEvent)), f"{type(vertex)}"

        if isinstance(vertex, ModelFreeOption):
            return vertex.get_goal_for_rollout()
        if vertex.get_target_position() is not None:
            return vertex.get_target_obs(), vertex.get_target_position()
        sample = random.choice(vertex.effect_set)
        return sample.obs, sample.pos

    @staticmethod
    def is_state_inside_vertex(state, info, vertex):
        assert isinstance(info, dict), f"{type(info)}"
        assert isinstance(state, atari_wrappers.LazyFrames), f"{type(state)}"
        assert isinstance(vertex, (ModelFreeOption, SalientEvent)), f"{type(vertex)}"

        # State satisfies the salient event
        if isinstance(vertex, SalientEvent):
            return vertex(info)

        # State is in the term set of the option
        return vertex.is_term_true(state, info)

    def get_corresponding_events(self, state, info):
        return [event for event in self.salient_events if event(info)]

    def add_salient_event(self, salient_event):
        assert isinstance(salient_event, SalientEvent)
        self.salient_events.append(salient_event)

    @staticmethod
    def _get_player_position(info):
        return info["player_x"], info["player_y"]
