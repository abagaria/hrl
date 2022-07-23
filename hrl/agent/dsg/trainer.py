import ipdb
import time
import random
import pickle
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.shortest_paths as shortest_paths

from scipy import special
from pfrl.wrappers import atari_wrappers

from hrl.agent.dsc.utils import info_to_pos
from .dsg import SkillGraphAgent
from ..dsc.dsc import RobustDSC
from hrl.salient_event.salient_event import SalientEvent
from hrl.agent.bonus_based_exploration.RND_Agent import RNDAgent
from hrl.agent.dsc.utils import plot_classifier_predictions
from hrl.agent.dsg.utils import visualize_consolidation_probabilities, visualize_all_events
from hrl.agent.dsg.utils import visualize_graph_nodes_with_expansion_probabilities, plot_distance_table
from hrl.agent.dsg.utils import get_regions_in_first_screen, get_lineant_regions_in_first_screen


class DSGTrainer:
    def __init__(self, env, dsc, dsg, rnd,
                 expansion_freq,
                 expansion_duration,
                 rnd_log_filename,
                 goal_selection_criterion="random",
                 predefined_events=[],
                 enable_rnd_logging=False,
                 disable_graph_expansion=False,
                 reject_jumping_states=False,
                 use_strict_regions=True,
                 make_off_policy_update=False,
                 goal_selection_epsilon=0.2,
                 boltzmann_temperature=2.0,
                 create_sparse_graph=False,
                 use_empirical_distances=False,
                 expansion_fraction_threshold=0.5):
        assert isinstance(dsc, RobustDSC)
        assert isinstance(dsg, SkillGraphAgent)
        assert isinstance(rnd, RNDAgent)
        assert goal_selection_criterion in ("random",
                                            "closest",
                                            "random_unconnected",
                                            "boltzmann_unconnected",
                                            "competence")

        self.env = env
        self.dsc_agent = dsc
        self.dsg_agent = dsg
        self.rnd_agent = rnd
        self.expansion_freq = expansion_freq
        self.expansion_duration = expansion_duration
        self.init_salient_event = dsc.init_salient_event
        self.goal_selection_criterion = goal_selection_criterion
        self.reject_jumping_states = reject_jumping_states
        self.use_strict_regions = use_strict_regions
        self.make_off_policy_update = make_off_policy_update
        self.goal_selection_epsilon = goal_selection_epsilon
        self.boltzmann_temperature = boltzmann_temperature
        self.create_sparse_graph = create_sparse_graph
        self.use_empirical_distances = use_empirical_distances
        self.expansion_fraction_threshold = expansion_fraction_threshold

        self.salient_events = []
        self.predefined_events = predefined_events

        for event in predefined_events + [self.init_salient_event]:
            self.add_salient_event(event)

        # Logging for exploration rollouts
        self.n_extrinsic_subgoals = 0
        self.n_intrinsic_subgoals = 0
        self.rnd_extrinsic_rewards = []
        self.rnd_intrinsic_rewards = []
        self.rnd_log_filename = rnd_log_filename
        self.enable_rnd_logging = enable_rnd_logging
        self.disable_graph_expansion = disable_graph_expansion
        self.deleted_events = []

        # Logging for distance function evaluations
        self.distance_classification_accuracies = []

        # Whether we are currently expanding or consolidating the graph
        self.graph_mode = "consolidation"

    # ---------------------------------------------------
    # Run loops
    # ---------------------------------------------------

    def should_expand(self, episode, method="fraction"):
        assert method in ("fraction", "frequency"), method

        if self.disable_graph_expansion:
            return False
        
        if method == "frequency" or len(self.predefined_events) > 0:
            return episode % self.expansion_freq == 0
        
        # When we have connected most of the graph, then its time to expand again
        s0 = self.init_salient_event.target_obs
        i0 = self.init_salient_event.target_info

        candidate_events = [event for event in self.salient_events if not event(i0)]
        
        n_total = len(candidate_events)
        n_unconnected = len(self._get_unconnected_events(s0, i0))
        n_connected = n_total - n_unconnected
        assert n_total >= n_unconnected, f"{n_total, n_unconnected}"

        return (n_total <= 4) or (n_connected / n_total) > self.expansion_fraction_threshold

    def run_loop(self, start_episode, num_episodes, consolidation_duration=50):
        
        # Each iteration contains N episodes of expansion or M episodes of consolidation
        iteration = 0
        episode = start_episode
        
        while episode < start_episode + num_episodes:

            if self.should_expand(episode) and self.graph_mode == "consolidation":
                self.graph_expansion_run_loop(episode, self.expansion_duration)
                episode += self.expansion_duration
            elif len(self.salient_events) > 0:
                self.graph_consolidation_run_loop(episode, duration=consolidation_duration)
                episode += consolidation_duration
            else:
                ipdb.set_trace()

            if len(self.predefined_events) > 0:
                self.distance_classification_accuracies.append(
                    self.dsg_agent.evaluate_distance_metric_accuracy(
                       self.salient_events, self.dsg_agent.distance_metric
                    )
                ) 

            t0 = time.time()
            with open(self.dsc_agent.log_file, "wb+") as f:
                pickle.dump(self.dsg_agent.gc_successes, f)
            print(f"[Episode={episode} Seed={self.dsc_agent.seed}] Took {time.time() - t0}s to save gc logs")

            if iteration > 0 and iteration % 5 == 0:
                visualize_graph_nodes_with_expansion_probabilities(self.dsg_agent,
                                                           episode,
                                                           self.dsc_agent.experiment_name,
                                                           self.dsc_agent.seed)

                visualize_consolidation_probabilities(self, episode, 
                                                    self.dsc_agent.experiment_name, self.dsc_agent.seed)

                visualize_all_events(self, episode, self.dsc_agent.experiment_name, self.dsc_agent.seed)

                plot_distance_table(self.dsg_agent.node_distances, self.salient_events, episode,
                                    self.dsc_agent.experiment_name, self.dsc_agent.seed)

                if self.dsc_agent.rnd_data_path:
                    for option in self.dsc_agent.mature_options:
                        plot_classifier_predictions(
                            option, self.dsc_agent.rnd_frames,
                            self.dsc_agent.rnd_rams, episode, 
                            self.dsc_agent.seed, self.dsc_agent.experiment_name
                        )
            
            iteration += 1  

    def graph_expansion_run_loop(self, start_episode, num_episodes):
        exploration_inits = []
        exploration_actions = []
        exploration_observations = []
        exploration_visited_infos = []
        exploration_extrinsic_rewards = []

        self.graph_mode = "expansion"

        # Single flat list with all the infos visited during planning and exploration
        full_trajectories = []

        for episode in range(start_episode, start_episode + num_episodes):
            state, info = self.env.reset()
            expansion_node = self.dsg_agent.get_node_to_expand()
            expansion_node.n_expansion_attempts += 1
            
            print("=" * 80); print(f"[Expansion] Episode: {episode} Step: {self.env.T}"); print("=" * 80)
            print(f"Attempting to expand {expansion_node}")

            # Use the planner to get to the expansion node
            state, info, done, reset, reached, expansion_trajectory = self.dsg_agent.run_loop(
                                                                        state=state,
                                                                        info=info,
                                                                        goal_salient_event=expansion_node,
                                                                        episode=episode,
                                                                        eval_mode=False)

            if expansion_trajectory:
                full_trajectories.extend(expansion_trajectory)

            if reached and not done and not reset:
                expansion_node.n_expansions_completed += 1
                init_state, observations, actions, rewards, visited_infos = self.exploration_rollout(state, episode)

                if len(observations) > 0:
                    exploration_inits.append(init_state)
                    exploration_actions.append(actions)
                    full_trajectories.extend(visited_infos)
                    exploration_observations.append(observations)
                    exploration_extrinsic_rewards.append(rewards)
                    exploration_visited_infos.append(visited_infos)

        # Examine *all* exploration trajectories and extract potential salient events
        if len(exploration_observations) > 0:

            intrinsic_subgoals, extrinsic_subgoals, \
            intrinsic_trajectory_idx, extrinsic_trajectory_idx = self.new_subgoal_extractor(
                exploration_observations,
                exploration_extrinsic_rewards,
                exploration_visited_infos
            )

            if self.create_sparse_graph:
                intrinsic_subgoals, intrinsic_trajectory_idx = self.filter_subgoals_based_on_sparsity_cond(
                    intrinsic_subgoals, intrinsic_trajectory_idx
                )

            if intrinsic_subgoals or extrinsic_subgoals:
                new_events = self.convert_discovered_goals_to_salient_events(
                    extrinsic_subgoals+intrinsic_subgoals
                )
        
                if self.make_off_policy_update:
                    for i in intrinsic_trajectory_idx + extrinsic_trajectory_idx:
                        pfrl_observations = self.dopamine2pfrl(exploration_inits[i], exploration_observations[i])
                        exploration_trajectory = self.create_exploration_trajectory(pfrl_observations,
                                                                                    exploration_actions[i],
                                                                                    exploration_extrinsic_rewards[i],
                                                                                    exploration_visited_infos[i],
                                                                                    new_events)
                        self.off_policy_update(exploration_trajectory)

                if self.enable_rnd_logging:
                    self.log_rnd_progress(intrinsic_subgoals, extrinsic_subgoals, new_events, episode)
        
        # Now that we have created new `self.salient_events`, update their distances to other old nodes
        if self.use_empirical_distances:
            self.dsg_agent.update_empirical_distance_estimates(
                full_trajectories,
                self.salient_events
            )

    def exploration_rollout(self, state, episode):
        assert isinstance(state, atari_wrappers.LazyFrames)

        # convert LazyFrame to np array for dopamine
        initial_state = np.asarray(state)
        initial_state = np.reshape(initial_state, self.rnd_agent._agent.state.shape)

        observations, actions, rewards, intrinsic_rewards, visited_infos = self.rnd_agent.rollout(initial_state)

        self.rnd_extrinsic_rewards.append(rewards)
        self.rnd_intrinsic_rewards.append(intrinsic_rewards)
        print(f"[RND Rollout] Reward: {rewards.sum()}\tIntrinsicReward: {intrinsic_rewards.sum()}")

        return initial_state, observations, actions, rewards, visited_infos
        
    def dopamine2pfrl(self, init_state, observations):
        """ Convert a dopamine trajectory into one that can be consumed by pfrl. """
        
        frame_stack = 4
        pfrl_observations = []
        s0 = init_state.transpose(3, 1, 2, 0)  # (1, 84, 84, 4) -> (4, 84, 84, 1)
        assert s0.shape == (4, 84, 84, 1), s0.shape
        padded_observations = list(s0) + list(observations)

        for i in range(len(padded_observations)-3):
            frames = padded_observations[i : i+frame_stack]
            lazy_frames = self.stack_to_lz(frames)
            pfrl_observations.append(lazy_frames)

        return pfrl_observations

    def create_exploration_trajectory(self, observations, actions, rewards, infos, new_events):
        exploration_trajectory = []
        start_observations = observations[:-1]
        next_observations = observations[1:]

        assert len(start_observations) == len(next_observations) == len(actions) == len(rewards) == len(infos)

        # Convert trajectory into a list of (s, a, r, s', done, reset, info) tuples
        # We are going to assume 0 rewards for all transitions for now, later in `off_policy_update`,
        # we will give the final transition a positive terminal reward
        for o, a, op, info in zip(start_observations, actions, next_observations, infos):
            exploration_trajectory.append((o, a, 0., op, False, False, info))

            # Truncate the traj when we hit the newly created salient event
            if any([event(info) for event in new_events]):
                print(f"Final info of the off-policy trajectory is {info}")
                print(f"Going to make off-policy update on {len(exploration_trajectory)} transitions.")
                break

        return exploration_trajectory

    def off_policy_update(self, exploration_trajectory):
        """ Take successful RND trajectory and make an off-policy update to the exploitation policy. """
        final_transition = exploration_trajectory[-1]
        reached_goal = final_transition[3]
        assert isinstance(reached_goal, atari_wrappers.LazyFrames), type(reached_goal)
        relabeled_trajectory = self.dsc_agent.global_option.positive_relabel(exploration_trajectory)
        self.dsc_agent.global_option.experience_replay(relabeled_trajectory, reached_goal)

    def log_rnd_progress(self, intrinsic_subgoals, extrinsic_subgoals, added_events, episode):

        with open(self.rnd_log_filename, "wb+") as f:
            pickle.dump({
                "rint": self.rnd_intrinsic_rewards,
                "rext": self.rnd_extrinsic_rewards
            }, f)

        base_str = self.rnd_log_filename.split(".")[0]

        if len(extrinsic_subgoals) > 0:
            extrinsic_subgoal_filename = f"{base_str}_extrinsic_{self.n_extrinsic_subgoals}.pkl"

            with open(extrinsic_subgoal_filename, "wb+") as f:
                pickle.dump(extrinsic_subgoals, f)

            self.n_extrinsic_subgoals += 1

        if len(intrinsic_subgoals) > 0:
            intrinsic_subgoal_filename = f"{base_str}_intrinsic_{self.n_intrinsic_subgoals}.pkl"

            with open(intrinsic_subgoal_filename, "wb+") as f:
                pickle.dump(intrinsic_subgoals[0], f)

            self.n_intrinsic_subgoals += 1

        if len(added_events) > 0:
            for event in added_events:
                pos_label = tuple(event.target_pos[:2])
                salient_event_filename = f"{base_str}_event_{pos_label}.pkl"

                with open(salient_event_filename, "wb+") as f:
                    pickle.dump({
                        "obs": event.target_obs,
                        "info": event.target_info
                    }, f)

                logdir = f"plots/{self.dsc_agent.experiment_name}/{self.dsc_agent.seed}"
                img_filename = f"{logdir}/accepted_events/event_{pos_label}.png"
                plt.imshow(np.array(event.target_obs)[-1])
                plt.title(str(pos_label))
                plt.savefig(img_filename)
                plt.close()

        # self.rnd_agent.plot_value(episode=episode)

    def graph_consolidation_run_loop(self, episode, duration):
        
        self.graph_mode = "consolidation"
        
        for current_episode in range(episode, episode+duration):
            done = False
            reset = False
            state, info = self.env.reset()

            print("=" * 80); print(f"[Consolidation] Episode: {current_episode} Step: {self.env.T}"); print("=" * 80)

            while not done and not reset:
                pos = info['player_x'], info['player_y']
                event = self.select_goal_salient_event(state, info)

                self.create_skill_chains_if_needed(state, info, event)
                print(f"[Graph Consolidation] From {pos} to {event.target_pos}")
                state, info, done, reset, reached, traj = self.dsg_agent.run_loop(state=state,
                                                                            info=info,
                                                                            goal_salient_event=event,
                                                                            episode=current_episode,
                                                                            eval_mode=False)

                if self.use_empirical_distances:
                    self.dsg_agent.update_empirical_distance_estimates(
                        traj, self.salient_events
                    )

                if reached:
                    print(f"DSG successfully reached {event}")

            assert done or reset, f"{done, reset}"

            if current_episode > 0 and current_episode % 10 == 0:
                self.add_potential_edges_to_graph()
                [self.dsg_agent.modify_node_connections(o) for o in self.dsc_agent.mature_options]

            if current_episode > 0 and current_episode % 10 == 0:
                self.delete_potential_nodes_from_graph()

        return state, info

    # ---------------------------------------------------
    # Salient Event Selection
    # ---------------------------------------------------

    def select_goal_salient_event(self, state, info):
        """ Select goal node to target during graph consolidation. """
        if random.random() > self.goal_selection_epsilon:

            if self.goal_selection_criterion == "closest":
                selected_event = self._select_closest_unconnected_salient_event(state, info)

                if selected_event is not None:
                    print(f"[Closest] DSG selected event {selected_event}")
                    return selected_event

            if self.goal_selection_criterion == "random_unconnected":
                selected_event = self._select_random_unconnected_salient_event(state, info)

                if selected_event is not None:
                    print(f"[RandomUnconnected] DSG selected event {selected_event}")
                    return selected_event
            
            if self.goal_selection_criterion == "boltzmann_unconnected":
                selected_event = self._select_boltzmann_closest_salient_event(state, info)

                if selected_event is not None:
                    print(f"[BoltzmannClosest] DSG selected event: {selected_event}")
                    return selected_event

            if self.goal_selection_criterion == "competence":
                selected_event = self._select_competence_progress_salient_event(state, info, "unconnected")

                if selected_event is not None:
                    print(f"[{self.goal_selection_criterion}] DSG selectd event: {selected_event}")
                    return selected_event

        selected_event = self._select_competence_progress_salient_event(state, info, "connected")
        
        if selected_event is not None:
            print(f"[CompetenceConnected] DSG selected event: {selected_event}")
            return selected_event

        selected_event = self._randomly_select_salient_event(state, info)
        print(f"[Random] DSG selected event {selected_event}")
        return selected_event

    def _randomly_select_salient_event(self, state, info):
        num_tries = 0
        target_event = None

        while target_event is None and num_tries < 100 and len(self.salient_events) > 0:
            target_event = random.choice(self.salient_events)

            # If you are already at the target_event, then re-sample
            if target_event(info):
                target_event = None

            num_tries += 1

        if target_event is not None:
            print(f"[Random] Deep skill graphs target event: {target_event}")

        return target_event

    def _select_random_unconnected_salient_event(self, state, info):
        unconnected_events = self._get_unconnected_events(state, info)

        if len(unconnected_events) > 0:
            return random.choice(unconnected_events)

    def _select_closest_unconnected_salient_event(self, state, info):
        unconnected_events = self._get_unconnected_events(state, info)
        current_events = self.get_corresponding_salient_events(state, info)

        closest_pair = self.dsg_agent.get_closest_pair_of_vertices(
            current_events,
            unconnected_events,
            metric="vf"
        )

        if closest_pair is not None:
            return closest_pair[1]

    def _select_boltzmann_closest_salient_event(self, state, info):
        current_events = self.get_corresponding_salient_events(state, info)
        unconnected_events = self._get_unconnected_events(state, info)

        if current_events and unconnected_events:

            if len(unconnected_events) == 1:
                return unconnected_events[0]

            distance_matrix = self.dsg_agent.get_distance_matrix(
                current_events,
                unconnected_events, 
                metric="empirical"
            )

            scores = 1. / distance_matrix
            
            # Higher the temperature, wider the output distribution
            probabilities = special.softmax(
                scores / 0.02,
                axis=1
            )

            print(f"Unconnected Events: {unconnected_events} | Probs: {probabilities}")

            selected_event = np.random.choice(unconnected_events, size=1, p=probabilities.squeeze())[0]
            assert isinstance(selected_event, SalientEvent), type(selected_event)

            return selected_event
    
    def _select_competence_progress_salient_event(self, state, info, select_among):
        """ Use competence progress to determine which salient event to target next. """
        # TODO: Currently this is agnostic to the start state of the learning curve
        assert select_among in ("connected", "unconnected"), select_among

        scores = []
        window_size = 50  # episodes  # TODO: How to pick this?

        if select_among == "unconnected":
            candidate_events = self._get_unconnected_events(state, info)
        else:
            candidate_events = self._get_connected_events(state, info)

        if len(candidate_events) > 1:
        
            for event in candidate_events:
                key = tuple(event.target_pos)
                if key in self.dsg_agent.gc_successes and \
                    len(self.dsg_agent.gc_successes[key]) >= (2 * window_size):
                    n_recent_successes = sum(self.dsg_agent.gc_successes[key][-window_size:])
                    n_past_successes = sum(self.dsg_agent.gc_successes[key][-2*window_size:window_size])
                    score = abs(n_recent_successes - n_past_successes) / window_size
                else:  # Optimistic initialization
                    score = 1.
                
                scores.append(score)

            scores = np.array(scores)
            probabilities = special.softmax(scores / self.boltzmann_temperature)

            selected_event = np.random.choice(candidate_events, size=1, p=probabilities.squeeze())[0]
            assert isinstance(selected_event, SalientEvent), type(selected_event)

            print(f"Candidate events: {candidate_events} | Probabilities: {probabilities}")

            return selected_event

        if len(candidate_events) == 1:
            return candidate_events[0]

    def _get_unconnected_events(self, state, info):
        candidate_events = [event for event in self.salient_events if not event(info)]
        unconnected_events = self.dsg_agent.planner.get_unconnected_nodes(state, info, candidate_events)
        return unconnected_events

    def _get_connected_events(self, state, info):
        candidate_events = [event for event in self.salient_events if not event(info)]
        connected_events = [event for event in candidate_events if self.dsg_agent.planner.does_path_exist(state, info, event)]
        return connected_events

    # ---------------------------------------------------
    # Manage skill chains
    # ---------------------------------------------------

    def create_skill_chains_if_needed(self, state, info, goal_salient_event):
        current_salient_events = self.get_corresponding_salient_events(state, info)

        for init_event in current_salient_events:
            if not self.dsg_agent.planner.does_path_exist(state, info, goal_salient_event) and \
                not self.is_path_under_construction(state, info, init_event, goal_salient_event):

                closest_event_pair = self.dsg_agent.choose_closest_source_target_vertex_pair(
                    state, info, goal_salient_event, choose_among_events=True
                )

                init, target = init_event, goal_salient_event

                if closest_event_pair is not None:
                    init, target = closest_event_pair[0], closest_event_pair[1]

                if not self.is_path_under_construction(state, info, init, target):
                    print(f"[DeepSkillGraphsAgent] Creating chain from {init} -> {target}")
                    self.dsc_agent.create_new_chain(init_event=init, target_event=target)

    def is_path_under_construction(self, state, info, start_event, goal_event):
        assert isinstance(state, atari_wrappers.LazyFrames), f"{type(state)}"
        assert isinstance(start_event, SalientEvent), f"{type(start_event)}"
        assert isinstance(goal_event, SalientEvent), f"{type(goal_event)}"

        match = lambda c: c.init_salient_event == start_event and c.target_salient_event == goal_event

        if any([match(c) for c in self.dsc_agent.chains]):
            return True

        current_salient_events = [event for event in self.salient_events if event(info)]
        under_construction = any([self.does_path_exist_in_optimistic_graph(event, goal_event)
                                  for event in current_salient_events])
        return under_construction

    def does_path_exist_in_optimistic_graph(self, node1, node2):

        # Create a lightweight copy of the plan-graph
        optimistic_graph = nx.DiGraph()
        for edge in self.dsg_agent.planner.plan_graph.edges:
            optimistic_graph.add_edge(str(edge[0]), str(edge[1]))

        # Pretend as if all unfinished chains have been learned and add them to the new graph
        unfinished_chains = [chain for chain in self.dsc_agent.chains if not chain.is_chain_completed()]
        for chain in unfinished_chains:
            optimistic_graph.add_edge(str(chain.init_salient_event), str(chain.target_salient_event))

        # Return if there is a path in this "optimistic" graph
        if str(node1) not in optimistic_graph or str(node2) not in optimistic_graph:
            return False

        return shortest_paths.has_path(optimistic_graph, str(node1), str(node2))

    # ---------------------------------------------------
    # Manage salient events
    # ---------------------------------------------------

    def get_corresponding_salient_events(self, state, info):
        # TODO: Need this if we remove init_event from salient_events
        # salient_events = [self.init_salient_event] + self.salient_events
        return [event for event in self.salient_events if event(info)]

    # ---------------------------------------------------
    # Manage skill graph
    # ---------------------------------------------------

    def add_potential_edges_to_graph(self):
        t0 = time.time()
        [self.dsg_agent.add_potential_edges(o) for o in self.dsg_agent.planner.option_nodes]
        print(f"Took {time.time() - t0}s to add potential edges.")

    def delete_potential_nodes_from_graph(self):
        """ Delete nodes that are impossible to re-trigger. """
        
        # Find bad events
        bad_events = []
        for event in self.salient_events:
            key = event.target_pos[0], event.target_pos[1], event.target_info["room_number"]
            if key in self.env.imaginary_ladder_locations:
                bad_events.append(event)

        # Log deleted events
        if bad_events:
            print("*" * 80)
            print(f"Deleting {bad_events}")
            print("*" * 80)
            self.deleted_events.extend(bad_events)

        # Delete the bad salient events from the graph
        for event in bad_events:
            print(f"Removing {event} from the list of salient events.")
            self.salient_events.remove(event)
            self.dsg_agent.salient_events.remove(event)
        
        # for chain in self.dsc_agent.chains:
        #     if chain.target_salient_event in bad_events:
        #         print(f"Removing {chain} targeting {chain.target_salient_event}")
        #         self.dsc_agent.chains.remove(chain)
        
        # for option in self.dsc_agent.new_options:
        #     if option.target_salient_event in bad_events:
        #         print(f"Removing {option} targeting {option.target_salient_event}")
        #         self.dsc_agent.new_options.remove(option)

    # ---------------------------------------------------
    # Skill graph expansion
    # ---------------------------------------------------

    @staticmethod
    def stack_to_lz(frames):
        """ Convert a list of dopamine frames to a LazyFrames object. """
        def _safe_transpose(x):  # Avoid transpose and use squeezes instead
            x = x.squeeze(2)
            return x[np.newaxis, ...]

        if len(frames) < 4:
            n_pad_frames = 4 - len(frames)
            pad_frames = [frames[0] for _ in range(n_pad_frames)]
            frames = pad_frames + frames
            assert len(frames) == 4, len(frames)
        
        if frames[0].shape == (84, 84, 1):
            frames = [_safe_transpose(frame) for frame in frames]

        return atari_wrappers.LazyFrames(frames, stack_axis=0)

    def extract_subgoals(self, observations, infos, extrinsic_rewards):

        def extract_subgoals_from_ext_rewards(obs, info, rewards):
            indexes = [i for i, r in enumerate(rewards) if r > 0]
            if len(indexes) > 0:
                states = [self.stack_to_lz(obs[max(0, i-3):min(i+1, len(obs))]) for i in indexes]
                selected_infos = [info[i] for i in indexes]
                positive_rewards = [rewards[i] for i in indexes]
                return list(zip(states, selected_infos, positive_rewards))
            return []

        def extract_subgoals_from_int_rewards(obs, info):
            r_int = self.rnd_agent.reward_function(obs)
            i = r_int.argmax()
            state = self.stack_to_lz(obs[max(0, i-3):min(i+1, len(obs))])
            return [(state, info[i], r_int[i])]

        subgoals1 = extract_subgoals_from_ext_rewards(observations, infos, extrinsic_rewards)
        subgoals2 = extract_subgoals_from_int_rewards(observations, infos)

        return subgoals1, subgoals2

    def score_exploration_trajectory(self, observations, rewards, infos):
        """ Given a dopamine trajectory, return the following:
        1. The original trajectory
        2. Intrinsic score of the trajectory
        3. Transition corresponding to the highest intrinsic score
        4. Transitions corresponding to the positive extrinsic rewards
        """
        def frame2state(traj, idx):
            """ Convert a dopamine obs to a pfrl LazyFrames object. """
            return self.stack_to_lz(
                traj[
                    max(0, idx - 3) : min(idx + 1, len(traj))
                ]
            )

        f_observations, f_rewards, f_infos = self.filter_exploration_trajectory(observations, rewards, infos)

        if len(f_observations) > 0:

            intrinsic_rewards = self.rnd_agent.reward_function(f_observations)
            
            i = intrinsic_rewards.argmax()
            intrinsic_score = intrinsic_rewards.max()

            # TODO: we should be indexing into the unfiltered observations
            best_intrinsic_state = frame2state(f_observations, i)
            best_intrinsic_sir_triple = best_intrinsic_state, f_infos[i], intrinsic_score

            pos_extrinsic_idx = [j for j in range(len(f_rewards)) if f_rewards[j] > 0]
            best_extrinsic_sir_triples = [(frame2state(f_observations, j), f_infos[j],\
                                        f_rewards[j]) for j in pos_extrinsic_idx]

            return intrinsic_score, best_intrinsic_sir_triple, best_extrinsic_sir_triples
        
        return -np.inf, [], []

    def new_subgoal_extractor(self, observations, rewards, infos):
        """ Given unfiltered dopamine trajectories (represented as lists of lists), 
        return the following:
        1. (s, i, r) triple corresponding to the highest intrinsic reward
        2. (s, i, r) triple corresponding to positive extrinsic rewards
        3. Trajectory index corresponding to the highest intrinsic reward
        4. Trajectory indices corresponding to positive extrinsic rewards
        """
        extrinsic_triples = []
        best_intrinsic_score = -np.inf
        best_intrinsic_triple = None
        best_intrinsic_trajectory_idx = None
        extrinsic_trajectory_idx = []

        for i, (obs_traj, reward_traj, info_traj) in enumerate(zip(observations, rewards, infos)):
            intrinsic_score, best_int_triple, ext_triples = self.score_exploration_trajectory(
                obs_traj, reward_traj, info_traj
            )
            
            if intrinsic_score > best_intrinsic_score:
                best_intrinsic_score = intrinsic_score
                best_intrinsic_triple = best_int_triple
                best_intrinsic_trajectory_idx = i
            
            if len(ext_triples) > 0:
                extrinsic_triples.extend(ext_triples)
                extrinsic_trajectory_idx.append(i)

        intrinsic_triples = [best_intrinsic_triple] if best_intrinsic_triple else []

        return intrinsic_triples, extrinsic_triples, \
               [best_intrinsic_trajectory_idx], extrinsic_trajectory_idx

    def filter_subgoals_based_on_sparsity_cond(self, sir_triples, trajectory_idx):
        filtered_triples = []
        filtered_trajectory_idx = []
        
        sess = self.rnd_agent._agent.intrinsic_model._sess
        rnd_mean = sess.run(self.rnd_agent._agent.intrinsic_model.reward_mean)
        rnd_std = sess.run(self.rnd_agent._agent.intrinsic_model.reward_std)

        for sir, idx in zip(sir_triples, trajectory_idx):
            intrinsic_reward = sir[2]
            assert isinstance(intrinsic_reward, float), intrinsic_reward
            if intrinsic_reward > rnd_mean + (6. * rnd_std):
                filtered_triples.append(sir)
                filtered_trajectory_idx.append(idx)
        return filtered_triples, filtered_trajectory_idx

    def should_reject_new_event(self, info, regions):
        def get_satisfied_regions(i):
            return [region for region in regions if regions[region](i)]

        # Regions that the new salient event is in
        new_satisfied_regions = get_satisfied_regions(info)

        # Regions that existing salient events are in
        old_satisfied_regions = [get_satisfied_regions(e.target_info) for e in self.salient_events]

        # A single salient event is expected to be in a single region
        if len(new_satisfied_regions) > 1:
            ipdb.set_trace()
        
        # If the salient event is not in any of the acceptable regions, reject it
        if len(new_satisfied_regions) == 0:
            return True
        
        # If the salient event is in a region where we already have another event, reject it
        satisfied_region = new_satisfied_regions[0]
        return satisfied_region in itertools.chain.from_iterable(old_satisfied_regions)

    def filter_exploration_trajectory(self, observations, rewards, infos):
        """ Given some observations, return the ones that pass our filtering rules. """

        def is_inside_another_event(info):
            return any([event(info) for event in self.salient_events])

        def is_close_to_another_event(info):
            distances = [event.distance(info) for event in self.salient_events]
            return any([distance < 11. for distance in distances])

        def get_options_for_filtering():
            # Only considering root options because their classifiers tend to be tighter
            if self.dsc_agent.use_pos_for_init:
                root_options = self.dsc_agent.mature_options
            else:
                root_options = [option for option in self.dsc_agent.mature_options if option.parent is None]

            # Only consider options that don't target the start state salient event
            root_options = [option for option in root_options if option.target_salient_event != self.init_salient_event]
            
            # Only consider options which don't tend to predict true everywhere
            fp_cond = lambda clf: (clf.get_false_positive_rate() < 0.7).all()
            filtered_options = [option for option in root_options if fp_cond(option.initiation_classifier)]
            return filtered_options

        def apply_option_filtering_cond(states, rewards, infos):
            t0 = time.time()
            start_n_states = len(states)

            if start_n_states == 0:
                return [], [], []

            options = get_options_for_filtering()

            states = np.array(states).squeeze(3)  # (N, 84, 84, 1) --> (N, 84, 84)
            assert states.shape == (start_n_states, 84, 84), states.shape
            
            for option in options:

                if states.shape[0] > 0:
                    inits = option.initiation_classifier.batched_pessimistic_predict(states)
                    inits = inits.squeeze(1)  # (N, 1) --> (N,)
                    print(f"Mean init: {inits.mean()} for {option}")
                    accepted_idx = np.argwhere(inits != 1).squeeze(1)  # (N, 1) --> (M,)
                    assert inits.shape == (states.shape[0],), inits.shape

                    if len(accepted_idx) > 0:
                        states = states[accepted_idx, ...]
                        rewards = [rewards[i] for i in accepted_idx]
                        infos = [infos[i] for i in accepted_idx]
                    else: # If we ended up rejecting everything (M=0), return early
                        return [], [], []

            # (N, 84, 84) -> list of N frames of shape (1, 84, 84) (for stacking later)
            filtered_states = [state[np.newaxis, ...] for state in states]
            print(f"[option-filtering] Took {time.time()-t0}s to go from {start_n_states} --> {len(filtered_states)} states.")

            assert len(filtered_states) == len(rewards) == len(infos), ipdb.set_trace()
            return filtered_states, rewards, infos

        def should_reject(obs, info):
            return info["uncontrollable"] or \
                   info["buggy_state"] or\
                   is_inside_another_event(info) or \
                   is_close_to_another_event(info)
        
        if len(observations) > 3:
            first_pass_triples = [(obs, reward, info) for obs, reward, info in 
                                zip(observations, rewards, infos) if not should_reject(obs, info)]
            
            first_pass_observations = [triple[0] for triple in first_pass_triples]
            first_pass_rewards = [triple[1] for triple in first_pass_triples]
            first_pass_infos = [triple[2] for triple in first_pass_triples]

            return first_pass_observations, first_pass_rewards, first_pass_infos
            
            # if len(first_pass_observations) > 3:
                
            #     accepted_observations, accepted_rewards, accepted_infos = apply_option_filtering_cond(
            #         first_pass_observations, first_pass_rewards, first_pass_infos
            #     )

            #     return accepted_observations, accepted_rewards, accepted_infos
            
        return [], [], []

    def get_intrinsic_values(self, observations):
        assert isinstance(observations, np.ndarray)
        return self.rnd_agent.value_function(observations)

    def convert_discovered_goals_to_salient_events(self, discovered_goals):
        """ Convert a list of discovered goal states to salient events. """
        added_events = []
        for obs, info, reward in discovered_goals:
            event = SalientEvent(obs, info, tol=2.)
            print("Accepted New Salient Event: ", event)
            added_events.append(event)

            # Add the discovered event only if we are not in gc experiment mode
            if len(self.predefined_events) == 0:
                self.add_salient_event(event)

        return added_events

    def add_salient_event(self, new_event):
        print("[DSGTrainer] Adding new SalientEvent ", new_event)
        self.salient_events.append(new_event)
        self.dsg_agent.salient_events.append(new_event)
