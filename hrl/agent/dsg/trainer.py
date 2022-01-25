import gym
import ipdb
import time
import random
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.shortest_paths as shortest_paths

from pfrl.wrappers import atari_wrappers
from .dsg import SkillGraphAgent
from ..dsc.dsc import RobustDSC
from hrl.agent.dsc.utils import pos_to_info
from hrl.salient_event.salient_event import SalientEvent
from hrl.agent.bonus_based_exploration.RND_Agent import RNDAgent
from hrl.agent.dsg.utils import visualize_graph_nodes_with_expansion_probabilities


class DSGTrainer:
    def __init__(self, env, dsc, dsg, rnd,
                 expansion_freq, expansion_duration,
                 rnd_log_filename,
                 goal_selection_criterion="random",
                 predefined_events=[],
                 enable_rnd_logging=False,
                 disable_graph_expansion=False):
        assert isinstance(env, gym.Env)
        assert isinstance(dsc, RobustDSC)
        assert isinstance(dsg, SkillGraphAgent)
        assert isinstance(rnd, RNDAgent)
        assert goal_selection_criterion in ("random", "closest", "random_unconnected")

        self.env = env
        self.dsc_agent = dsc
        self.dsg_agent = dsg
        self.rnd_agent = rnd
        self.expansion_freq = expansion_freq
        self.expansion_duration = expansion_duration
        self.init_salient_event = dsc.init_salient_event
        self.goal_selection_criterion = goal_selection_criterion

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

    # ---------------------------------------------------
    # Run loops
    # ---------------------------------------------------

    def run_loop(self, start_episode, num_episodes):
        for episode in range(start_episode, start_episode + num_episodes):
            print("=" * 80); print(f"Episode: {episode} Step: {self.env.T}"); print("=" * 80)

            if (not self.disable_graph_expansion) and (episode % self.expansion_freq == 0):
                self.graph_expansion_run_loop(episode, self.expansion_duration)
            else:
                self.graph_consolidation_run_loop(episode)

            t0 = time.time()
            with open(self.dsc_agent.log_file, "wb+") as f:
                pickle.dump(self.dsg_agent.gc_successes, f)
            print(f"[Episode={episode}, Seed={self.dsc_agent.seed}] Took {time.time() - t0}s to save gc logs")

    def graph_expansion_run_loop(self, start_episode, num_episodes):
        exploration_observations = []
        exploration_visited_positions = []
        exploration_extrinsic_rewards = []

        for episode in range(start_episode, start_episode + num_episodes):
            state, info = self.env.reset()
            expansion_node = self.dsg_agent.get_node_to_expand()
            print(f"[Episode={episode}] Attempting to expand {expansion_node}")

            # Use the planner to get to the expansion node
            state, info, done, reset, reached = self.dsg_agent.run_loop(state=state,
                                                                        info=info,
                                                                        goal_salient_event=expansion_node,
                                                                        episode=episode,
                                                                        eval_mode=False)

            if reached and not done and not reset:
                observations, rewards, visited_positions = self.exploration_rollout(state, episode)

                exploration_observations.extend(observations)
                exploration_extrinsic_rewards.extend(rewards)
                exploration_visited_positions.extend(visited_positions)

        # Examine *all* exploration trajectories and extract potential salient events
        if len(exploration_observations) > 0:
            extrinsic_subgoals, intrinsic_subgoals = self.extract_subgoals(
                exploration_observations,
                exploration_visited_positions,
                exploration_extrinsic_rewards,
            )

            chosen_intrinsic_goal = self.extract_best_intrinsic_subgoal(intrinsic_subgoals)
            new_events = self.extract_salient_events(extrinsic_subgoals+[chosen_intrinsic_goal])

            if self.enable_rnd_logging:
                self.log_rnd_progress([chosen_intrinsic_goal], extrinsic_subgoals, new_events, episode)

    def exploration_rollout(self, state, episode):
        assert isinstance(state, atari_wrappers.LazyFrames)

        # convert LazyFrame to np array for dopamine
        initial_state = np.asarray(state)
        initial_state = np.reshape(state, self.rnd_agent._agent.state.shape)

        observations, rewards, intrinsic_rewards, visited_positions = self.rnd_agent.rollout(initial_state)

        self.rnd_extrinsic_rewards.append(rewards)
        self.rnd_intrinsic_rewards.append(intrinsic_rewards)
        print(f"[RND Rollout] Episode {episode}\tReward: {rewards.sum()}\tIntrinsicReward: {intrinsic_rewards.sum()}")

        return observations, rewards, visited_positions

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
                        "pos": event.target_pos
                    }, f)

                img_filename = f"plots/{self.dsc_agent.experiment_name}/{self.dsc_agent.seed}/event_{pos_label}.png"
                plt.imshow(np.array(event.target_obs)[-1])
                plt.title(str(pos_label))
                plt.savefig(img_filename)
                plt.close()

        visualize_graph_nodes_with_expansion_probabilities(self.dsg_agent,
                                                           episode,
                                                           self.dsc_agent.experiment_name,
                                                           self.dsc_agent.seed)

        self.rnd_agent.plot_value(episode=episode)

    def graph_consolidation_run_loop(self, episode):
        done = False
        reset = False
        state, info = self.env.reset()

        while not done and not reset:
            pos = info['player_x'], info['player_y']
            event = self.select_goal_salient_event(state, info)

            self.create_skill_chains_if_needed(state, info, event)
            print(f"[Graph Consolidation] From {pos} to {event.target_pos}")
            state, info, done, reset, reached = self.dsg_agent.run_loop(state=state,
                                                                        info=info,
                                                                        goal_salient_event=event,
                                                                        episode=episode,
                                                                        eval_mode=False)

            if reached:
                print(f"DSG successfully reached {event}")

        assert done or reset, f"{done, reset}"

        if episode > 0 and episode % 10 == 0:
            self.add_potential_edges_to_graph()

        return state, info

    # ---------------------------------------------------
    # Salient Event Selection
    # ---------------------------------------------------

    def select_goal_salient_event(self, state, info):
        """ Select goal node to target during graph consolidation. """

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

    def _get_unconnected_events(self, state, info):
        candidate_events = [event for event in self.salient_events if not event(info)]
        unconnected_events = self.dsg_agent.planner.get_unconnected_nodes(state, info, candidate_events)
        return unconnected_events

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

    # ---------------------------------------------------
    # Skill graph expansion
    # ---------------------------------------------------

    def extract_subgoals(self, observations, positions, extrinsic_rewards):
        def stack_to_lz(frames):
            """ Convert a list of frames to a LazyFrames object. """
            frames = [frame.transpose(2, 0, 1) for frame in frames]
            return atari_wrappers.LazyFrames(frames, stack_axis=0)

        def extract_subgoals_from_ext_rewards(obs, pos, rewards):
            indexes = [i for i, r in enumerate(rewards) if r > 0]
            if len(indexes) > 0:
                states = [stack_to_lz(obs[i-3:i+1]) for i in indexes]
                selected_positions = [pos[i] for i in indexes]
                positive_rewards = [rewards[i] for i in indexes]
                return list(zip(states, selected_positions, positive_rewards))
            return []

        def extract_subgoals_from_int_rewards(obs, pos):
            r_int = self.rnd_agent.reward_function(obs)
            i = r_int.argmax()
            state = stack_to_lz(obs[i-3:i+1])
            return [(state, pos[i], r_int[i])]

        subgoals1 = extract_subgoals_from_ext_rewards(observations, positions, extrinsic_rewards)
        subgoals2 = extract_subgoals_from_int_rewards(observations, positions)

        return subgoals1, subgoals2

    def get_intrinsic_values(self, observations):
        assert isinstance(observations, np.ndarray)
        return self.rnd_agent.value_function(observations)

    @staticmethod
    def extract_best_intrinsic_subgoal(s_r_pairs):
        best_obs = None
        best_pos = None
        max_intrinsic_reward = -np.inf

        for obs, position, reward in s_r_pairs:

            if reward > max_intrinsic_reward:
                best_obs = obs
                best_pos = position
                max_intrinsic_reward = reward

        return best_obs, best_pos, max_intrinsic_reward

    def extract_salient_events(self, discovered_goals):
        """ Convert a list of discovered goal states to salient events. """
        added_events = []
        for obs, info_tuple, reward in discovered_goals:
            pos = np.array([info_tuple[0], info_tuple[1]])
            event = SalientEvent(obs, pos, tol=2.)
            if not self.should_reject_new_event(event, info_tuple):
                print("Accepted New Salient Event: ", event)
                added_events.append(event)
                self.add_salient_event(event)
        return added_events

    def should_reject_new_event(self, salient_event, info_tuple):
        """ Use heuristics to reject some new salient events. """
        assert isinstance(info_tuple, tuple), info_tuple
        assert isinstance(salient_event, SalientEvent), salient_event

        def satisfies_existing_event(s, p):
            return any([event(p) for event in self.salient_events])

        def satisfies_existing_option(s, p):
            return any([o.pessimistic_is_init_true(s, pos_to_info(p)) for o in self.dsc_agent.mature_options])

        def satisfies_death_condition(i):
            falling = i[-1]
            return falling

        target_state = salient_event.target_obs
        target_pos = salient_event.target_pos

        return satisfies_existing_event(target_state, target_pos) or \
            satisfies_existing_option(target_state, target_pos) or \
            satisfies_death_condition(info_tuple)

    def add_salient_event(self, new_event):
        print("[DSGTrainer] Adding new SalientEvent ", new_event)
        self.salient_events.append(new_event)
        self.dsg_agent.salient_events.append(new_event)
