import gym
import time
import random
import pickle
import networkx as nx
import networkx.algorithms.shortest_paths as shortest_paths

from collections import defaultdict
from pfrl.wrappers import atari_wrappers
from .dsg import SkillGraphAgent
from ..dsc.dsc import RobustDSC
from hrl.salient_event.salient_event import SalientEvent


class DSGTrainer:
    def __init__(self, env, dsc, dsg, 
                 expansion_freq, expansion_duration,
                 predefined_events=[]):
        assert isinstance(env, gym.Env)
        assert isinstance(dsc, RobustDSC)
        assert isinstance(dsg, SkillGraphAgent)

        self.env = env
        self.dsc_agent = dsc
        self.dsg_agent = dsg
        self.expansion_freq = expansion_freq
        self.expansion_duration = expansion_duration
        self.init_salient_event = dsc.init_salient_event
        
        self.predefined_events = predefined_events
        self.generated_salient_events = predefined_events
        self.salient_events = predefined_events + [self.init_salient_event]

        for event in self.salient_events:
            self.dsg_agent.add_salient_event(event)

        # Map goal to success curve
        self.gc_successes = defaultdict(list)

    # ---------------------------------------------------
    # Run loops 
    # ---------------------------------------------------

    def run_loop(self, start_episode, num_episodes):
        for episode in range(start_episode, start_episode + num_episodes):
            print("=" * 80); print(f"Episode: {episode} Step: {self.env.T}"); print("=" * 80)
            if episode % self.expansion_freq == 0:
                self.graph_expansion_run_loop(episode, self.expansion_duration)
            else:
                self.graph_consolidation_run_loop(episode)
            
            with open(self.dsc_agent.log_file, "wb+") as f:
                pickle.dump(self.gc_successes, f)

    def graph_expansion_run_loop(self, start_episode, num_episodes):
        for episode in range(start_episode, start_episode + num_episodes):
            pass

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
            
            # Log success or failure for the pursued goal
            self.gc_successes[tuple(event.target_pos)].append(reached)

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
        selected_event = self._select_closest_unconnected_salient_event(state, info)

        if selected_event is not None:
            print(f"[Closest] DSG selected event {selected_event}")
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

    def _select_closest_unconnected_salient_event(self, state, info):
        unconnected_events = self._get_unconnected_events(state, info)
        current_events = self.get_corresponding_salient_events(state, info)
        
        closest_pair = self.dsg_agent.get_closest_pair_of_vertices(
            current_events,
            unconnected_events
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

    def create_salient_event(self, target_state, target_pos):
        salient_event = SalientEvent(target_state, target_pos)
        self.salient_events.append(salient_event)
        self.generated_salient_events(salient_event)
        self.dsg_agent.add_salient_event(salient_event)

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

