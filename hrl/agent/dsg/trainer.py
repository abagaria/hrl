import gym
import time
import random
import pickle
import ipdb
import numpy as np
import networkx as nx
import networkx.algorithms.shortest_paths as shortest_paths

from collections import defaultdict
from pfrl.wrappers import atari_wrappers
from .dsg import SkillGraphAgent
from ..dsc.dsc import RobustDSC
from hrl.salient_event.salient_event import SalientEvent
from hrl.agent.bonus_based_exploration.RND_Agent import RNDAgent
from hrl.agent.dsg.utils import visualize_graph_nodes_with_expansion_probabilities

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


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
        assert goal_selection_criterion in ("random", "closest")

        self.env = env
        self.dsc_agent = dsc
        self.dsg_agent = dsg
        self.rnd_agent = rnd
        self.expansion_freq = expansion_freq
        self.expansion_duration = expansion_duration
        self.init_salient_event = dsc.init_salient_event
        self.goal_selection_criterion = goal_selection_criterion
        self.disable_graph_expansion = disable_graph_expansion

        self.predefined_events = predefined_events
        self.generated_salient_events = predefined_events
        self.salient_events = predefined_events + [self.init_salient_event]

        for event in self.salient_events:
            self.dsg_agent.add_salient_event(event)

        # Map goal to success curve
        self.gc_successes = defaultdict(list)

        # Logging for exploration rollouts
        self.n_extrinsic_subgoals = 0
        self.n_intrinsic_subgoals = 0
        self.rnd_extrinsic_rewards = []
        self.rnd_intrinsic_rewards = []
        self.rnd_log_filename = rnd_log_filename
        self.enable_rnd_logging = enable_rnd_logging

    # ---------------------------------------------------
    # Run loops
    # ---------------------------------------------------

    def run_loop(self, start_episode, num_episodes):
        for episode in range(start_episode, start_episode + num_episodes):
            print("=" * 80)
            print(f"Episode: {episode} Step: {self.env.T}")
            print("=" * 80)
            if (not self.disable_graph_expansion) and (episode % self.expansion_freq == 0):
                self.graph_expansion_run_loop(episode, self.expansion_duration)
            else:
                self.graph_consolidation_run_loop(episode)

            t0 = time.time()
            with open(self.dsc_agent.log_file, "wb+") as f:
                pickle.dump(self.gc_successes, f)
            print(
                f"[Episode={episode}, Seed={self.dsc_agent.seed}] Took {time.time() - t0}s to save gc logs")

    def test_distance_metrics_run_loop(self, start_episode, num_episodes, plot_dir, test_freq=10):

        metrics = [
            "euclidean",
            "vf",
            "ucb"
        ]

        accuracies = {
            'episode': [],
            'euclidean': [],
            'vf': [],
            'ucb': []
        }

        base_plot_dir = plot_dir

        for current_episode in range(start_episode, start_episode + num_episodes, test_freq):

            # Run normal loop
            self.run_loop(current_episode, test_freq)

            #####################################################
            #               Run metric test                     #
            #####################################################

            # to get the accurate episode in the figure names we add test_freq to current episode
            print("[Testing] Running distance metric tests")

            episode = current_episode + test_freq

            accuracies['episode'].append(episode)

            for metric in metrics:
                plot_dir = os.path.join(base_plot_dir, metric)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir, exist_ok=True)

                predicted_y_values, true_y_values, x_values, accuracy = self.dsg_agent.test_distance_metrics(
                    self.salient_events, metric)

                labels = []
                for event in self.salient_events:
                    labels.append(str(event.target_pos) +
                                  " visits:" + str(len(event.effect_set)))

                accuracies[metric].append(accuracy)
                cm = confusion_matrix(true_y_values, predicted_y_values,
                                      normalize="true", labels=np.arange(len(self.salient_events)))
                cm_input = confusion_matrix(
                    x_values, predicted_y_values, labels=np.arange(len(self.salient_events)))

                fig_name = os.path.join(
                    plot_dir, 'metric-{}-episode-{}-truth-predicted.png'.format(metric, episode))

                DSGTrainer.plot_confusion_matrix(cm, fig_name, cmap=plt.cm.Blues, class_names=labels,
                                                 x_label="Predicted Closest Location", y_label="True Closest Location")
                fig_name = os.path.join(
                    plot_dir, 'metric-{}-episode-{}-input-output.png'.format(metric, episode))
                DSGTrainer.plot_confusion_matrix(
                    cm_input, fig_name, cmap=plt.cm.Blues, class_names=labels, x_label="Source Location", y_label="Destination Location")

                if metric == metrics[0] and current_episode == 0:
                    fig_name = os.path.join(
                        base_plot_dir, 'metric-lut-input-output.png')
                    cm_input = confusion_matrix(
                        x_values, true_y_values, labels=np.arange(len(self.salient_events)))
                    DSGTrainer.plot_confusion_matrix(
                        cm_input, fig_name, cmap=plt.cm.Blues, class_names=labels, x_label="Source Location", y_label="Destination Location")

            if len(accuracies['episode']) > 20:
                self.make_accuracy_plot(metrics, accuracies, base_plot_dir, test_freq)

    @staticmethod
    def make_accuracy_plot(metrics, accuracy_dict, base_plot_dir, test_freq):
        def moving_average(a, n):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n-1:] / n

        def smoothen_data(scores, n=15):
            smoothened_cols = scores.shape[1] - n + 1
            smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
            for i in range(scores.shape[0]):
                smoothened_data[i, :] = moving_average(scores[i, :], n=n)
            return smoothened_data

        accuracy_array = []

        for metric in metrics:
            accuracy_array.append(accuracy_dict[metric])

        smooth_accuracy_array = smoothen_data(np.array(accuracy_array))

        plt.clf()

        for i, metric in enumerate(metrics):
            smooth_accuracy = smooth_accuracy_array[i, :]
            plt.plot(range(len(smooth_accuracy)), smooth_accuracy, "*-", label=metric)

        plt.legend()
        plt.xlabel(f'Episode (x{test_freq})')
        plt.ylabel('Accuracy')
        fig_name = os.path.join(base_plot_dir, 'accuracy.png')
        plt.savefig(fig_name)
        plt.close('all')

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, save_fig_name, class_names=None, cmap=plt.cm.Blues, title=None, x_label=None, y_label=None):

        confusion_matrix = np.flip(confusion_matrix, axis=0)

        if class_names is not None:
            assert len(class_names) == confusion_matrix.shape[0]
        plt.clf()
        plt.imshow(confusion_matrix, cmap=cmap, interpolation='nearest')
        if title is not None:
            plt.title(title)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            # plt.yticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names[::-1])
        cmap_min, cmap_max = cmap(0), cmap(1.0)
        thresh = (confusion_matrix.max()+confusion_matrix.min()) / 2.0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[0]):
                color = cmap_max if (
                    confusion_matrix[i, j] < thresh) else cmap_min
                plt.text(j, i, "{:.2f}".format(
                    confusion_matrix[i, j]), ha="center", va="center", color=color)

        # plt.show()
        plt.savefig(save_fig_name, bbox_inches='tight')
        plt.close('all')

    def graph_expansion_run_loop(self, start_episode, num_episodes):
        intrinsic_subgoals = []
        extrinsic_subgoals = []

        for episode in range(start_episode, start_episode + num_episodes):
            state, info = self.env.reset()
            expansion_node = self.dsg_agent.get_node_to_expand()
            print(f"[Episode={episode}] Attempting to expand {expansion_node}")

            state, info, done, reset, reached = self.dsg_agent.run_loop(state=state,
                                                                        info=info,
                                                                        goal_salient_event=expansion_node,
                                                                        episode=episode,
                                                                        eval_mode=False)

            if reached and not done and not reset:
                observations, rewards, intrinsic_rewards, visited_positions = self.exploration_rollout(
                    state)
                print(
                    f"[RND Rollout] Episode {episode}\tSum Reward: {rewards.sum()}\tSum RewardInt: {intrinsic_rewards.sum()}")

                extrinsic_subgoal_proposals, intrinsic_subgoal_proposals = self.extract_subgoals(
                    observations,
                    visited_positions,
                    rewards,
                )
                if len(extrinsic_subgoal_proposals) > 0:
                    extrinsic_subgoals.extend(extrinsic_subgoal_proposals)

                if len(intrinsic_subgoal_proposals) > 0:
                    intrinsic_subgoals.extend(intrinsic_subgoal_proposals)

                self.rnd_extrinsic_rewards.append(rewards)
                self.rnd_intrinsic_rewards.append(intrinsic_rewards)

        if self.enable_rnd_logging:
            self.log_rnd_progress(intrinsic_subgoals,
                                  extrinsic_subgoals, episode)

    def exploration_rollout(self, state):
        assert isinstance(state, atari_wrappers.LazyFrames)

        # convert LazyFrame to np array for dopamine
        initial_state = np.asarray(state)
        initial_state = np.reshape(state, self.rnd_agent._agent.state.shape)

        observations, rewards, intrinsic_rewards, visited_positions = self.rnd_agent.rollout(
            initial_state=initial_state)

        return observations, rewards, intrinsic_rewards, visited_positions

    def log_rnd_progress(self, intrinsic_subgoals, extrinsic_subgoals, episode):
        best_spr_triple = self.extract_best_intrinsic_subgoal(
            intrinsic_subgoals)

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
                pickle.dump(best_spr_triple, f)

            self.n_intrinsic_subgoals += 1

        visualize_graph_nodes_with_expansion_probabilities(self.dsg_agent,
                                                           episode,
                                                           self.dsc_agent.experiment_name,
                                                           self.dsc_agent.seed)

        # self.rnd_agent.plot_value(episode=episode)

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
        """ Select goal node to target during graph consolidation. """

        if self.goal_selection_criterion == "closest":
            selected_event = self._select_closest_unconnected_salient_event(
                state, info)

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
            unconnected_events,
            metric="vf"
        )

        if closest_pair is not None:
            return closest_pair[1]

    def _get_unconnected_events(self, state, info):
        candidate_events = [
            event for event in self.salient_events if not event(info)]
        unconnected_events = self.dsg_agent.planner.get_unconnected_nodes(
            state, info, candidate_events)
        return unconnected_events

    # ---------------------------------------------------
    # Manage skill chains
    # ---------------------------------------------------

    def create_skill_chains_if_needed(self, state, info, goal_salient_event):
        current_salient_events = self.get_corresponding_salient_events(
            state, info)

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
                    print(
                        f"[DeepSkillGraphsAgent] Creating chain from {init} -> {target}")
                    self.dsc_agent.create_new_chain(
                        init_event=init, target_event=target)

    def is_path_under_construction(self, state, info, start_event, goal_event):
        assert isinstance(state, atari_wrappers.LazyFrames), f"{type(state)}"
        assert isinstance(start_event, SalientEvent), f"{type(start_event)}"
        assert isinstance(goal_event, SalientEvent), f"{type(goal_event)}"

        def match(
            c): return c.init_salient_event == start_event and c.target_salient_event == goal_event
        if any([match(c) for c in self.dsc_agent.chains]):
            return True

        current_salient_events = [
            event for event in self.salient_events if event(info)]
        under_construction = any([self.does_path_exist_in_optimistic_graph(event, goal_event)
                                  for event in current_salient_events])
        return under_construction

    def does_path_exist_in_optimistic_graph(self, node1, node2):

        # Create a lightweight copy of the plan-graph
        optimistic_graph = nx.DiGraph()
        for edge in self.dsg_agent.planner.plan_graph.edges:
            optimistic_graph.add_edge(str(edge[0]), str(edge[1]))

        # Pretend as if all unfinished chains have been learned and add them to the new graph
        unfinished_chains = [
            chain for chain in self.dsc_agent.chains if not chain.is_chain_completed()]
        for chain in unfinished_chains:
            optimistic_graph.add_edge(
                str(chain.init_salient_event), str(chain.target_salient_event))

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
        [self.dsg_agent.add_potential_edges(
            o) for o in self.dsg_agent.planner.option_nodes]
        print(f"Took {time.time() - t0}s to add potential edges.")

    # ---------------------------------------------------
    # Skill graph expansion
    # ---------------------------------------------------

    def extract_subgoals(self, observations, positions, extrinsic_rewards):
        def extract_subgoals_from_ext_rewards(obs, pos, rewards):
            sgs = []
            for x, p, r in zip(obs, pos, rewards):
                if r > 0:
                    sgs.append((x, p, r))
            return sgs

        def extract_subgoals_from_int_rewards(obs, pos):
            r_int = self.rnd_agent.reward_function(obs)
            i = r_int.argmax()
            return [(obs[i], pos[i], r_int[i])]

        subgoals1 = extract_subgoals_from_ext_rewards(
            observations, positions, extrinsic_rewards)
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

    def exploration_rollout(self, state):
        assert isinstance(state, atari_wrappers.LazyFrames)

        # convert LazyFrame to numpy array
        initial_state = np.asarray(state)
        # reshape state for rnd agent
        initial_state = np.reshape(state, self.rnd_agent._agent.state.shape)

        observations, rewards, intrinsic_rewards, visited_positions = self.rnd_agent.rollout(
            initial_state=initial_state)

        return observations, rewards, intrinsic_rewards, visited_positions
