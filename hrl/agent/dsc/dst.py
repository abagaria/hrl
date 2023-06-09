import os
import ipdb
import torch
import pickle
import argparse
import numpy as np
from copy import deepcopy
from functools import reduce
from collections import deque

from hrl.agent.dsc.utils import *
from hrl.agent.dsc.MBOptionClass import ModelBasedOption


class RobustDST(object):
    def __init__(self, mdp, warmup_episodes, gestation_period, buffer_length,
                 use_vf, use_global_vf, use_model, lr_a, lr_c,
                 max_steps, use_diverse_starts, use_dense_rewards, experiment_name,
                 logging_freq, evaluation_freq, device, seed, multithread_mpc,
                 generate_init_gif, max_num_children):
        self.seed = seed
        self.device = device
        self.use_vf = use_vf
        self.use_model = use_model
        self.buffer_length = buffer_length
        self.use_global_vf = use_global_vf
        self.max_steps = max_steps
        self.logging_freq = logging_freq
        self.evaluation_freq = evaluation_freq
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards
        self.generate_init_gif = generate_init_gif
        self.max_num_children = max_num_children

        self.gestation_period = gestation_period

        self.lr_a = lr_a
        self.lr_c = lr_c

        self.mdp = mdp
        self.init_salient_event = self.mdp.get_start_state_salient_event()
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.skill_tree = SkillTree(options=[self.goal_option])
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.log = {}

    def act(self, state):
        option_names = self.skill_tree.traverse()
        options = [self.skill_tree.get_option(name) for name in option_names]

        # Use this list rather than `self.mature_options` to take advantage of the tree.bfs() traversal order
        # Checking if classifiers exist because it is possible that you are at init-done, but don't have clf yet
        cond = lambda o: o.get_training_phase() == "initiation_done" \
                         and o.optimistic_classifier is not None \
                         and o.pessimistic_classifier is not None

        options_in_maturation = [option for option in options if cond(option)]
        selected_option_and_subgoal = self._pick_among_mature_options(options_in_maturation, state)

        if selected_option_and_subgoal is not None:
            print(f"Skill-Trees::Act() chose {selected_option_and_subgoal[0]} in initiation-done phase")
            return selected_option_and_subgoal

        # Among options in gestation, goal-option gets highest priority
        if self.goal_option.get_training_phase() == "gestation" and self.goal_option.is_init_true(state):
            return self.goal_option, self.goal_option.get_goal_for_rollout()

        options_in_gestation = [option for option in options if option.get_training_phase() == "gestation"]
        if len(options_in_gestation) > 0:
            selected_option_and_subgoal = self._pick_among_options_in_gestation(options_in_gestation, state)
            if selected_option_and_subgoal is not None:
                selected_option = selected_option_and_subgoal[0]
                print(f"Skill-Trees::Act() chose {selected_option} (parent={selected_option.parent}) in gestation")
                return selected_option_and_subgoal

        new_option = self.create_new_option(state)
        if new_option is not None:
            return new_option, new_option.get_goal_for_rollout()

        return self.global_option, self.pick_subgoal_for_global_option(state)

    def _pick_among_options_in_gestation(self, options_in_gestation, state):
        nearest_option = self.find_nearest_option_in_tree(state)
        for child_option in nearest_option.children:  # type: ModelBasedOption
            if child_option in options_in_gestation:
                return child_option, child_option.get_goal_for_rollout()

    @staticmethod
    def _pick_among_mature_options(options_in_maturation, state):
        for option in options_in_maturation:  # type: ModelBasedOption
            if option.is_init_true(state):
                subgoal = option.get_goal_for_rollout()
                if not option.is_at_local_goal(state, subgoal):
                    return option, subgoal

    def create_new_option(self, state):
        nearest_option = self.find_nearest_option_in_tree(state)
        if self.should_create_child_option(nearest_option):
            new_option = self.create_child_option(parent_option=nearest_option)
            self.add_new_options([new_option])
            return new_option

    def random_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_done:
            state = deepcopy(self.mdp.cur_state)
            action = self.mdp.action_space.sample()
            next_state, reward, done, _ = self.mdp.step(action)
            if self.use_model:
                self.global_option.update_model(state, action, reward, next_state, done)
            step_number += 1
        return step_number

    def dsc_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_done:
            state = deepcopy(self.mdp.cur_state)

            selected_option, subgoal = self.act(state)

            transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal)
            self.manage_chain_after_rollout(selected_option)

            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes=300, num_steps=150, start_episode=0):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset(episode)

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode == self.warmup_episodes - 1 and self.use_model:
                self.learn_dynamics_model(epochs=50)
            elif episode >= self.warmup_episodes and self.use_model:
                self.learn_dynamics_model(epochs=5)

            self.log_success_metrics(episode)

        return per_episode_durations

    def learn_dynamics_model(self, epochs):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=epochs, batch_size=1024)
        for option in self.skill_tree.options:
            option.solver.model = self.global_option.solver.model

    def should_create_child_option(self, parent_option):
        assert isinstance(parent_option, ModelBasedOption)

        global_condition = not parent_option.pessimistic_is_init_true(self.mdp.init_state)
        local_condition = len(parent_option.children) < parent_option.max_num_children and \
                            parent_option.get_training_phase() == "initiation_done"

        return local_condition and global_condition

    def create_child_option(self, parent_option):
        assert isinstance(parent_option, ModelBasedOption)

        if not self.should_create_child_option(parent_option):
            return None

        prefix = "option_1" if parent_option.parent is None else parent_option.name
        name = prefix + f"_{len(parent_option.children)}"

        new_option = self.create_model_based_option(name, parent=parent_option)
        print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
        parent_option.children.append(new_option)

        return new_option

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

    def find_nearest_option_in_tree(self, state):
        if len(self.mature_options) == 1:
            return self.mature_options[0]

        if len(self.mature_options) > 1:
            samples = [option.sample_from_termination_region() for option in self.mature_options]
            states = np.repeat(state[None, ...], len(samples), axis=0)
            goals = np.array(samples)
            values = self.global_option.value_function(states, goals).squeeze().tolist()
            distances = [(option, -value) for option, value in zip(self.mature_options, values)]
            nearest_option = sorted(distances, key=lambda x: x[1])[0][0]  # type: ModelBasedOption
            return nearest_option

    def pick_subgoal_for_global_option(self, state):
        nearest_option = self.find_nearest_option_in_tree(state)
        sampled_goal = nearest_option.sample_from_initiation_region_fast_and_epsilon()

        if isinstance(sampled_goal, np.ndarray):
            return sampled_goal.squeeze()

        return self.global_option.extract_goal_dimensions(sampled_goal)

    def add_new_options(self, new_options):
        for new_option in new_options:  # type: ModelBasedOption
            if new_option is not None:
                self.new_options.append(new_option)
                self.skill_tree.add_node(new_option)

    def log_success_metrics(self, episode):
        options = self.mature_options + self.new_options
        individual_option_data = {option.name: option.get_option_success_rate() for option in options}
        overall_success = reduce(lambda x,y: x*y, individual_option_data.values())
        self.log[episode] = {"individual_option_data": individual_option_data, "success_rate": overall_success}

        if episode % self.evaluation_freq == 0 and episode > self.warmup_episodes:
            success, step_count, _ = test_agent(self, 1, self.max_steps, get_trajectories=False)

            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]

            with open(f"results/{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            options = self.mature_options + self.new_options

            for option in self.mature_options:
                episode_label = episode if self.generate_init_gif else -1
                plot_two_class_classifier(option, episode_label, self.experiment_name, plot_examples=True)

            for option in options:
                if self.use_global_vf:
                    make_chunked_goal_conditioned_value_function_plot(option.global_value_learner,
                                                                      goal=option.get_goal_for_rollout(),
                                                                      episode=-1, seed=self.seed,
                                                                      experiment_name=self.experiment_name,
                                                                      option_idx=option.option_idx)
                else:
                    make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                                      goal=option.get_goal_for_rollout(),
                                                                      episode=-1, seed=self.seed,
                                                                      experiment_name=self.experiment_name)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.skill_tree.options) + 1 if parent is not None else 1
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  init_salient_event=self.init_salient_event,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.global_option.value_learner,
                                  use_model=self.use_model,
                                  use_global_vf=self.use_global_vf,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  option_idx=option_idx,
                                  max_num_children=self.max_num_children)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=100, max_steps=self.max_steps, device=self.device,
                                  init_salient_event=self.init_salient_event,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=None,
                                  use_model=self.use_model,
                                  use_global_vf=self.use_global_vf,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  option_idx=0,
                                  max_num_children=self.max_num_children)
        return option

    def reset(self, episode):
        self.mdp.reset()

        if self.use_diverse_starts and episode > self.warmup_episodes:
            goal_cond = lambda s: self.mdp.is_goal_region(np.array([s]))[0]
            random_state = self.mdp.sample_random_state(reject_cond=goal_cond)
            random_position = self.mdp.get_position(random_state)
            if not self.mdp.is_goal_region(random_state):
                self.mdp.set_xy(random_position)

        print(f"[Episode {episode}] Reset state to {self.mdp.cur_state}")


def test_agent(exp, num_experiments, num_steps, get_trajectories=False):
    def rollout():
        step_number = 0
        episodic_trajectory = []
        while step_number < num_steps and not \
        exp.mdp.sparse_gc_reward_func(exp.mdp.cur_state, exp.mdp.goal_state, {})[1]:
            state = deepcopy(exp.mdp.cur_state)
            selected_option, subgoal = exp.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal,
                                                          eval_mode=True)
            if get_trajectories:
                episodic_trajectory.append((selected_option.option_idx, transitions))

            step_number += len(transitions)
        return step_number, episodic_trajectory

    success = 0
    step_counts = []

    trajectories = []

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken, trajectory = rollout()
        if steps_taken != num_steps:
            success += 1
        step_counts.append(steps_taken)

        if get_trajectories:
            trajectories.append(trajectory)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts, trajectories
