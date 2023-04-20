import ipdb
import time
import pickle
import numpy as np
from copy import deepcopy
from functools import reduce
from collections import deque
from hrl.agent.dsc.utils import *
from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.dsc.MBOptionClass import ModelBasedOption
from hrl.agent.dsc.classifier.initiation_gvf import GoalConditionedInitiationGVF


class RobustDSC(object):
    def __init__(self, mdp, warmup_episodes, max_steps, gestation_period, buffer_length, use_vf, use_global_vf, use_model,
                 use_diverse_starts, use_dense_rewards, lr_c, lr_a,
                 experiment_name, device,
                 logging_freq, generate_init_gif, evaluation_freq, seed, multithread_mpc,
                 max_num_children,
                 init_classifier_type, optimistic_threshold, pessimistic_threshold,
                 use_initiation_gvf, use_reachability_goal_sampling,
                 only_reweigh_negative_examples):

        self.lr_c = lr_c
        self.lr_a = lr_a

        self.device = device
        self.use_vf = use_vf
        self.use_global_vf = use_global_vf
        self.use_model = use_model
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.max_steps = max_steps
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards
        self.multithread_mpc = multithread_mpc
        self.use_initiation_gvf = use_initiation_gvf
        self.use_reachability_goal_sampling = use_reachability_goal_sampling
        self.only_reweigh_negative_examples = only_reweigh_negative_examples
        
        self.init_classifier_type = init_classifier_type
        self.optimistic_threshold = optimistic_threshold
        self.pessimistic_threshold = pessimistic_threshold

        self.seed = seed
        self.logging_freq = logging_freq
        self.evaluation_freq = evaluation_freq
        self.generate_init_gif = generate_init_gif

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period

        self.mdp = mdp
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.uvfa_policy = self.create_uvfa_policy(lr_a, lr_c)

        self.initiation_gvf = GoalConditionedInitiationGVF(
            target_policy=self.uvfa_policy.actor,
            n_actions=mdp.action_space_size(),
            n_input_channels=mdp.state_space_size()+2,
            batch_size=64,
        ) if use_initiation_gvf else None

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.log = {}

    @staticmethod
    def _pick_earliest_option(state, options):
        for option in options:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option

    def act(self, state):
        # current_option = self._pick_earliest_option(state, self.chain)
        # return current_option if current_option is not None else self.global_option
        for option in self.chain:
            if option.is_init_true(state):
                subgoal = option.get_goal_for_rollout(
                    state if self.use_reachability_goal_sampling else None
                )
                if not option.is_at_local_goal(state, subgoal):
                    return option, subgoal
        return self.global_option, self.global_option.get_goal_for_rollout()

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

            # Overwrite the subgoal for the global-option
            if selected_option == self.global_option:
                subgoal = self.pick_subgoal_for_global_option(state)

            transitions, reward = selected_option.rollout(step_number=step_number, goal=subgoal)

            if len(transitions) == 0:
                break

            self.manage_chain_after_rollout(selected_option)
            step_number += len(transitions)
        
        self.update_initiation_learners(n_updates=step_number)

        return step_number

    def run_loop(self, num_episodes, num_steps, start_episode=0):
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

    def update_initiation_learners(self, n_updates):
        """Update the initiation GVF and then update all the init-classifiers."""

        if self.use_initiation_gvf:
            t0 = time.time()
            self.initiation_gvf.update(n_updates)
            print(f'Took {time.time()-t0}s to update Initiation GVF')

        t0 = time.time()
        for option in self.chain:
            assert isinstance(option, ModelBasedOption)
            option.initiation_classifier.fit_initiation_classifier(
                self.initiation_gvf,
                goal=option.get_goal_for_rollout() if self.use_initiation_gvf else None
            )
        print(f'Took {time.time()-t0}s to update initiation classifiers.')

    def log_success_metrics(self, episode):
        individual_option_data = {option.name: option.get_option_success_rate() for option in self.chain}
        overall_success = reduce(lambda x,y: x*y, individual_option_data.values())
        self.log[episode] = {"individual_option_data": individual_option_data, "success_rate": overall_success}

        if episode % self.evaluation_freq == 0 and episode > self.warmup_episodes:
            success, step_count = test_agent(self, 1, self.max_steps)

            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]

            with open(f"results/{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def learn_dynamics_model(self, epochs=50, batch_size=1024):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=epochs, batch_size=batch_size)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def is_chain_complete(self):
        learned = all([option.get_training_phase() == "initiation_done" for option in self.chain])
        covered = self.mature_options[-1].is_init_true(
            self.mdp.start_state
        )
        return learned and covered 

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.contains_init_state()
        return False

    def contains_init_state(self):
        for option in self.mature_options:
            if option.is_init_true(
                self.mdp.start_state
            ):
                return True
        return False

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def find_nearest_option_in_chain(self, state):
        if len(self.mature_options) > 0:
            distances = [(option, option.distance_to_state(state)) for option in self.mature_options]
            nearest_option = sorted(distances, key=lambda x: x[1])[0][0]  # type: ModelBasedOption
            return nearest_option

    def pick_subgoal_for_global_option(self, state):
        nearest_option = self.find_nearest_option_in_chain(state)
        if nearest_option is not None:
            goal = nearest_option.initiation_classifier.sample()
            if goal is not None:
                return nearest_option.extract_goal_dimensions(goal)
        return self.global_option.get_goal_for_rollout()

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            options = self.mature_options + self.new_options

            for option in self.mature_options:
                assert isinstance(option, ModelBasedOption)
                episode_label = episode if self.generate_init_gif else -1
                option.initiation_classifier.plot_initiation_classifier(
                    self.mdp,
                    option.solver.replay_buffer, 
                    option.name, 
                    episode_label,
                    self.experiment_name,
                    self.seed
                )

            for option in options:
                print(f"Plotting value function for {option}")
                if self.use_global_vf:
                    goal = option.get_goal_for_rollout()
                    make_chunked_goal_conditioned_value_function_plot(option.global_value_learner,
                                                                    goal=goal,
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name,
                                                                    option_idx=option.option_idx)
                    if self.use_initiation_gvf:
                        visualize_initiation_gvf(
                            self.initiation_gvf,
                            self.uvfa_policy.actor,
                            goal,
                            episode,
                            self.experiment_name,
                            self.seed
                        )
                else:
                    make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                                    goal=option.get_goal_for_rollout(),
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.uvfa_policy,
                                  option_idx=option_idx,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  only_reweigh_negative_examples=self.only_reweigh_negative_examples,
                                  multithread_mpc=self.multithread_mpc,
                                  init_classifier_type=self.init_classifier_type,
                                  optimistic_threshold=self.optimistic_threshold,
                                  pessimistic_threshold=self.pessimistic_threshold,
                                  initiation_gvf=self.initiation_gvf)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=self.buffer_length,
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=200, max_steps=self.max_steps, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.uvfa_policy,
                                  option_idx=0,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  only_reweigh_negative_examples=self.only_reweigh_negative_examples,
                                  multithread_mpc=self.multithread_mpc,
                                  init_classifier_type=self.init_classifier_type,
                                  optimistic_threshold=self.optimistic_threshold,
                                  pessimistic_threshold=self.pessimistic_threshold,
                                  initiation_gvf=self.initiation_gvf)
        return option
    
    def create_uvfa_policy(self, lr_a, lr_c):
        return TD3(state_dim=self.mdp.state_space_size()+2,
                   action_dim=self.mdp.action_space_size(),
                   max_action=1.,
                   name=f"uvfa-td3-agent",
                   device=self.device,
                   lr_c=lr_c, lr_a=lr_a,
                   use_output_normalization=False)

    def reset(self, episode):
        self.mdp.reset()

        if self.use_diverse_starts and episode > self.warmup_episodes and episode % self.evaluation_freq != 0:
            random_state = self.mdp.sample_random_state()
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)

def test_agent(exp, num_experiments, num_steps):
    def rollout():
        step_number = 0
        while step_number < num_steps and not exp.mdp.sparse_gc_reward_func(exp.mdp.cur_state, exp.mdp.goal_state)[1]:

            state = deepcopy(exp.mdp.cur_state)
            selected_option, subgoal = exp.act(state)
            transitions, reward = selected_option.rollout(step_number=step_number, goal=subgoal, eval_mode=True)
            step_number += len(transitions)
        return step_number

    success = 0
    step_counts = []

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken = rollout()
        if steps_taken != num_steps:
            success += 1
        step_counts.append(steps_taken)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts

def get_trajectory(exp, num_steps):
    exp.mdp.reset()
    
    traj = []
    step_number = 0
    
    while step_number < num_steps and not exp.mdp.sparse_gc_reward_function(exp.mdp.cur_state, exp.mdp.goal_state, {})[1]:
        state = deepcopy(exp.mdp.cur_state)
        selected_option, subgoal = exp.act(state)
        transitions, reward = selected_option.rollout(step_number=step_number, rollout_goal=subgoal, eval_mode=True)
        step_number += len(transitions)
        traj.append((selected_option.name, transitions))
    return traj, step_number
