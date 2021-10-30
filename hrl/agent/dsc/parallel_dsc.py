from copy import deepcopy
from collections import deque

import numpy as np

from hrl.agent.dsc.utils import *
from hrl.salient_event.SalientEventClass import SalientEvent
from hrl.envs.vector_env import EpisodicSyncVectorEnv
from hrl.option.parallel_model_based_option import ParallelModelBasedOption


class ParallelRobustDSC:
    def __init__(self, env, warmup_episodes, max_steps, gestation_period, buffer_length, use_vf, use_global_vf, use_model,
                 use_diverse_starts, use_dense_rewards, lr_c, lr_a, clear_option_buffers, goal_state,
                 use_global_option_subgoals, experiment_name, device,
                 logging_freq, generate_init_gif, seed, multithread_mpc):

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
        self.clear_option_buffers = clear_option_buffers
        self.use_global_option_subgoals = use_global_option_subgoals
        self.goal_state = goal_state

        self.multithread_mpc = multithread_mpc

        self.seed = seed
        self.logging_freq = logging_freq
        self.generate_init_gif = generate_init_gif

        self.buffer_length = buffer_length
        self.gestation_period = gestation_period

        self.env = env
        assert isinstance(env, EpisodicSyncVectorEnv)
        self.nenvs = env.nenvs
        self.target_salient_event = SalientEvent(target_state=goal_state, event_idx=1)

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.log = {}

        # house keeping for parallelized envs
        self.current_controlling_options = [self.global_option] * self.nenvs
        self.current_subgoals = [self.global_option.get_goal_for_rollout()] * self.nenvs
        
    def reset(self):
        self.current_controlling_options = [self.global_option] * self.nenvs
        self.current_subgoals = [self.global_option.get_goal_for_rollout()] * self.nenvs

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
                subgoal = option.get_goal_for_rollout()
                if not option.is_at_local_goal(state, subgoal):
                    return option, subgoal
        return self.global_option, self.global_option.get_goal_for_rollout()
    
    def batch_observe(self, batch_states, batch_actions, batch_rewards, batch_nest_states, batch_dones):
        """
        main API for interacting with the env: observe batched transitions
        """
        assert len(batch_states) == self.nenvs
        for i in range(self.nenvs):
            self.current_controlling_options[i].observe(
                batch_states[i],
                batch_actions[i],
                batch_rewards[i],
                batch_nest_states[i],
                batch_dones[i]
            )

    def batch_act(self, batch_states, evaluation_mode=False):
        """
        main API for interacting with the env: observe a batched state and 
        return batched primitive actions for the environment

        the batched_states should be the original state, not enhanced by the goal
        """
        if not evaluation_mode:
            assert len(batch_states) == self.nenvs
        batch_options, batch_goals = self.batch_pick_options_and_goals(batch_states, evaluation_mode)
        batched_actions = []
        for i, (option, goal) in enumerate(zip(batch_options, batch_goals)):
            a = option.act(batch_states[i], goal, eval_mode=evaluation_mode)
            batched_actions.append(a)
        return batched_actions
        

    def batch_pick_options_and_goals(self, batch_states, evaluation_mode=False):
        """
        1. picking the options and goals
        2. updating the two lists agent keeps track of
        3. updating the options themselves
        """
        if evaluation_mode:
            nenvs = len(batch_states)
        else:
            assert len(batch_states) == self.nenvs
        for i, state in enumerate(batch_states):
            # see if option rollout has terminated
            option = self.current_controlling_options[i]
            at_goal = option.is_at_local_goal(state, self.current_subgoals[i])
            timeout = option.rollout_num_steps >= option.timeout
            if at_goal or timeout:
                if not evaluation_mode:
                    # if should create new option
                    self.manage_chain_after_rollout(option)
                    # refine existing option
                    option.refine(rollout_goal=self.current_subgoals[i])
                # choose new controlling option
                o, g = self.act(state)
                self.current_controlling_options[i] = o
                self.current_subgoals[i] = g
        
        if evaluation_mode:
            return self.current_controlling_options[:nenvs], self.current_subgoals[:nenvs]
        return self.current_controlling_options, self.current_subgoals

    def learn_dynamics_model(self, epochs=50, batch_size=1024):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=epochs, batch_size=batch_size)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def is_chain_complete(self):
        return all([option.get_training_phase() == "initiation_done" for option in self.chain]) and self.mature_options[-1].is_init_true(np.array([0,0]))

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.contains_init_state()
        return False

    def contains_init_state(self):
        for option in self.mature_options:
            if option.is_init_true(np.array([0,0])):  # TODO: Get test-time start state automatically
                return True
        return False

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            if self.clear_option_buffers:
                self.filter_replay_buffer(executed_option)

        if executed_option.num_goal_hits == 2 * executed_option.gestation_period and self.clear_option_buffers:
            self.filter_replay_buffer(executed_option)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def find_nearest_option_in_chain(self, state):
        if len(self.mature_options) > 0:
            distances = [(option, option.distance_to_state(state)) for option in self.mature_options]
            nearest_option = sorted(distances, key=lambda x: x[1])[0][0]  # type: ParallelModelBasedOption
            return nearest_option

    def pick_subgoal_for_global_option(self, state):
        nearest_option = self.find_nearest_option_in_chain(state)
        if nearest_option is not None:
            return nearest_option.sample_from_initiation_region_fast_and_epsilon()
        return self.global_option.get_goal_for_rollout()

    def filter_replay_buffer(self, option):
        assert isinstance(option, ParallelModelBasedOption)
        print(f"Clearing the replay buffer for {option.name}")
        option.value_learner.replay_buffer.clear()

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
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name,
                                                                    option_idx=option.option_idx)
                else:
                    make_chunked_goal_conditioned_value_function_plot(option.value_learner,
                                                                    goal=option.get_goal_for_rollout(),
                                                                    episode=episode, seed=self.seed,
                                                                    experiment_name=self.experiment_name)

    def create_model_based_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ParallelModelBasedOption(parent=parent, env=self.env,
                                  buffer_length=self.buffer_length,
                                  goal_state_size=len(self.goal_state),
                                  global_init=False,
                                  gestation_period=self.gestation_period,
                                  timeout=200, 
                                  max_steps=self.max_steps, 
                                  device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=self.global_option.value_learner,
                                  option_idx=option_idx,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  multithread_mpc=self.multithread_mpc)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
        option = ParallelModelBasedOption(parent=None, env=self.env,
                                  buffer_length=self.buffer_length,
                                  goal_state_size=len(self.goal_state),
                                  global_init=True,
                                  gestation_period=self.gestation_period,
                                  timeout=200, 
                                  max_steps=self.max_steps, 
                                  device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None,
                                  use_vf=self.use_vf,
                                  use_global_vf=self.use_global_vf,
                                  use_model=self.use_model,
                                  dense_reward=self.use_dense_rewards,
                                  global_value_learner=None,
                                  option_idx=0,
                                  lr_c=self.lr_c, lr_a=self.lr_a,
                                  multithread_mpc=self.multithread_mpc)
        return option
