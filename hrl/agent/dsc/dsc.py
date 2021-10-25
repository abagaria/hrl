import pickle
from hrl.agent.dsc.utils import *
from .option import ModelFreeOption
from ...salient_event.salient_event import SalientEvent


class RobustDSC(object):
    def __init__(self, mdp, gestation_period, buffer_length,
                 experiment_name, gpu_id, use_oracle_rf,
                 init_obs, init_pos, target_obs, target_pos,
                 seed, log_filename):

        self.mdp = mdp
        self.seed = seed
        self.gpu_id = gpu_id
        self.experiment_name = experiment_name

        self.use_oracle_rf = use_oracle_rf
        self.buffer_length = buffer_length
        self.gestation_period = gestation_period

        self.init_salient_event = SalientEvent(init_obs, init_pos, tol=2.)
        self.target_salient_event = SalientEvent(target_obs, target_pos, tol=2.)

        self.global_option = self.create_global_option()
        self.goal_option = self.create_local_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]
        self.mature_options = []

        self.log_file = log_filename

    def act(self, state):
        for option in self.chain:
            if option.is_init_true(state):
                subgoal = option.get_goal_for_rollout()
                if not option.is_term_true(state):
                    return option, subgoal
        return self.global_option, self.global_option.get_goal_for_rollout()

    def dsc_rollout(self, state, pos, episode, eval_mode=False):
        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        rollout_trajectory = []

        while not done and not reset and not reached:
            selected_option, subgoal = self.act(pos)  # TODO: Maybe pass in the subgoal..
            next_state, done, reset, visited_positions, goal_pos = selected_option.rollout(state, pos,
                                                                                           eval_mode=eval_mode)
            
            self.manage_chain_after_rollout(selected_option, episode)

            state = next_state
            pos = self.mdp.get_current_position()

            reward, reached = self.global_option.rf(pos, (123, 148))

            episode_reward += reward
            episode_length += len(visited_positions)

            rollout_trajectory.append({
                "goal": goal_pos,
                "trajectory": visited_positions,
                "option": selected_option.option_idx,
            })

        return rollout_trajectory, episode_reward, episode_length

    def run_loop(self, num_steps):
        step = 0
        episode = 0

        _log_steps = []
        _log_rewards = []

        while step < num_steps:
            state = self.mdp.reset()
            position = self.mdp.get_current_position()

            _, reward, length = self.dsc_rollout(state, position, episode)

            episode += 1
            step += length

            _log_steps.append(step)
            _log_rewards.append(reward)

            with open(self.log_file, "wb+") as f:
                episode_metrics = {
                                "step": _log_steps, 
                                "reward": _log_rewards,
                }
                pickle.dump(episode_metrics, f)

            print(f"Episode: {episode}, T: {step}, Reward: {reward}")

            if episode > 0 and episode % 100 == 0:
                for option in self.mature_options:
                    plot_two_class_classifier(option, episode, self.experiment_name, seed=self.seed)

    def is_chain_complete(self):
        return all([option.get_training_phase() == "initiation_done" for option in self.chain]) \
                and self.contains_init_state()

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.contains_init_state()
        return False

    def contains_init_state(self):
        start_pos = self.init_salient_event.get_target_position()
        for option in self.mature_options:
            if option.is_init_true(start_pos):
                return True
        return False

    def manage_chain_after_rollout(self, executed_option, episode):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

            plot_two_class_classifier(executed_option, episode, self.experiment_name, seed=self.seed)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_local_option(name, parent=self.mature_options[-1])
            print(f"Creating {name}, parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def find_nearest_option_in_chain(self, state): # TODO: This needs to take in a pos
        if len(self.mature_options) > 0:
            distances = [(option, option.distance_to_state(state)) for option in self.mature_options]
            nearest_option = sorted(distances, key=lambda x: x[1])[0][0]
            return nearest_option

    def pick_subgoal_for_global_option(self, state):  # TODO: This needs to take in a pos
        nearest_option = self.find_nearest_option_in_chain(state)
        if nearest_option is not None:
            return nearest_option.initiation_classifier.sample()
        return self.global_option.get_goal_for_rollout()

    def create_local_option(self, name, parent=None):
        option_idx = len(self.chain) + 1 if parent is not None else 1
        option = ModelFreeOption(name=name,
                                 option_idx=option_idx,
                                 parent=parent, 
                                 timeout=200, 
                                 env=self.mdp,
                                 global_init=False,
                                 global_solver=self.global_option.solver,
                                 gpu_id=self.gpu_id,
                                 buffer_length=self.buffer_length,
                                 gestation_period=self.gestation_period,
                                 n_training_steps=int(2e6),  # TODO
                                 init_salient_event=self.init_salient_event,
                                 target_salient_event=self.target_salient_event,
                                 use_oracle_rf=self.use_oracle_rf)
        return option

    def create_global_option(self):
        option = ModelFreeOption(name="global-option",
                                 option_idx=0,
                                 parent=None,
                                 timeout=100,
                                 env=self.mdp,
                                 global_init=True,
                                 global_solver=None,
                                 gpu_id=self.gpu_id,
                                 buffer_length=self.buffer_length,
                                 gestation_period=self.gestation_period,
                                 n_training_steps=int(2e6),  # TODO
                                 init_salient_event=self.init_salient_event,
                                 target_salient_event=self.target_salient_event,
                                 use_oracle_rf=self.use_oracle_rf)
        return option
