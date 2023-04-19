# import ipdb
import pickle
import numpy as np

from hrl.agent.dsc.utils import *
from hrl.agent.dsc.chain import SkillChain
from hrl.agent.dsc.option import ModelFreeOption
from hrl.salient_event.salient_event import SalientEvent


class RobustDSC(object):
    def __init__(self, mdp, gestation_period, buffer_length,
                 experiment_name, gpu_id,
                 init_event,
                 use_oracle_rf, use_rf_on_pos_traj,
                 use_rf_on_neg_traj, replay_original_goal_on_pos,
                 use_pos_for_init,
                 p_her, max_num_options, seed, log_filename,
                 num_kmeans_clusters, sift_threshold,
                 classifier_type, use_full_neg_traj, use_pessimistic_relabel,
                 noisy_net_sigma, rnd_data_path):

        self.mdp = mdp
        self.seed = seed
        self.p_her = p_her

        self.mdp = mdp
        self.seed = seed
        self.gpu_id = gpu_id
        self.experiment_name = experiment_name

        self.use_oracle_rf = use_oracle_rf
        self.use_rf_on_pos_traj = use_rf_on_pos_traj
        self.use_rf_on_neg_traj = use_rf_on_neg_traj
        self.replay_original_goal_on_pos = replay_original_goal_on_pos

        self.max_num_options = max_num_options
        self.use_pos_for_init = use_pos_for_init
        
        self.buffer_length = buffer_length
        self.gestation_period = gestation_period
        self.init_salient_event = init_event
        self.num_kmeans_clusters = num_kmeans_clusters
        self.sift_threshold = sift_threshold
        self.noisy_net_sigma = noisy_net_sigma
        self.classifier_type = classifier_type
        self.use_full_neg_traj = use_full_neg_traj
        self.use_pessimistic_relabel = use_pessimistic_relabel

        self.global_option = self.create_global_option()

        self.chains = []
        self.new_options = []
        self.mature_options = []
        self.current_option_idx = 1

        self.log_file = log_filename
        self.rnd_data_path = rnd_data_path

        if rnd_data_path:
            _, ram_trajectories, frame_trajectories = get_saved_trajectories(
                rnd_data_path, n_trajectories=100
            )

            self.rnd_rams = flatten(ram_trajectories)
            self.rnd_frames = flatten(frame_trajectories)

    # ------------------------------------------------------------
    # Action selection methods
    # ------------------------------------------------------------

    @staticmethod
    def _pick_earliest_option(state, info, options):
        cond = lambda x, y: x.is_init_true(*y) and not x.is_term_true(*y)
        for option in options:
            if cond(option, (state, info)):
                return option

    def _get_chains_corresponding_to_goal(self, goal_info):
        chains = [chain for chain in self.chains if chain.target_salient_event(goal_info)]

        if len(chains) == 0:
            for chain in self.chains:
                for option in chain.options:
                    if option.is_term_true(None, goal_info):
                        chains.append(chain)
        return chains

    def act(self, state, info, goal_info):
        chains_targeting_goal = self._get_chains_corresponding_to_goal(goal_info)
        for chain in chains_targeting_goal:
            for option in chain.options:
                if option.is_init_true(state, info) and not option.is_term_true(state, info):
                    return option
        return self.global_option

    # ------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------

    def dsc_rollout(self, state, info, goal_salient_event, episode,
                    eval_mode=False, interrupt_handle=lambda state, info: False):
        assert isinstance(goal_salient_event, SalientEvent)
        assert len(self.chains) > 0, "Create skill chains b/w constructor and rollout"

        done = False
        reset = False
        reached = False

        episode_length = 0
        episode_reward = 0.
        rollout_trajectory = []
        learned_options = []

        while not done and not reset and not reached and not interrupt_handle(state, info):
            selected_option = self.act(state, info, goal_salient_event.target_info)
            next_state, done, reset, _, goal_pos, info, transitions = selected_option.rollout(
                                                                                state,
                                                                                info,
                                                                                goal_salient_event,
                                                                                eval_mode=eval_mode
                                                                            )
            infos = [transition[-1] for transition in transitions]  # these do not contain the start info
            finished = self.manage_chain_after_option_rollout(selected_option, episode)

            reward, reached = self.global_option.rf(self.mdp.get_current_position(),
                                                    goal_salient_event.target_pos)

            state = next_state
            episode_reward += reward
            episode_length += len(infos)
            if finished: learned_options.append(selected_option)

            rollout_trajectory.append({
                "goal": goal_pos,
                "trajectory": infos,
                "option": selected_option.option_idx,
            })

        # Was returning `episode_reward, episode_length` as well

        return state, info, done, reset, learned_options, rollout_trajectory

    def run_loop(self, goal_salient_event, num_steps):
        step = 0
        episode = 0

        _log_steps = []
        _log_rewards = []

        while step < num_steps:
            state, info = self.mdp.reset()

            _, _, _, _, _, reward, length = self.dsc_rollout(state, info, goal_salient_event, episode)

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
                    if self.use_pos_for_init:
                        plot_two_class_classifier(option, episode, self.experiment_name, seed=self.seed)
                    else:
                        option.initiation_classifier.plot_training_predictions(option.name, episode,
                                                                               self.experiment_name, self.seed)

    # ------------------------------------------------------------
    # Managing the skill chains
    # ------------------------------------------------------------

    def should_create_children_options(self, parent_option):

        if parent_option is None:
            return False

        return self.should_create_more_options() \
               and parent_option.get_training_phase() == "initiation_done" \
               and self.chains[parent_option.chain_id - 1].should_continue_chaining()

    def should_create_more_options(self):
        """ Continue chaining as long as any chain is still accepting new options. """
        return any([chain.should_continue_chaining() for chain in self.chains])

    def add_new_option_to_skill_chain(self, new_option):
        self.new_options.append(new_option)
        chain_idx = new_option.chain_id - 1
        self.chains[chain_idx].options.append(new_option)

    def finish_option_initiation_phase(self, option):
        assert option.get_training_phase() == "initiation_done"
        self.new_options.remove(option)
        self.mature_options.append(option)

    def manage_chain_after_option_rollout(self, option, episode):

        if option in self.new_options and option.get_training_phase() != "gestation":
            self.finish_option_initiation_phase(option)
            should_create_children = self.should_create_children_options(option)
            if should_create_children:
                new_option = self.create_child_option(parent=option)
                self.add_new_option_to_skill_chain(new_option)
            
            if self.use_pos_for_init:
                plot_two_class_classifier(option, episode, self.experiment_name, seed=self.seed)
            else:
                option.initiation_classifier.plot_training_predictions(option.name, episode,
                                                                       self.experiment_name, self.seed)

                if self.rnd_data_path and option.target_salient_event:
                    plot_classifier_predictions(
                        option, self.rnd_frames, self.rnd_rams,
                        episode, self.seed, self.experiment_name
                    )

            return True

        return False

    # ------------------------------------------------------------
    # Convenience functions
    # ------------------------------------------------------------

    def create_new_chain(self, *, init_event, target_event):
        chain_id = len(self.chains) + 1
        name = f"goal-option-{chain_id}"
        root_option = self.create_local_option(name=name,
                                               parent=None,
                                               chain_idx=chain_id,
                                               init_event=init_event,
                                               target_event=target_event)
        self.new_options.append(root_option)

        new_chain = SkillChain(options=[root_option], 
                               chain_id=chain_id,
							   target_salient_event=target_event,
							   init_salient_event=init_event)
        self.add_skill_chain(new_chain)
        
        return new_chain

    def add_skill_chain(self, new_chain):
        if new_chain not in self.chains:
            self.chains.append(new_chain)

    def create_local_option(self,
                            name,
                            init_event, 
                            target_event,
                            chain_idx,
                            parent=None):
        option = ModelFreeOption(name=name,
                                 option_idx=self.current_option_idx,
                                 parent=parent, 
                                 timeout=np.inf, 
                                 env=self.mdp,
                                 global_init=False,
                                 global_solver=self.global_option.solver,
                                 gpu_id=self.gpu_id,
                                 buffer_length=self.buffer_length,
                                 gestation_period=self.gestation_period,
                                 n_training_steps=int(2e6),  # TODO
                                 init_salient_event=init_event,
                                 target_salient_event=target_event,
                                 
                                 use_oracle_rf=self.use_oracle_rf,
                                 use_rf_on_pos_traj=self.use_rf_on_pos_traj,
                                 use_rf_on_neg_traj=self.use_rf_on_neg_traj,
                                 replay_original_goal_on_pos=self.replay_original_goal_on_pos,

                                 max_num_options=self.max_num_options,
                                 use_pos_for_init=self.use_pos_for_init,
                                 chain_id=chain_idx,
                                 p_her=self.p_her,
                                 num_kmeans_clusters=self.num_kmeans_clusters,
                                 sift_threshold=self.sift_threshold,
                                 classifier_type=self.classifier_type,
                                 use_full_neg_traj=self.use_full_neg_traj,
                                 use_pessimistic_relabel=self.use_pessimistic_relabel,
                                 noisy_net_sigma=self.noisy_net_sigma)
        self.current_option_idx += 1
        return option

    def create_global_option(self):
        option = ModelFreeOption(name="global-option",
                                 option_idx=0,
                                 parent=None,
                                 timeout=np.inf,
                                 env=self.mdp,
                                 global_init=True,
                                 global_solver=None,
                                 gpu_id=self.gpu_id,
                                 buffer_length=self.buffer_length,
                                 gestation_period=self.gestation_period,
                                 n_training_steps=int(2e6),  # TODO
                                 init_salient_event=self.init_salient_event,
                                 target_salient_event=None,
                                 
                                 use_oracle_rf=self.use_oracle_rf,
                                 use_rf_on_pos_traj=self.use_rf_on_pos_traj,
                                 use_rf_on_neg_traj=self.use_rf_on_neg_traj,
                                 replay_original_goal_on_pos=self.replay_original_goal_on_pos,

                                 max_num_options=self.max_num_options,
                                 use_pos_for_init=self.use_pos_for_init,
                                 chain_id=0,
                                 p_her=self.p_her,
                                 num_kmeans_clusters=self.num_kmeans_clusters,
                                 sift_threshold=self.sift_threshold,
                                 classifier_type=self.classifier_type,
                                 use_full_neg_traj=self.use_full_neg_traj,
                                 use_pessimistic_relabel=self.use_pessimistic_relabel,
                                 noisy_net_sigma=self.noisy_net_sigma)
        return option

    def create_child_option(self, parent):
        assert isinstance(parent, ModelFreeOption)

        # Create new option whose termination is the initiation of the option we just trained
        prefix = f"option_{parent.name.split('_')[-1]}" if "goal_option" in parent.name else parent.name
        name = prefix + "-{}".format(len(parent.children))
        print("Creating {} with parent {}".format(name, parent.name))

        new_option = self.create_local_option(name,
                                              parent.init_salient_event,
                                              parent.target_salient_event,
                                              chain_idx=parent.chain_id,
                                              parent=parent)
        parent.children.append(new_option)
        return new_option
