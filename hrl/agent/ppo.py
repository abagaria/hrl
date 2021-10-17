from logging import getLogger

import torch
import pfrl
from pfrl.utils.batch_states import batch_states


class PPO(pfrl.agents.PPO):
    """
    implementation of PPO
    
    override the batch_act and batch_observe function from pfrl, and don't support
    the recurrent feature from pfrl (because _batch_observe_eval) is not implemented
    """
    def __init__(self,
                 model,
                 optimizer,
                 obs_normalizer=None,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coef=1,
                 entropy_coef=0.01,
                 update_interval=2048,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=None,
                 standardize_advantages=True,
                 batch_states=batch_states,
                 max_recurrent_sequence_len=None,
                 act_deterministically=False,
                 max_grad_norm=None,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 value_loss_stats_window=100,
                 policy_loss_stats_window=100):
        super().__init__(model,
                         optimizer,
                         obs_normalizer=obs_normalizer,
                         gpu=gpu,
                         gamma=gamma,
                         lambd=lambd,
                         phi=phi,
                         value_func_coef=value_func_coef,
                         entropy_coef=entropy_coef,
                         update_interval=update_interval,
                         minibatch_size=minibatch_size,
                         epochs=epochs,
                         clip_eps=clip_eps,
                         clip_eps_vf=clip_eps_vf,
                         standardize_advantages=standardize_advantages,
                         batch_states=batch_states,
                         recurrent=False,  # we don't have support for recurrent models yet!
                         max_recurrent_sequence_len=max_recurrent_sequence_len,
                         act_deterministically=act_deterministically,
                         max_grad_norm=max_grad_norm,
                         value_stats_window=value_stats_window,
                         entropy_stats_window=entropy_stats_window,
                         value_loss_stats_window=value_loss_stats_window,
                         policy_loss_stats_window=policy_loss_stats_window)
        
        self.replay_buffer = self.memory  # for compatibility with plotting API

    def batch_act(self, batch_obs, evaluation_mode=False):
        if evaluation_mode:
            with self.eval_mode():
                assert not self.training
                return self._batch_act_eval(batch_obs)
        else:
            assert self.training
            return self._batch_act_train(batch_obs)

    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)
    
        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            action_distrib, batch_value = self.model(b_state)
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        return batch_action

    def batch_observe(self, states, actions, rewards, next_states, is_terminals):
        if self.training:
            for i, (state, action, reward, next_state, terminal) in enumerate(
                zip(
                    states,
                    actions,
                    rewards,
                    next_states,
                    is_terminals
                )
            ):
                if state is not None:
                    assert action is not None
                    transition = {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "nonterminal": 0.0 if terminal else 1.0,
                    }
                    self.batch_last_episode[i].append(transition)
                if terminal:
                    assert self.batch_last_episode[i]
                    self.memory.append(self.batch_last_episode[i])
                    self.batch_last_episode[i] = []

            self.train_prev_recurrent_states = None
            self._update_if_dataset_is_ready()
    
    def get_values(self, states):
        """
        get the value for states, according to the critic network
        """
        with torch.no_grad(), pfrl.utils.evaluating(self.model), self.eval_mode():
            batched_states = self.batch_states(states, self.device, self.phi)
            if self.obs_normalizer:
                batched_states = self.obs_normalizer(batched_states, update=False)
            distribs, vs_pred = self.model(states)
            return vs_pred
