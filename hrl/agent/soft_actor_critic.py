from logging import getLogger

import torch
import pfrl
from pfrl.utils.batch_states import batch_states


class SoftActorCritic(pfrl.agents.SoftActorCritic):
    """
    implementation of soft actor critic
    
    override the batch_act and batch_observe function from pfrl
    """
    def __init__(self, 
                policy, 
                q_func1, 
                q_func2, 
                policy_optimizer, 
                q_func1_optimizer, 
                q_func2_optimizer, 
                replay_buffer, 
                gamma, 
                gpu=None, 
                replay_start_size=10000, 
                minibatch_size=100, 
                update_interval=1, 
                phi=lambda x:x, 
                soft_update_tau=0.005, 
                max_grad_norm=None, 
                logger=getLogger(__name__), 
                batch_states=batch_states, 
                burnin_action_func=None, 
                initial_temperature=1, 
                entropy_target=None, 
                temperature_optimizer_lr=None, 
                act_deterministically=True):
        super().__init__(policy, 
                        q_func1, 
                        q_func2, 
                        policy_optimizer, 
                        q_func1_optimizer, 
                        q_func2_optimizer, 
                        replay_buffer, 
                        gamma, 
                        gpu=gpu, 
                        replay_start_size=replay_start_size, 
                        minibatch_size=minibatch_size, 
                        update_interval=update_interval, 
                        phi=phi, 
                        soft_update_tau=soft_update_tau, 
                        max_grad_norm=max_grad_norm, 
                        logger=logger, 
                        batch_states=batch_states, 
                        burnin_action_func=burnin_action_func, 
                        initial_temperature=initial_temperature, 
                        entropy_target=entropy_target, 
                        temperature_optimizer_lr=temperature_optimizer_lr, 
                        act_deterministically=act_deterministically)
    
    def batch_act(self, batch_obs, evaluation_mode=False):
        if evaluation_mode:
            with self.eval_mode():
                assert not self.training
                return self._batch_act_eval(batch_obs)
        else:
            assert self.training
            return self._batch_act_train(batch_obs)

    def batch_observe(self, states, actions, rewards, next_states, is_terminals):
        if self.training:
            for i in range(len(states)):
                self.t += 1
                self.replay_buffer.append(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    is_state_terminal=is_terminals[i],
                    env_id=0,  # the env_id doesn't really matter for the replay buf
                )
                if is_terminals[i]:
                    self.replay_buffer.stop_current_episode(env_id=0)
                self.replay_updater.update_if_necessary(self.t)
    
    def get_qvalues(self, states, actions):
        with torch.no_grad(), pfrl.utils.evaluating(self.q_func1), \
                pfrl.utils.evaluating(self.q_func2), self.eval_mode():
            q1 = self.q_func1((states, actions))
            q2 = self.q_func2((states, actions))
        return torch.min(q1, q2)
