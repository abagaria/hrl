from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Sequence
import pfrl
from pfrl.agents import CategoricalDoubleDQN
from pfrl.agents.categorical_dqn import compute_value_loss
from pfrl.agents.dqn import batch_experiences
import torch
from pfrl.replay_buffers import PrioritizedReplayBuffer

from hrl.agent.dq_demonstrations.combined_replay_buffer import CombinedPrioritizedReplayBuffer

class SupervisedCategoricalDoubleDQN(CategoricalDoubleDQN):

    def __init__(self, 
            q_function: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            replay_buffer: pfrl.replay_buffer.AbstractReplayBuffer, 
            gamma: float, 
            explorer: pfrl.explorer.Explorer, 
            gpu: Optional[int] = None, 
            replay_start_size: int = 50000, 
            minibatch_size: int = 32, 
            update_interval: int = 1, 
            target_update_interval: int = 10000, 
            clip_delta: bool = True, 
            phi: Callable[[Any], Any] = ..., 
            target_update_method: str = "hard", 
            soft_update_tau: float = 0.01, 
            n_times_update: int = 1, 
            batch_accumulator: str = "mean", 
            episodic_update_len: Optional[int] = None, 
            logger: Logger = ..., 
            batch_states: Callable[[Sequence[Any], torch.device, Callable[[Any], Any]], Any] = ..., 
            recurrent: bool = False, 
            max_grad_norm: Optional[float] = None,
            supervised_batchsize=32):
        super().__init__(q_function, 
            optimizer, 
            replay_buffer, 
            gamma, 
            explorer, 
            gpu, 
            replay_start_size, 
            minibatch_size, 
            update_interval, 
            target_update_interval, 
            clip_delta, 
            phi, 
            target_update_method, 
            soft_update_tau, 
            n_times_update, 
            batch_accumulator, 
            episodic_update_len, 
            logger, 
            batch_states, 
            recurrent, 
            max_grad_norm)

        assert isinstance(self.replay_buffer, CombinedPrioritizedReplayBuffer)
        self.supervised_batchsize = 32

    def update(
        self,
        experiences: List[List[Dict[str, Any]]],
        errors_out:Optional[list] = None
    ) -> None:
        """ Update model from experiences.
        
        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
                  - supervised_lambda (float): L2 weight coefficient (experience loss weight).
                    Should be 0 for experience buffer and between 1/0 for trajectory 
                    buffer. 
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.
        Returns:
            None
        
         """

        has_weight = "weight" in experiences[0][0]
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states
        )
        exp_batch["supervised_lambda"] = torch.tensor(
            [elem[0]["supervised_lambda"] for elem in experiences],
            device=self.device,
            dtype=torch.float32
        )
        if has_weight:
            exp_batch["weight"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32
            )
            if errors_out is None:
                errors_out=[]
        loss = self._overall_loss(exp_batch, errors_out=errors_out)
        if has_weight:
            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1


    def _overall_loss(self, exp_batch, errors_out=None):
        dq_loss = self._compute_dq_loss(exp_batch, errors_out)
        supervised_loss = self._compute_supervised_loss(exp_batch)

        loss =  dq_loss \
                + supervised_loss

        return loss

    def _compute_dq_loss(self, exp_batch, errors_out=None):
        return super()._compute_loss(exp_batch, errors_out)

    def _compute_supervised_loss(self, exp_batch):

        batch_size = exp_batch["reward"].shape[0]

        batch_state = exp_batch["state"]
        batch_action = exp_batch["action"]
        batch_lambda = exp_batch["supervised_lambda"]

        model_qout = self.model(batch_state)
        model_value = model_qout.max
        model_actions = model_qout.greedy_actions
        margin_loss = self._margin_loss(batch_action, model_actions)

        expert_q = self.model(batch_state).evaluate_actions(batch_action)

        eltwise_loss = torch.mul(batch_lambda, model_value + margin_loss - expert_q)

        eltwise_loss = torch.reshape(
            eltwise_loss,
            (batch_size, 1)
        )

        return compute_value_loss(
            eltwise_loss,
            batch_accumulator=self.batch_accumulator
        )

    @staticmethod
    def _margin_loss(expert_actions, actions):

        """ if expert action == action: 0.8 else 0 """

        loss = torch.eq(expert_actions, actions)
        # invert boolean matrix from torch.eq()
        loss = ~loss
        loss = loss.float()

        return loss*0.8

    def add_demonstration_trajectory(self, trajectories):

        for trajectory in trajectories:
            self.replay_buffer.append(supervised_lambda=1, **trajectory)

    def bootstrap_train_step(self):
        transitions = self.replay_buffer.sample(self.supervised_batchsize)
        self.replay_updater.update_func(transitions)



