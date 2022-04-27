import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn

from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.td3.utils import load as load_agent

buffer_fname = 'td3_replay_buffer.pkl'
data = pickle.load(open(buffer_fname, 'rb'))

agent = TD3(state_dim=29,
            action_dim=8,
            max_action=1.,
            use_output_normalization=False,
            device=torch.device("cpu"))
agent_fname = '../antreacher_dense_save_rbuf_policy/0/td3_episode_500'
load_agent(agent, agent_fname)

####################################################
# Reduce data size for fast testing
zero_r_idx = np.where(data['reward'] == 0)
print(zero_r_idx[0])
data_idx = list(range(500)) + list(zero_r_idx[0])

data['state'] = data['state'][data_idx, :]
data['action'] = data['action'][data_idx, :]
data['reward'] = data['reward'][data_idx, :]
data['next_state'] = data['next_state'][data_idx, :]
####################################################

state_dim = data["state"].shape[1]
action_dim = data["action"].shape[1]


class QFitter(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        value = self.linear_relu_stack(x)
        return value


q_fitter = QFitter(state_dim, action_dim)

learning_rate = 0.1
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(q_fitter.parameters(), lr=learning_rate)

def optimize_model(data, target_q, q_fitter, loss_func, optimizer, num_epochs=101):
    state_action = torch.from_numpy(np.concatenate((data["state"], data["action"]), axis=1).astype(np.float32))
    for epoch in range(num_epochs):
        # Feed forward
        q_pred = q_fitter(state_action.requires_grad_())

        # Calculate the loss
        loss = loss_func(q_pred, target_q)

        # Backward propagation: caluclate gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print('epoch {}: loss = {}'.format(epoch, loss.item()))


class FQE:
    def __init__(self, data, pi_eval, learning_rate=0.01):
        self.data = data
        self.pi_eval = pi_eval
        self.state_dim = data["state"].shape[1]
        self.action_dim = data["action"].shape[1]

        self.q_fitter = QFitter(self.state_dim, self.action_dim)

        self.learning_rate = learning_rate

    def fit(self, num_iter=None, gamma=0.98):
        if num_iter is None:
            num_iter = np.ceil(1 / (1 - gamma)).astype(int)
        reward = torch.from_numpy(data["reward"].astype(np.float32)).view(-1, 1)
        state = torch.from_numpy(data["state"].astype(np.float32))
        target_q = reward
        next_action = self.pi_eval(state)
        next_state_action = torch.cat(
            (torch.from_numpy(data["next_state"].astype(np.float32)), next_action), dim=1)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.q_fitter.parameters(), lr=self.learning_rate)
        for iter in range(num_iter):
            print('Iteration: {}'.format(iter))
            optimize_model(self.data, target_q, self.q_fitter, self.loss_func, self.optimizer)
            target_q = reward + gamma * self.q_fitter(next_state_action).detach()

    def predict(self, state):
        next_action = self.pi_eval(state)
        state_policy_action = torch.cat(
            (state, next_action), dim=1)
        return self.q_fitter(state_policy_action)

fqe = FQE(data, agent.actor)
fqe.fit()

torch.save(fqe.q_fitter.state_dict(), "tmp")