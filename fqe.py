import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import os
import time

import torch
import torch.nn as nn

from hrl.agent.td3.TD3AgentClass import TD3
from hrl.agent.td3.utils import load as load_agent

import argparse

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

# def sample_batch(data, batch_size=50):
#     zero_r_idx = np.where(data['reward'] == 0)
#     sample_idx = zero_r_idx + random.sample(range(len(data["state"])), batch_size)
#
#     batch_data = {}
#     batch_data['state'] = data['state'][sample_idx, :]
#     batch_data['action'] = data['action'][sample_idx, :]
#     batch_data['reward'] = data['reward'][sample_idx, :]
#     batch_data['next_state'] = data['next_state'][sample_idx, :]
#     return batch_data

def generate_sample_idx(data, batch_size=50):
    zero_r_idx = np.where(data['reward'] == 1)
    sample_idx = list(zero_r_idx[0]) + random.sample(range(len(data["state"])), batch_size - len(zero_r_idx[0]))
    return sample_idx

def optimize_model(data, target_q, q_fitter, loss_func, optimizer, batch_size, num_epochs):
    state_action = torch.from_numpy(np.concatenate((data["state"], data["action"]), axis=1).astype(np.float32))
    for epoch in range(num_epochs):
        # Sample a batch with the transitions reaching the goal
        sample_idx = generate_sample_idx(data, batch_size=batch_size)

        # Feed forward
        q_pred = q_fitter(state_action[sample_idx, :].requires_grad_())

        # Calculate the loss
        loss = loss_func(q_pred, target_q[sample_idx])

        # Backward propagation: caluclate gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        if epoch % 10 == 0 or epoch == num_epochs-1:
            print('epoch {}: loss = {}'.format(epoch, loss.item()))


class FQE:
    def __init__(self,
                 data,
                 pi_eval,
                 learning_rate=0.01,
                 device='cpu',
                 exp_name="tmp"):
        self.data = data
        self.pi_eval = pi_eval
        self.state_dim = data["state"].shape[1]
        self.action_dim = data["action"].shape[1]

        self.q_fitter = QFitter(self.state_dim, self.action_dim).to(device)

        self.learning_rate = learning_rate
        self.exp_name = exp_name


    def fit(self, num_iter, gamma, batch_size, num_epochs, save_interval=np.inf):
        reward = torch.from_numpy(self.data["reward"].astype(np.float32)).view(-1, 1)
        state = torch.from_numpy(self.data["state"].astype(np.float32))
        done = torch.from_numpy(data["done"].astype(np.float32))
        target_q = reward
        next_action = self.pi_eval(state)
        print('hi')
        next_state_action = torch.cat(
            (torch.from_numpy(self.data["next_state"].astype(np.float32)), next_action), dim=1)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.q_fitter.parameters(), lr=self.learning_rate)
        for iter in range(num_iter):
            print('Iteration: {}'.format(iter))
            optimize_model(self.data, target_q, self.q_fitter, self.loss_func, self.optimizer, batch_size, num_epochs)
            if (save_interval < np.inf and iter % save_interval == 0) or iter == num_iter-1:
                torch.save(self.q_fitter.state_dict(), "saved_results/{}/weights_{}".format(self.exp_name, iter))
            target_q = reward + (1. - done) * gamma * self.q_fitter(next_state_action).detach()

    def predict(self, state):
        next_action = self.pi_eval(state)
        state_policy_action = torch.cat(
            (state, next_action), dim=1)
        return self.q_fitter(state_policy_action)


if __name__ == '__main__':
    print('start')
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--save_interval', type=int, default=np.inf)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_iter', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    buffer_fname = 'td3_replay_buffer.pkl'
    data = pickle.load(open(buffer_fname, 'rb'))
    data["reward"] += 1
    data["done"] = (data["reward"] == 1).astype(float)

    agent = TD3(state_dim=29,
                action_dim=8,
                max_action=1.,
                use_output_normalization=False,
                device=torch.device(args.device))
    agent_fname = 'antreacher_dense_save_rbuf_policy/0/td3_episode_500'
    load_agent(agent, agent_fname)

    ####################################################
    # # Reduce data size for fast testing
    # zero_r_idx = np.where(data['reward'] == 0)
    # print(zero_r_idx[0])
    # data_idx = list(range(500)) + list(zero_r_idx[0])
    #
    # data['state'] = data['state'][data_idx, :]
    # data['action'] = data['action'][data_idx, :]
    # data['reward'] = data['reward'][data_idx, :]
    # data['next_state'] = data['next_state'][data_idx, :]
    ####################################################

    state_dim = data["state"].shape[1]
    action_dim = data["action"].shape[1]

    # q_fitter = QFitter(state_dim, action_dim)
    #
    # learning_rate = 0.01
    # loss_func = nn.MSELoss()
    # optimizer = torch.optim.SGD(q_fitter.parameters(), lr=learning_rate)

    if not os.path.exists('saved_results/{}/'.format(args.exp_name)):
        os.makedirs('saved_results/{}/'.format(args.exp_name))

    print('Created folder.')
    fqe = FQE(data,
              agent.actor,
              learning_rate=args.learning_rate,
              exp_name=args.exp_name,
              device=args.device)
    print('Created fqe.')
    fqe.fit(args.num_iter,
            gamma=args.gamma,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval)
    print('Fitted fqe.')

    with open('saved_results/{}/args.txt'.format(args.exp_name), 'w') as f:
        for arg in vars(args):
            f.write("{}: {}".format(arg, getattr(args, arg)))
            f.write("\n")
        t1 = time.time()
        f.write("\n")
        f.write("Runtime: {0:.2f} minutes.".format((t1-t0)/60))
    print('done')