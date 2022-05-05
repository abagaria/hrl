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

import pdb

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

def chunked_policy_prediction(policy, states, action_dim, chunk_size=100):
    pdb.set_trace()
    data_size = states.size(dim=0)
    actions = torch.zeros((data_size, action_dim))
    num_whole_chunks = data_size // chunk_size
    for i in range(num_whole_chunks):
        actions[i*chunk_size:(i+1)*chunk_size-1, :] = policy(states[i*chunk_size:(i+1)*chunk_size-1, :])
    if data_size % chunk_size != 0:
        actions[num_whole_chunks*chunk_size:, :] = policy(states[num_whole_chunks*chunk_size:, :])
    return actions



def generate_sample_idx(data, batch_size=50):
    zero_r_idx = np.where(data['reward'] == 1)
    sample_idx = list(zero_r_idx[0]) + random.sample(range(len(data["state"])), batch_size - len(zero_r_idx[0]))
    return sample_idx

class FQE:
    def __init__(self,
                 data,
                 pi_eval,
                 learning_rate=0.01,
                 device='cpu',
                 exp_name="tmp"):
        self.pi_eval = pi_eval
        self.state_dim = data["state"].shape[1]
        self.action_dim = data["action"].shape[1]
        self.data_size = data["action"].shape[0]

        self.q_fitter = QFitter(self.state_dim, self.action_dim).to(device)

        self.learning_rate = learning_rate
        self.exp_name = exp_name

        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_fitter.parameters(), lr=self.learning_rate)

        self.reward = torch.from_numpy(data["reward"].astype(np.float32)).view(-1, 1)
        self.state = torch.from_numpy(data["state"].astype(np.float32))
        self.state_action = torch.from_numpy(np.concatenate((data["state"], data["action"]), axis=1).astype(np.float32))
        self.done = torch.from_numpy(data["done"].astype(np.float32))
        next_action = chunked_policy_prediction(self.pi_eval, self.state, self.action_dim)
        self.next_state_action = torch.cat(
            (torch.from_numpy(data["next_state"].astype(np.float32)), next_action), dim=1)

    def optimize_model(self, gamma, batch_size, num_batches):

        for idx_batch in range(num_batches):
            # Sample a batch with the transitions reaching the goal
            sample_idx = generate_sample_idx(data, batch_size=batch_size)

            # Feed forward
            q_pred = self.q_fitter(self.state_action[sample_idx, :].requires_grad_())

            # Compute target
            target_q = self.reward[sample_idx, :] + (1. - self.done[sample_idx, :]) * gamma * self.q_fitter(
                self.next_state_action[sample_idx, :]).detach()

            # Calculate the loss
            loss = self.loss_func(q_pred, target_q)

            # Backward propagation: caluclate gradients
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Clear gradients
            self.optimizer.zero_grad()

            if idx_batch % 10 == 0 or idx_batch == num_batches - 1:
                print('Batch {}: loss = {}'.format(idx_batch, loss.item()))

    def fit(self, num_iter, gamma, batch_size, num_batches, save_interval=np.inf):
        for iteration in range(num_iter):
            print('Iteration: {}'.format(iteration))
            self.optimize_model(gamma, batch_size, num_batches)
            if (save_interval < np.inf and iteration % save_interval == 0) or iteration == num_iter-1:
                torch.save(self.q_fitter.state_dict(), "saved_results/{}/weights_{}".format(self.exp_name, iteration))


    def predict(self, state):
        next_action = self.pi_eval(state)
        state_policy_action = torch.cat(
            (state, next_action), dim=1)
        return self.q_fitter(state_policy_action)


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--save_interval', type=int, default=np.inf)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_iter', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_batches', type=int, default=100)
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

    fqe = FQE(data,
              agent.actor,
              learning_rate=args.learning_rate,
              exp_name=args.exp_name,
              device=args.device)
    fqe.fit(args.num_iter,
            gamma=args.gamma,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            save_interval=args.save_interval)

    with open('saved_results/{}/args.txt'.format(args.exp_name), 'w') as f:
        for arg in vars(args):
            f.write("{}: {}".format(arg, getattr(args, arg)))
            f.write("\n")
        t1 = time.time()
        f.write("\n")
        f.write("Runtime: {0:.2f} minutes.".format((t1-t0)/60))