import os
import ipdb
import torch
from torch import nn
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt

from hrl.agent.td3.replay_buffer import DistanceLearnerReplayBuffer


def quantile_loss(errors, quantile, k=1.0, reduce='mean'):
    loss = torch.where(
            errors < -quantile * k,
            quantile * errors.abs(),
            torch.where(
                errors > (1. - quantile) * k,
                (1. - quantile) * errors.abs(),
                (1. / (2 * k)) * errors ** 2
                )
            )
    if reduce == 'mean':
        return loss.mean()
    elif reduce == 'none':
        return loss
    else:
        raise ValueError('invalid input for `reduce`')

def quantile_l2_loss(errors, quantile, reduce='mean'):
    indicator = torch.where(
            errors < 0,
            torch.zeros_like(errors),
            torch.ones_like(errors)
            )
    loss = torch.abs(quantile - indicator) * (errors ** 2)
    if reduce == 'mean':
        return loss.mean()
    elif reduce == 'none':
        return loss
    else:
        raise ValueError('invalid input for `reduce`')

class DistanceLearner():
    def __init__(self, input_state_dim, batch_size, learning_rate, device, quantile, loss_ord=1):
        self.device = device
        self.batch_size = batch_size
        self.model = DistanceNetwork(input_dim=input_state_dim * 2, output_dim=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = DistanceLearnerReplayBuffer(input_state_dim)
        self.quantile = quantile
        if loss_ord == 1:
            self.loss_fn = quantile_loss
        elif loss_ord == 2:
            self.loss_fn = quantile_l2_loss 
        else:
            raise NotImplementedError

    def train_loop(self, train_data):
        loop_loss = IterativeAverage()
        for X, y in tqdm(train_data):
            ipdb.set_trace()
            # forward pass
            self.optimizer.zero_grad()
            X = X.float().to(self.device)
            y = y.to(self.device)
            pred = self.model(X).squeeze()
            loss = self.loss_fn(pred - y, self.quantile)

            loss.backward()
            self.optimizer.step()

            # update plotting vars
            loop_loss.add(loss.item())

        print(loop_loss.avg())
    
    @torch.no_grad()
    def forward(self, train_data):
        return self.model(train_data).squeeze()

    def save(self, savedir):
        torch.save({
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()
            }, os.path.join(savedir, f"pytorch_model.pt"))

    def experience_replay(self, states):
        self.replay_buffer.add(states)
        for _ in range(len(states)):
            self.train_loop(self.replay_buffer.sample_pairs(self.batch_size))

    def plot(self, savedir, num_batches_to_plot=4):
        x_pos, preds = [], []
        _, episode_end_idx = self.replay_buffer[0]
        start_state, _ = self.replay_buffer[(episode_end_idx + 1) % self.replay_buffer.size]
        for _ in range(num_batches_to_plot):
            start_states = torch.broadcast_to(start_state, (self.batch_size, start_state.shape[0]))
            end_states, dist = self.replay_buffer.sample(self.batch_size)
            X = torch.concatenate((start_states, end_states), axis=1)
            pred = self.forward(X).cpu().tolist()
            X = X.cpu().numpy()

            x_pos.extend(X[:, 0])
            preds.extend(pred)
        plt.scatter(x_pos, preds, alpha=0.2)
        plt.xlabel('x pos')
        plt.ylabel('Num steps from start')
        plt.savefig(f"{savedir}/quantile_map_distances.png")
        plt.close('all')



class DistanceNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DistanceNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # probs = logits.softmax(dim=1)
        return logits


class IterativeAverage():
    def __init__(self):
        self.n = 0
        self.sum = 0

    def add(self, x):
        self.n += 1
        self.sum += x

    def avg(self):
        if self.n > 0:
            return self.sum / self.n
        else:
            return 1e-8