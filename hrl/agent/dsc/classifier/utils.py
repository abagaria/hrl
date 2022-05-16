import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsClassifierMLP(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.l1 = nn.Linear(obs_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, obs):
        return self.l3(
            F.leaky_relu(
                self.l2(
                    F.leaky_relu(
                        self.l1(obs)
                    )
                )
            )
        )
