from torch import nn
import torch


class DistanceFunction:
    def __init__(self, state_dim, learn_online, distance_model_path=""):

        self.distance_function = DistanceNetwork(input_dim=2 * state_dim, output_dim=1)
        if learn_online:
            self.distance_optimizer = torch.optim.Adam(self.distance_function.parameters(), lr=3e-4)
        else:
            checkpoint = torch.load(distance_model_path)
            self.distance_function.load_state_dict(checkpoint['model'])
            
        self.distance_function.to(self.device)

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