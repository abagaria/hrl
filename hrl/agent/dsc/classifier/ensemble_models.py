import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl.initializers import init_chainer_default


class PositionMLP(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.l1 = nn.Linear(2, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 1)
        
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

        self.to(device)
        
    def forward(self, inputs):
        x = self.l1(inputs)
        x = nn.functional.selu(x)
        x = self.l2(x)
        x = nn.functional.selu(x)
        x = self.l3(x)
        return x

    @torch.no_grad()
    def predict(self, inputs):
        logits = self.forward(inputs)
        probabilities = torch.sigmoid(logits)
        classes = probabilities >= 0.5
        return classes.float().cpu().numpy().squeeze()


def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class SmallAtariCNN(nn.Module):
    """Small CNN module proposed for DQN in NeurIPS DL Workshop, 2013.

    See: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self, n_input_channels=4, n_output_channels=256, activation=F.relu, bias=0.1
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 16, 8, stride=4),
                nn.Conv2d(16, 32, 4, stride=2),
            ]
        )
        self.output = nn.Linear(2592, n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        try:
            h_flat = h.view(h.size(0), -1)
        except:  # Stay away from transpose() and view() if possible!
            # h_flat = h.contiguous().view(h.size(0), -1)
            ipdb.set_trace()
        return self.activation(self.output(h_flat))



class ImageCNN(torch.nn.Module):
    def __init__(self, device, n_input_channels=1):
        super().__init__()

        self.device = device

        # SmallAtariCNN has 256 output channels
        self.feature_extractor = SmallAtariCNN(
            n_input_channels=n_input_channels
        )

        self.model = nn.Sequential(
            self.feature_extractor,
            nn.ReLU(),
            nn.Linear(self.feature_extractor.n_output_channels, 1)
        )

        self.to(device)

    def forward(self, image):

        # Add batch and channel dimensions to lone inputs
        if image.shape == (84, 84):
            image = image.unsqueeze(0).unsqueeze(0)
        
        # Add channel dimension if it is missing
        elif image[0].shape == (84, 84):
            image = image.unsqueeze(1)
            
        return self.model(image)

    @torch.no_grad()
    def predict(self, inputs):
        logits = self.forward(inputs)
        probabilities = torch.sigmoid(logits)
        classes = probabilities >= 0.5
        return classes.float().cpu().numpy().squeeze()

    @torch.no_grad()
    def extract_features(self, image_tensor):
        feature_tensor = self.forward(image_tensor)
        return feature_tensor.view(
            feature_tensor.size(0), -1
        ).cpu().numpy()


class EnsembleModel(torch.nn.Module):
    def __init__(self,
                 base_model : torch.nn.Module,
                 prior_model : torch.nn.Module,
                 prior_scale : float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.prior_model = prior_model
        self.prior_scale = prior_scale

    def forward(self, inputs):
        with torch.no_grad():
            prior_out = self.prior_model(inputs)
            prior_out = prior_out.detach()
        model_out = self.base_model(inputs)
        return model_out + (self.prior_scale * prior_out)
