import torch
import torch.nn as nn
from pfrl import nn as pnn


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


class ImageCNN(torch.nn.Module):
    def __init__(self, device, n_input_channels=1):
        super().__init__()

        self.device = device

        # SmallAtariCNN has 256 output channels
        self.feature_extractor = pnn.SmallAtariCNN(
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