import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def build_classical_model(input_dim, num_classes, config):
    """
    Builds and returns a classical MLP model.
    """
    model = MLPClassifier(input_dim, num_classes)
    return model
