"""Hold encoder classes"""

from torch import nn
import torch


class LeNet5Encoder(nn.Module):
    """LeNet5 architecture"""
    def __init__(self, latent_dim, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=120, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=2 * latent_dim)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = torch.flatten(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
