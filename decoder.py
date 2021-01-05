"""Hold decoder classes"""

from torch import nn
import torch


class LeNet5Decoder(nn.Module):
    """Reverse LeNet5 architecture"""
    def __init__(self, latent_dim, output_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=400, bias=True)
        self.pool1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2)
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=1)
        self.pool2 = nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=output_channels, kernel_size=5, stride=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.shape[0], 16, 5, 5)
        x = torch.tanh(self.pool1())
        x = torch.tanh(self.pool2(self.conv1(x)))
        x = torch.sigmoid(self.conv2(x))
        return x
