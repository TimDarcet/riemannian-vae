"""
pytorch-lightning VAE model for MVA computational stats course
Authors:
Timothee Darcet timothee.darcet@gmail.com
Clement Grisi
resnet18-VAE code inspired by
towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
"""

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder
from torch import nn
import torch
from utils import kl_divergence, jacobian



class VAE(pl.LightningModule):
    """A simple VAE class with Resnet18 backends using Pytorch-lightning"""
    # Default params are for CIFAR10
    def __init__(self, latent_dim=256, input_height=32, input_channels=3):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.encoder.conv1 = nn.Conv2d(input_channels, 64,
                                       kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1), bias=False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )
        self.decoder.conv1 = nn.Conv2d(64, input_channels,
                                       kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1), bias=False)
        # Output probability distribution std
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        return self.decoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, data, _):
        x, _ = data
        batch_size = x.shape[0]
        # Get latent distrib
        mu, log_var = self.encoder(x).view(batch_size, 2, -1).transpose(0, 1)
        std = torch.exp(log_var / 2)
        # Sample it
        latent_z = torch.distributions.Normal(mu, std).rsample()
        # Get decoded distrib
        x_hat = self(latent_z)
        # Reconstruction likelihood
        recon_lh = torch.distributions.Normal(x_hat, torch.exp(self.log_scale)).log_prob(x).sum()
        # KL divergence
        kl_div = kl_divergence(latent_z, mu, std)
        # ELBO
        elbo = (kl_div - recon_lh).mean()
        # log metrics
        self.log_dict({
            'train_elbo': elbo,
            'train_kl': kl_div.mean(),
            'train_recon_likelihood': recon_lh.mean()
        }, prog_bar=True)
        return elbo

    def validation_step(self, data, _):
        x, _ = data
        batch_size = x.shape[0]
        # Get latent distrib
        mu, log_var = self.encoder(x).view(batch_size, 2, -1).transpose(0, 1)
        std = torch.exp(log_var / 2)
        # Sample it
        latent_z = torch.distributions.Normal(mu, std).rsample()
        # Get decoded distrib
        x_hat = self(latent_z)
        # Reconstruction likelihood
        recon_lh = torch.distributions.Normal(x_hat, torch.exp(self.log_scale)).log_prob(x).sum()
        # KL divergence
        kl_div = kl_divergence(latent_z, mu, std)
        # ELBO
        elbo = (kl_div - recon_lh).mean()
        # log metrics
        self.log_dict({
            'val_elbo': elbo,
            'val_kl': kl_div.mean(),
            'val_recon_likelihood': recon_lh.mean()
        }, prog_bar=True)
        return elbo

    def gen_image(self):
        """Generate a random image by sampling the latent distribution"""
        mu = torch.zeros(self.latent_dim)
        std = torch.ones(self.latent_dim)
        latent_z = torch.distributions.Normal(mu, std).rsample()
        return self(torch.unsqueeze(latent_z, 0))

    def random_walk(self, length=10, std=1):
        """Generate a sequence of random images
        by doing a gaussian random walk starting at zero in the latent space.
        Uses a diagonal covariance matrix equal to std times the identity."""
        latent_z = torch.zeros((length, self.latent_dim))
        stds = std * torch.ones(self.latent_dim)
        for i in range(1, length):
            latent_z[i] = torch.distributions.Normal(latent_z[i - 1], stds).rsample()
        return self(latent_z)

    # def generator_jacobian(self, latent_z):
    #     """Calculate the jacobian of the generator network at latent_z"""
    #     latent_z.requires_grad = True
    #     output = torch.flatten(self(torch.unsqueeze(latent_z, 0)))
    #     gen_jacobian = jacobian(output, latent_z)
    #     return gen_jacobian

    def generator_jacobian(self, latent_z):
        """Calculate the jacobian of the generator network at latent_z"""
        def flat_gen(z):
            return torch.flatten(self(torch.unsqueeze(z, 0)))
        latent_z.requires_grad = True
        return torch.autograd.functional.jacobian(flat_gen, latent_z, strict=True)

    def riemann_walk(self, length=10, std=1):
        """Generate a sequence of random images
        by doing a gaussian random walk starting at zero in the latent space.
        Uses the Riemannian metric as covariance."""
        latent_z = torch.zeros((length, self.latent_dim))
        for i in range(1, length):
            jac = self.generator_jacobian(latent_z[i - 1])
            metric = jac.T @ jac
            # TODO: Given the metric, what should I use as covariance matrix?
            # The metric itself seems fine
            # Add in a small identity matrix to handle singular covariance
            covariance = std * (metric + 0.01 * torch.eye(self.latent_dim))
            distrib = torch.distributions.MultivariateNormal(latent_z[i - 1], covariance)
            latent_z[i] = distrib.rsample()
        return self(latent_z)
