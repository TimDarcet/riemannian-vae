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
from utils import kl_divergence


class VAE(pl.LightningModule):
    """A simple VAE class with Resnet18 backends using Pytorch-lightning"""
    def __init__(self, latent_dim=256, input_height=32):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )
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
        })
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
        })
        return elbo
