"""Hold utility functions"""

import torch


def kl_divergence(z, mu, std):
    """Calculate Monte carlo KL divergence"""
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # return kl
    return (log_qzx - log_pz).sum(-1)
