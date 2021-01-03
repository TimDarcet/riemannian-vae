"""
Hold utility functions.
Differentitation functions from https://github.com/pytorch/pytorch/issues/8304
"""

import matplotlib.pyplot as plt
import torch
from tqdm import trange


def normalize_image(image):
    """Normalize values from (0, 1) to (-1, 1) range"""
    return (image * 2 - 1).clamp(-1, 1)

def denormalize_image(image):
    """Normalize values from (-1, 1) to (0, 1) range"""
    return (image * .5 + .5).clamp(0, 1)

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

def display_image(image, ax=plt):
    """Display an image with matplotlib"""
    ax.axis("off")
    ax.imshow(denormalize_image(image.cpu().detach()).permute(1, 2, 0))

def display_images(imbatch):
    """Display a batch of images with matplotlib"""
    n_im = imbatch.shape[0]
    n_rows = n_im // 4 + 1
    _, axes = plt.subplots(n_rows, 4, squeeze=False, figsize=(4 * 6.4, n_rows * 4.8))
    for r in range(n_rows):
        for c in range(4):
            axes[r][c].axis("off")
    for i in range(imbatch.shape[0]):
        display_image(imbatch[i], axes[i // 4][i % 4])

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True,
                               only_inputs=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs;
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0])
    for i in trange(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        tmp = gradient(y, x, grad_outputs=grad_outputs)
        jac[i] = gradient(y, x, grad_outputs=grad_outputs)
    return jac
