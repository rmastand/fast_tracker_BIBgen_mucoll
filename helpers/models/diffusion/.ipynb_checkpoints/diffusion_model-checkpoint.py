import numpy as np 
import torch
import torch.nn as nn


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        # reshape to (B, C, 1) for broadcasting in 1D convs
        return self.dense(x)[..., None]


class ScoreNet1D(nn.Module):
    """A time-dependent score-based model built upon a 1D U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[8, 8, 16], embed_dim=32):
        super().__init__()

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # -------------------------
        # Encoding path
        # -------------------------
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=2, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(2, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=1, stride=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(2, channels[1])

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size=1, stride=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(2, channels[2])

        # -------------------------
        # Decoding path
        # -------------------------
        self.tconv4 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size=1, stride=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[1])
        self.tgnorm4 = nn.GroupNorm(2, channels[1])

        self.tconv3 = nn.ConvTranspose1d(
            channels[1] + channels[1],
            channels[0],
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.dense6 = Dense(embed_dim, channels[0])
        self.tgnorm3 = nn.GroupNorm(2, channels[0])

        self.tconv2 = nn.ConvTranspose1d(
            channels[0] + channels[0],
            1,
            kernel_size=2,
            stride=1,
            bias=False,
        )

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        """
        x: (B, 1, L)
        t: (B,)
        """
        embed = self.act(self.embed(t))

        # -------- Encoder --------
        h1 = self.conv1(x)
        h1 = h1 + self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        # -------- Decoder --------
        h = self.tconv4(h3)
        h = h + self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat([h, h2], dim=1))
        h = h + self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h1], dim=1))

        # -------- Normalize score --------
        h = h / self.marginal_prob_std(t)[:, None, None]
        return h


def marginal_prob_std(t, sigma=25., device='cuda'):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.  

    Returns:
    The standard deviation.
    """    
    t = t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma=25., device='cuda'):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return sigma**t.to(device)


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    # x is expected to be (B, L); make it (B, 1, L)
    if x.ndim == 2:
        x = x.unsqueeze(1)

    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)

    perturbed_x = x + z * std[:, None, None]
    score = model(perturbed_x, random_t)

    loss = torch.mean(
        torch.sum((score * std[:, None, None] + z) ** 2, dim=(1, 2))
    )
    return loss
