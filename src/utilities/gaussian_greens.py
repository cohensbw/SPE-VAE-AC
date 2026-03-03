import torch
import numpy as np
from scipy import special

def hermite_nodes_weights(N_nodes, device=None, dtype=torch.float64):
    """
    Precompute Gauss–Hermite nodes and weights once using SciPy,
    then convert to torch tensors.
    """
    x_np, w_np = special.roots_hermite(N_nodes)
    x = torch.tensor(x_np, device=device, dtype=dtype)
    w = torch.tensor(w_np, device=device, dtype=dtype)
    return x, w


def gaussian_greens(mu, sigma, taus, beta, N_nodes):

    # get nodes x and weights w for gauss-hermite quadrature
    # x: (N_nodes)
    # w: (N_nodes)
    x, w = special.roots_hermite(N_nodes)

    # map nodes x to frequencies omega
    # omegas: (N_normals, N_nodes)
    omegas = mu[:, None] + np.sqrt(2.0) * sigma[:, None] * x[None, :]

    # construct the integrand using a numerically stable log-sum-exp approach
    # integrand: (N_normals, L_tau, N_nodes)
    a = taus[None, :, None] * omegas[:, None, :]
    b = (beta - taus[None, :, None]) * omegas[:, None, :]
    log_denom = special.logsumexp([a, b], axis=0)
    I = w[None, None, :] * np.exp(-log_denom)

    # sum of omega axis of integrand to evaluate integral to get G(tau)
    # G_tau: (N_normals, L_tau)
    G_tau = np.sum(I, axis=2) / np.sqrt(np.pi)

    return G_tau

def gaussian_greens_torch(mu, sigma, taus, beta, x, w):
    """
    Torch equivalent of gaussian_greens

    Parameters
    ----------
    mu      : (N_normals,)
    sigma   : (N_normals,)
    taus    : (L_tau,)
    beta    : scalar
    x, w    : (N_nodes,) Gauss–Hermite nodes and weights
    """

    # omegas: (N_normals, N_nodes)
    omegas = mu[:, None] + torch.sqrt(torch.tensor(2.0, device=mu.device)) \
             * sigma[:, None] * x[None, :]

    # a, b: (N_normals, L_tau, N_nodes)
    a = taus[None, :, None] * omegas[:, None, :]
    b = (beta - taus[None, :, None]) * omegas[:, None, :]

    # log_denom: (N_normals, L_tau, N_nodes)
    log_denom = torch.logsumexp(
        torch.stack([a, b], dim=0),
        dim=0
    )

    # integrand
    I = w[None, None, :] * torch.exp(-log_denom)

    # G_tau: (N_normals, L_tau)
    G_tau = torch.sum(I, dim=2) / torch.sqrt(torch.tensor(torch.pi, device=mu.device))

    return G_tau


def add_cov_noise(G_tau, sqrt_cov, N_samples):

    # G_tau: (N_normals, L_tau)
    (N_normals, L_tau) = G_tau.shape

    # construct normal random vectors R
    # R: (N_normals, N_samples, L_tau)
    R = np.random.randn((N_normals, N_samples, L_tau))

    # correlated noise based on the square root of a covariance matrix
    # sqrt_cov: (L_tau, L_tau)
    # noise: (N_normals, N_samples, L_tau)
    noise = np.einsum('ij,nsj->nsi', sqrt_cov, R)

    # construct noisy G(tau) samples
    # G_tau_noisy: (N_normals, N_samples, L_tau)
    G_tau_noisy = G_tau[:, None, :] + noise

    return G_tau_noisy

def add_cov_noise_torch(G_tau, sqrt_cov, N_samples):
    """
    Torch equivalent of add_cov_noise

    Parameters
    ----------
    G_tau     : (N_normals, L_tau)
    sqrt_cov  : (L_tau, L_tau)
    N_samples : int
    """

    N_normals, L_tau = G_tau.shape
    device = G_tau.device
    dtype = G_tau.dtype

    # R: (N_normals, N_samples, L_tau)
    R = torch.randn(
        (N_normals, N_samples, L_tau),
        device=device,
        dtype=dtype
    )

    # noise: (N_normals, N_samples, L_tau)
    noise = torch.einsum('ij,nsj->nsi', sqrt_cov, R)

    # G_tau_noisy: (N_normals, N_samples, L_tau)
    G_tau_noisy = G_tau[:, None, :] + noise

    return G_tau_noisy
