# This script is for loading and processing the dataset
import numpy as np
from scipy import special
import torch
from torch.utils.data import Dataset

# construct a dataset of the jackknife sample means of the read in samples
class SimpleGreensDataset(Dataset):
    
    def __init__(self, filepath, transform=None, dtype = torch.float32, trim_beta = True):
        
        # trim off tau = beta value
        samples = np.loadtxt(filepath, delimiter=",")
        if trim_beta:
            samples = samples[:,0:-1]
        self.data = torch.tensor(samples, dtype = dtype)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transformation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
            
        return x

# construct a dataset of the jackknife sample means of the read in samples
class JackknifeGreensDataset(Dataset):
    
    def __init__(self, filepath, transform=None, dtype = torch.float32, trim_beta = True):
        
        # Load all Green's functions at once, trimming off tau = beta value
        samples = np.loadtxt(filepath, delimiter=",")
        if trim_beta:
            samples = samples[:,0:-1]
        
        # get the number of samples
        N_samples = samples.shape[0]
        
        # get the average of all the samples i.e. the sample mean
        sample_mean = np.mean(samples, axis = 0)
        
        # calculate the jackknife sample means. This expression updates the sample
        # mean to reflect removing one sample, doing so for each input sample.
        jackknife_sample_means = (N_samples*sample_mean - samples)/(N_samples - 1)
        
        self.data = torch.tensor(jackknife_sample_means, dtype = dtype)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transformation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
            
        return x
    

# construct a dataset of the means of bootstrap samples generated based on the read in samples
class BootstrapGreensDataset(Dataset):
    
    def __init__(self, filepath, N_bootstrap=None, transform=None, dtype = torch.float32, trim_beta = True):
        
        # Load all the Green's function as once, trimming off tau = beta value
        samples = np.loadtxt(filepath, delimiter=",")
        if trim_beta:
            samples = samples[:,0:-1]
        
        # get the number of samples
        N_samples = samples.shape[0]
        
        # length of imaginary-time axis
        Ltau = samples.shape[1]
        
        # Set the number of bootstrap samples
        if N_bootstrap == None:
            N_bootstrap = N_samples
            
        # initialize array to contain bootstrap sample means
        bootstrap_sample_means = np.empty((N_bootstrap, Ltau))
        
        # iterate over bootstrap samples
        for i in range(N_bootstrap):
            
            # generate bootstrap sample
            indices = np.random.randint(0, N_samples, N_samples)
            bootstrap_sample = samples[indices]
            
            # calculate the mean of current bootstrap sample
            bootstrap_sample_means[i] = bootstrap_sample.mean(axis=0)
            
        self.data = torch.tensor(bootstrap_sample_means, dtype = dtype)
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transformation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
            
        return x
    
# construct a dataset for pretraining
class PretrainingDataset(Dataset):

    def __init__(
        self,
        N, Nnodes, beta, dtau, sqrt_cov,
        mu_mu, mu_sigma, sigma_mu, sigma_sigma,
        transform=None,
        dtype=torch.float32,
        device="cpu"
    ):
 
        self.N = N
        self.Nnodes = Nnodes
        self.beta = torch.tensor(beta, dtype=dtype, device=device)
        self.dtau = torch.tensor(dtau, dtype=dtype, device=device)
        Ltau = int(np.round(beta / dtau))
        self.Ltau = Ltau
        self.taus = torch.linspace(0.0, self.beta-self.dtau, Ltau, dtype=dtype, device=device)
        self.sqrt_cov = torch.tensor(sqrt_cov, dtype=dtype, device=device)
        self.mu_mu = torch.tensor(mu_mu, dtype=dtype, device=device)
        self.mu_sigma = torch.tensor(mu_sigma, dtype=dtype, device=device)
        self.sigma_mu = torch.tensor(sigma_mu, dtype=dtype, device=device)
        self.sigma_sigma = torch.tensor(sigma_sigma, dtype=dtype, device=device)
        x, w = special.roots_hermite(Nnodes)
        self.x = torch.tensor(x, dtype=dtype, device=device)
        self.w = torch.tensor(w, dtype=dtype, device=device)
        self.sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        self.sqrt_pi = torch.sqrt(torch.tensor(np.pi, dtype=dtype, device=device))
        self.transform = transform

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        device = self.sqrt_cov.device

        # ------------------------------------------------------------------
        # Sample mean and standard deviation
        # ------------------------------------------------------------------
        # mu_mu : ()
        # mu_sigma : ()
        # torch.randn(1) : (1,)
        # ==> mu : (1,)
        mu = self.mu_mu + self.mu_sigma * torch.randn(1, device=device)

        # sigma_mu : ()
        # sigma_sigma : ()
        # torch.randn(1) : (1,)
        # ==> sigma : (1,)
        sigma = self.sigma_mu + self.sigma_sigma * torch.randn(1, device=device)

        # ------------------------------------------------------------------
        # Support points for Gaussian via Gauss–Hermite nodes
        # ------------------------------------------------------------------
        # x : (Nnodes,)
        # sqrt2 : ()
        # mu : (1,)
        # sigma : (1,)
        # ==> omegas : (Nnodes,)
        omegas = mu + self.sqrt2 * sigma * self.x.to(device)

        # ------------------------------------------------------------------
        # Construct exponent arguments
        # ------------------------------------------------------------------
        # taus : (Ltau,)
        # beta : ()
        # omegas : (Nnodes,)
        #
        # taus[:, None] : (Ltau, 1)
        # omegas[None, :] : (1, Nnodes)
        #
        # ==> a : (Ltau, Nnodes)
        # ==> b : (Ltau, Nnodes)
        a = self.taus[:, None] * omegas[None, :]
        b = (self.beta - self.taus)[:, None] * omegas[None, :]

        # ------------------------------------------------------------------
        # Stable denominator: log(exp(a) + exp(b))
        # ------------------------------------------------------------------
        # stack([a, b], dim=0) : (2, Ltau, Nnodes)
        # logsumexp(dim=0) : (Ltau, Nnodes)
        log_denom = torch.logsumexp(
            torch.stack([a, b], dim=0),
            dim=0
        )

        # ------------------------------------------------------------------
        # Gauss–Hermite quadrature
        # ------------------------------------------------------------------
        # w : (Nnodes,)
        # exp(-log_denom) : (Ltau, Nnodes)
        #
        # w[None, :] : (1, Nnodes)
        # ==> I : (Ltau, Nnodes)
        I = self.w[None, :].to(device) * torch.exp(-log_denom)

        # Sum over nodes
        # I : (Ltau, Nnodes)
        # ==> G_tau : (Ltau,)
        G_tau = torch.sum(I, dim=1) / self.sqrt_pi.to(device)

        # ------------------------------------------------------------------
        # Add correlated noise
        # ------------------------------------------------------------------
        # taus : (Ltau,)
        # ==> R : (Ltau,)
        R = torch.randn_like(self.taus)

        # sqrt_cov : (Ltau, Ltau)
        # R : (Ltau,)
        # ==> G_tau_noisy : (Ltau,)
        G_tau_noisy = G_tau + self.sqrt_cov @ R

        # ------------------------------------------------------------------
        # Optional transform
        # ------------------------------------------------------------------
        # G_tau_noisy : (Ltau,)
        if self.transform:
            G_tau_noisy = self.transform(G_tau_noisy)

        # mu, sigma are scalars stored as (1,) tensors
        return mu.item(), sigma.item(), G_tau_noisy
