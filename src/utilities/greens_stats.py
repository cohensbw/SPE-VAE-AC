import numpy as np

def calculate_cov(datafile):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,1:] # trim tau = 0, beta points
    Cov = np.cov(samples.T)
    Cov = Cov
    
    return Cov

def calculate_jackknife_cov(datafile):
    
    samples = np.loadtxt(datafile, delimiter=",")
    N = samples.shape[0]
    samples = samples[:,1:] # trim tau = 0, beta points
    sample_mean = np.mean(samples, axis=0)
    samples = (N*sample_mean - samples)/(N-1)
    Cov = np.cov(samples.T)
    
    return Cov

def calculate_std(datafile, of_mean = True):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,1:] # trim tau = 0, beta points
    std = np.std(samples, axis=0, ddof = 1)
    if of_mean:
        std = std / np.sqrt(samples.shape[0])
    
    return std

def calculate_cov_basis_map(Cov, rtol = 1e-4):
    
    U, s, Vh = np.linalg.svd(Cov, compute_uv=True, full_matrices=False, hermitian=True)
    e = np.sqrt(s)
    emax = e.max()
    mask = (e >= rtol*emax)
    e = e[mask]
    Vh = Vh[mask,:]
    U = U[:,mask]
    inv_sqrt_Cov = U @ np.diag(1/e) @ Vh
    inv_sqrt_Cov = inv_sqrt_Cov / np.sqrt(len(e))
    return inv_sqrt_Cov
    