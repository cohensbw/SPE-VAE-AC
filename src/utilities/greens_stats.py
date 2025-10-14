import numpy as np

def calculate_cov(datafile, of_mean = True):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,0:-1] # trim tau = beta point
    Cov = np.cov(samples.T)
    if of_mean:
        Cov = Cov / samples.shape[0]
    
    return Cov

def calculate_std(datafile, of_mean = True):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,0:-1] # trim tau = beta point
    std = np.std(samples, axis=0, ddof = 1)
    if of_mean:
        std = std / np.sqrt(samples.shape[0])
    
    return std

import numpy as np

def calculate_cov_basis_map(Cov, rtol=1e-8):
    
    # Eigen-decomposition (since Cov is Hermitian)
    eigvals, eigvecs = np.linalg.eigh(Cov)
    
    # Clip small negative values (from rounding) to zero
    eigvals = np.clip(eigvals, 0, None)
    
    # Keep only large enough modes
    e = np.sqrt(eigvals)
    emax = e.max()
    mask = e >= rtol * emax
    
    e = e[mask]
    U = eigvecs[:, mask]
    
    # Construct inverse sqrt(Cov)
    inv_sqrt_Cov = U @ np.diag(1 / e) @ U.T.conj()
    
    return inv_sqrt_Cov
    