import numpy as np

def calculate_cov(datafile, reg = 1e-6):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,0:-1] # trim tau = beta point
    Cov = np.cov(samples.T, bias=False)
    TrCov = np.trace(Cov)
    Cov += reg * TrCov/ Cov.shape[0] * np.eye(Cov.shape[0])

    return Cov

def calculate_var(datafile, reg = 1e-6):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,0:-1] # trim tau = beta point
    var = np.var(samples, axis=0, ddof = 1)
    var += reg * np.mean(var)
    
    return var

def calculate_var_and_derivatives(datafile, eps=1e-6):
    """
    Calculate regularized variance vectors for VAE loss:
    - var: variance of G(tau)
    - var2: variance of second derivative D2 G
    - var4: variance of fourth derivative D4 G
    
    Args:
        datafile : str
            Path to CSV file containing samples (shape: N_samples x Ltau+1)
        eps : float
            Ridge regularization
            
    Returns:
        var, var2, var4 : np.ndarray
            Regularized variance vectors 
    """
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,0:-1] # trim tau = beta point
    
    var = np.var(samples, axis=0, ddof = 1)
    var2 = var[2:] + 4 * var[1:-1] + var[:-2]
    var4 = var[4:] + 16 * var[3:-1] + 36 * var[2:-2] + 16 * var[1:-3] + var[:-4]
    
    var += eps * np.mean(var)
    var2 += eps * np.mean(var2)
    var4 += eps * np.mean(var4)
    
    return var, var2, var4

# def calculate_cov_and_derivatives(datafile, variance_threshold=0.99):
    
#     # Load samples and remove last tau point
#     samples = np.loadtxt(datafile, delimiter=",")
#     samples = samples[:, :-1]  # remove tau=beta point
#     N_samples, Ltau = samples.shape

#     # --- Covariance of G(tau) ---
#     C = np.cov(samples.T, bias=False)
    
#     # Whitening transformation for C with variance threshold
#     w, V = np.linalg.eigh(C)
#     w = np.maximum(w, 0.0)
#     w_sorted = np.sort(w)[::-1]  # Sort in descending order
#     cumsum = np.cumsum(w_sorted)
#     total_var = cumsum[-1]
#     n_components = np.searchsorted(cumsum, variance_threshold * total_var) + 1
#     w_floor = w_sorted[n_components - 1]
#     w = np.maximum(w, w_floor)
#     W = V @ np.diag(1/np.sqrt(w)) * np.sqrt(n_components)
#     C = V @ np.diag(w) @ V.T  # Reconstruct regularized covariance
    
#     # Trace of covariance
#     TrC = np.trace(C)

#     # --- Second derivative operator ---
#     D2 = np.zeros((Ltau-2, Ltau))
#     for i in range(Ltau-2):
#         D2[i, i]   = 1.0
#         D2[i, i+1] = -2.0
#         D2[i, i+2] = 1.0

#     # Covariance of second derivative
#     C2 = D2 @ C @ D2.T
    
#     # Trace of second derivative covariance
#     TrC2 = np.trace(C2)

#     # --- Fourth derivative operator ---
#     D4 = np.zeros((Ltau-4, Ltau))
#     for i in range(Ltau-4):
#         D4[i, i]   = 1.0
#         D4[i, i+1] = -4.0
#         D4[i, i+2] = 6.0
#         D4[i, i+3] = -4.0
#         D4[i, i+4] = 1.0

#     # Covariance of fourth derivative
#     C4 = D4 @ C @ D4.T
    
#     # Trace of fourth derivative covariance
#     TrC4 = np.trace(C4)
    
#     return W, TrC, TrC2, TrC4

def calculate_cov_and_derivatives(datafile, variance_threshold=0.99):
    
    # Load samples and remove last tau point
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:, :-1]  # remove tau=beta point
    N_samples, Ltau = samples.shape

    # --- Covariance of G(tau) ---
    C = np.cov(samples.T, bias=False)
    
    # Whitening transformation for C with variance threshold
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 0.0)
    idx = np.argsort(w)[::-1]  # Sort in descending order
    w_sorted = w[idx]
    V_sorted = V[:, idx]
    cumsum = np.cumsum(w_sorted)
    n_components = np.searchsorted(cumsum, variance_threshold * cumsum[-1]) + 1
    w_trunc = w_sorted[:n_components]
    V_trunc = V_sorted[:, :n_components]
    W = V_trunc @ np.diag(1/np.sqrt(w_trunc)) * np.sqrt(n_components)
    C = V_trunc @ np.diag(w_trunc) @ V_trunc.T  # Reconstruct regularized covariance
    
    # variances
    var0 = np.diag(C)

    # # --- Second derivative operator ---
    # D2 = np.zeros((Ltau-2, Ltau))
    # for i in range(Ltau-2):
    #     D2[i, i]   = 1.0
    #     D2[i, i+1] = -2.0
    #     D2[i, i+2] = 1.0

    # # Covariance of second derivative
    # C2 = D2 @ C @ D2.T
    
    # # variances
    # var2 = np.diag(C2)
    
    var2 = var0[2:] + 4 * var0[1:-1] + var0[:-2]

    # # --- Fourth derivative operator ---
    # D4 = np.zeros((Ltau-4, Ltau))
    # for i in range(Ltau-4):
    #     D4[i, i]   = 1.0
    #     D4[i, i+1] = -4.0
    #     D4[i, i+2] = 6.0
    #     D4[i, i+3] = -4.0
    #     D4[i, i+4] = 1.0

    # # Covariance of fourth derivative
    # C4 = D4 @ C @ D4.T
    
    # # variances
    # var4 = np.diag(C4)
    
    var4 = var0[4:] + 16 * var0[3:-1] + 36 * var0[2:-2] + 16 * var0[1:-3] + var0[:-4]
    
    return W, var0, var2, var4