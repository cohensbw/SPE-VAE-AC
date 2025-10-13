# %%

%load_ext autoreload
%autoreload 2
%matplotlib widget
    
# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import torch

# %%
from utilities.PoleToGreens import PoleToGaussLegendreGreens

# %%
def eval_Gz(z, a, b, epsilon, gamma):
    
    return (a + 1j*b)/((epsilon - 1j*gamma) - z)

def eval_Aw(w, a, b, epsilon, gamma):
    
    return np.imag(eval_Gz(w, a, b, epsilon, gamma))/np.pi

# %%
a1 = 0.6
b1 = 0.2
epsilon1 = 3.0
gamma1 = 0.6


a2 = 0.4
b2 = -0.2
epsilon2 = -2.0
gamma2 = 0.3

# %%
beta = 10.0
dtau = 0.05

# %%
pole_to_greens = PoleToGaussLegendreGreens(
    beta = beta,
    dtau = dtau,
    N_nodes = 256,
    N_iwn = 256,
    dtype = torch.float64
)

# %%

poles = torch.tensor([
    epsilon1 - 1j*gamma1,
    epsilon2 - 1j*gamma2
], dtype = torch.complex128)

residues = torch.tensor([
    a1 + 1j*b1,
    a2 + 1j*b2
], dtype = torch.complex128)


# %%
Gtau = pole_to_greens(poles, residues)[0]

# %%

f = lambda w: (
    eval_Aw(w, a1, b1, epsilon1, gamma1)
    + eval_Aw(w, a2, b2, epsilon2, gamma2)
    ) / (
        1 + np.exp(-beta*w)
    )

# %%

info = integrate.quad(f, -np.inf, np.inf, epsabs = 1e-12)
G0 = info[0]
info

# %%
float(Gtau[0]) - G0

# %%
