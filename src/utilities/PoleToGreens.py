import torch
import torch.nn as nn
from scipy import special

class PoleToGaussLegendreGreens(nn.Module):

    def __init__(self, beta, Ltau, N_nodes = 256, dtype = torch.float32):

        # Call nn.Module default _init_() method
        super().__init__()
        
        # Tau grid
        taus = torch.linspace(0.0, beta, Ltau, dtype=dtype)
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype)) # (,)
        self.register_buffer("Ltau", torch.tensor(Ltau, dtype=torch.int))
        self.register_buffer("taus", taus)  # (Ltau,)
        
        # Gauss–Legendre nodes + weights
        nodes, weights = special.roots_legendre(N_nodes)
        nodes = torch.tensor(nodes, dtype=dtype)       # (N_nodes,)
        weights = torch.tensor(weights, dtype=dtype)   # (N_nodes,)
        self.register_buffer("nodes", nodes)
        self.register_buffer("weights", weights)

        # Precompute Phi(t) = tan(pi/2 * t)
        phi = torch.tan(torch.pi/2 * nodes)  # (N_nodes,)
        self.register_buffer("phi", phi)

    def forward(self, poles, residues):
            
         # Ensure inputs are at least 2D: (batch, num_poles)
        if poles.ndim == 1:
            poles = poles.unsqueeze(0)       # (1, num_poles)
            residues = residues.unsqueeze(0) # (1, num_poles)
            
        epsilon = torch.real(poles)
        gamma = -torch.imag(poles)
        a = torch.real(residues)
        b = torch.imag(residues)

        # omegas: (batch, num_poles, N_nodes)
        omegas = epsilon[..., None] + gamma[..., None] * self.phi[None, None, :]

        # numerator: (batch, num_poles, N_nodes)
        numerator = 0.5 * (a[..., None] - b[..., None] * self.phi[None, None, :])

        # taus: (Ltau,)
        taus = self.taus
        
        # calculate arguments of exponents
        arg1 = taus[None, None, :, None] * omegas[..., None, :]
        arg2 = (taus[None, None, :, None] - self.beta) * omegas[..., None, :]
        
        # clamp the arguments of the exponents
        arg1 = torch.clamp(arg1, min=-50.0, max=50.0)
        arg2 = torch.clamp(arg2, min=-50.0, max=50.0)

        # denominator: (batch, num_poles, Ltau, N_nodes)
        denominator = torch.exp(arg1) + torch.exp(arg2)

        # integrand: (batch, num_poles, Ltau, N_nodes)
        integrand = numerator[..., None, :] / denominator

        # perform gaussian quadrature outputting shape (batch, num_poles, Ltau)
        G_tau = torch.matmul(integrand, self.weights)
        
        # Sum for each Green's function
        G_tau = G_tau.sum(dim=1)
        
        # normalizations
        norms = torch.matmul(numerator,self.weights).sum(dim=1)

        return G_tau, norms
