import torch
import torch.nn.functional as F

def vae_loss_1(Gtau_out, Gtau_in, mu, logvar, alpha, eta):
    
    # get the batch size
    batch_size = Gtau_out.size(0)
    
    # length of imaginary-time axis
    Ltau = Gtau_out.size(1)
    
    # clipped data
    Gtau_in  = Gtau_in[:,1:(Ltau-1)]
    Gtau_out = Gtau_out[:,1:(Ltau-1)]
    
    # Reconstruction loss averaged per batch
    mse_loss = F.mse_loss(Gtau_out, Gtau_in, reduction="sum") / batch_size

    # KL divergence averaged per batch
    logvar_clamped = torch.clamp(logvar, min=-30, max=30)
    kl_divergence = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()) / batch_size
    
    # add cost function to promote positivity of spectral function
    dG2 = torch.diff(Gtau_out,n=2)
    positivity_errors = torch.sum(F.relu(-dG2)) / batch_size
    # dG4 = torch.diff(dG2,n=2)
    # positivity_errors += torch.sum(F.relu(-dG4)) / batch_size

    return mse_loss + alpha * kl_divergence + eta * positivity_errors

def vae_loss_2(Gtau_out, Gtau_in, InvCov, mu, logvar, alpha, eta):
    
    # get the batch size
    batch_size = Gtau_out.size(0)
    
    # length of imaginary-time axis
    Ltau = Gtau_out.size(1)
    
    # clipped data
    Gtau_in  = Gtau_in[:,1:(Ltau-1)]
    Gtau_out = Gtau_out[:,1:(Ltau-1)]
    
    # Calculate reconstruction loss weighted by covariance matrix
    dGtau = Gtau_out - Gtau_in
    intermediate = dGtau @ InvCov
    mse_loss = (intermediate * dGtau).sum(dim=1)
    mse_loss = mse_loss.mean()

    # KL divergence averaged per batch
    logvar_clamped = torch.clamp(logvar, min=-30, max=30)
    kl_divergence = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()) / batch_size
    
    # add cost function to promote positivity of spectral function
    dG2 = torch.diff(Gtau_out,n=2)
    positivity_errors = torch.sum(F.relu(-dG2)) / batch_size
    # dG4 = torch.diff(dG2,n=2)
    # positivity_errors += torch.sum(F.relu(-dG4)) / batch_size

    return mse_loss + alpha * kl_divergence + eta * positivity_errors