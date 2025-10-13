import torch
import torch.nn.functional as F

def vae_loss(poles, residues, Gtau_out, Gtau_in, mu, logvar, alpha, eta0, eta2, std):
    
    # get dimension of array containing Green's functions
    batch_size, Ltau = Gtau_out.shape
    
    # add batch dimension of vector of standard deviation for broadasting,
    # and add floor value to std for numerical regularization purposes
    std = torch.clamp(std.unsqueeze(0), min = std.max() / 1.0e3)

    # calculate MSE loss
    dG = Gtau_out - Gtau_in
    mse_loss = torch.sum((dG / std) ** 2) / (Ltau * batch_size)

    # calculate KD divergence
    logvar_clamped = torch.clamp(logvar, min=-50, max=50)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar_clamped.exp()) / batch_size

    # penalize the G(tau) curves going negative
    positivity_error = torch.sum(F.relu(-Gtau_out / std) ** 2) / (Ltau * batch_size)
    # positivity_error = torch.sum(F.relu(-Gtau_out)) / batch_size

    # penalize the second derivative of G(tau) curve going negative
    dG2 = torch.diff(Gtau_out, n=2, dim=-1)
    dG2_std = torch.sqrt(std[:, :-2] ** 2 + 4 * std[:, 1:-1] ** 2 + std[:, 2:] ** 2)
    sfd_positivity_error = torch.sum(F.relu(-dG2 / dG2_std) ** 2) / ((Ltau - 2) * batch_size)
    # sfd_positivity_error = torch.sum(F.relu(-dG2)) / batch_size

    return mse_loss + alpha * kl_divergence + eta0 * positivity_error + eta2 * sfd_positivity_error