import torch
import torch.nn.functional as F

# def vae_loss(
#     poles, residues,
#     Gtau_out, Gtau_in,
#     mu, logvar,
#     alpha, eta0, eta2, eta4,
#     inv_var_0, inv_var_2, inv_var_4
# ):
    
#     # get dimension of array containing Green's functions
#     batch_size, Ltau = Gtau_out.shape
    
#     # calculate MSE loss
#     dG = Gtau_out - Gtau_in
#     mse_loss = torch.mean(dG**2 * inv_var_0)
    
#     # calculate KD divergence
#     logvar_clamped = torch.clamp(logvar, min=-50, max=50)
#     kl_divergence = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()) / batch_size
#     kl_loss = alpha * kl_divergence
    
#     # penalize G(tau) going negative
#     negativity_loss_0 = eta0 * torch.mean( F.relu(-Gtau_out)**2 * inv_var_0)
    
#     # penalize G''(tau) going negative
#     dG2 = Gtau_out[:, 2:] - 2 * Gtau_out[:, 1:-1] + Gtau_out[:, :-2]
#     negativity_loss_2 = eta2 * torch.mean(F.relu(-dG2)**2 * inv_var_2)
    
#     # penalize G''''(tau) going negative
#     dG4 = Gtau_out[:, 4:] - 4 * Gtau_out[:, 3:-1] + 6 * Gtau_out[:, 2:-2] - 4 * Gtau_out[:, 1:-3] + Gtau_out[:, :-4]
#     negativity_loss_4 = eta4 * torch.mean(F.relu(-dG4)**2 * inv_var_4)
    
#     # calculate total loss
#     total_loss = mse_loss + kl_loss + negativity_loss_0 + negativity_loss_2 + negativity_loss_4
    
#     return (total_loss, mse_loss, kl_loss, negativity_loss_0, negativity_loss_2, negativity_loss_4)

def vae_loss(
    poles, residues,
    Gtau_out, Gtau_in,
    mu, logvar,
    alpha, eta0, eta2, eta4,
    inv_sqrt_C, var0, var2, var4
):
    
    batch_size, Ltau = Gtau_out.shape

    # calculate MSE loss
    dG = Gtau_out - Gtau_in
    dG_white = dG @ inv_sqrt_C
    mse_loss = torch.mean(torch.sum(dG_white**2, dim=1) / Ltau)
    
    # KL divergence loss
    logvar_clamped = torch.clamp(logvar, min=-50, max=50)
    kl_divergence = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()) / batch_size
    kl_loss = alpha * kl_divergence

    # G(tau) negativity loss
    neg0 = F.relu(-Gtau_out)
    negativity_loss_0 = eta0 * torch.mean(torch.sum(neg0**2/var0, dim=1) / Ltau)
    # negativity_loss_0 = eta0 * torch.mean(torch.sum(neg0**2, dim=1)) / var0.mean() / Ltau

    # G''(tau) negativity loss
    dG2 = (Gtau_out[:, 2:] - 2 * Gtau_out[:, 1:-1] + Gtau_out[:, :-2])
    neg2 = F.relu(-dG2)
    negativity_loss_2 = eta2 * torch.mean(torch.sum(neg2**2/var2, dim=1) / (Ltau - 2))
    # negativity_loss_2 = eta2 * torch.mean(torch.sum(neg2**2, dim=1)) / var2.mean() / (Ltau - 2)

    # G''''(tau) negativity loss
    dG4 = (Gtau_out[:, 4:] - 4 * Gtau_out[:, 3:-1] + 6 * Gtau_out[:, 2:-2] - 4 * Gtau_out[:, 1:-3] + Gtau_out[:, :-4])
    neg4 = F.relu(-dG4)
    negativity_loss_4 = eta4 * torch.mean(torch.sum(neg4**2/var4, dim=1) / (Ltau - 4))
    # negativity_loss_4 = eta4 * torch.mean(torch.sum(neg4**2, dim=1)) / var4.mean() / (Ltau - 4)

    # total loss
    total_loss = (mse_loss + kl_loss + negativity_loss_0 + negativity_loss_2 + negativity_loss_4)

    return (total_loss, mse_loss, kl_loss, negativity_loss_0, negativity_loss_2, negativity_loss_4)
