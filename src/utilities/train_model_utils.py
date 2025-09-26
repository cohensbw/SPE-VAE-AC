import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from .loss_functions import vae_loss_1, vae_loss_2


# run a single training epoch
def run_training_epoch(epoch, dataloader, optimizer, model, alpha, eta, InvCov, DEVICE):
    model.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    training_loss = 0.0
    
    for i, x in loop:
        batch_size = x.shape[0]
        Gtau_in = x.reshape(batch_size, -1).to(DEVICE)

        # forward pass
        Gtau_out, poles, residues, mu, logvar = model(Gtau_in)

        # calculate loss
        if InvCov is None:
            loss = vae_loss_1(Gtau_out, Gtau_in, mu, logvar, alpha, eta)
        else:
            loss = vae_loss_2(Gtau_out, Gtau_in, InvCov, mu, logvar, alpha, eta)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        loop.set_description(f'Epoch [{epoch + 1}]')
        loop.set_postfix(loss=loss.item() / batch_size)
        
    training_loss /= len(dataloader.dataset)
    return training_loss


# validation epoch
def run_validation_epoch(epoch, dataloader, model, alpha, eta, InvCov, DEVICE):
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for x in dataloader:
            batch_size = x.shape[0]
            Gtau_in = x.reshape(batch_size, -1).to(DEVICE)
            Gtau_out, poles, residues, mu, logvar = model(Gtau_in)
            if InvCov is None:
                loss = vae_loss_1(Gtau_out, Gtau_in, mu, logvar, alpha, eta)
            else:
                loss = vae_loss_2(Gtau_out, Gtau_in, InvCov, mu, logvar, alpha, eta)
            validation_loss += loss.item()
    validation_loss /= len(dataloader.dataset)
    return validation_loss


# training loop with scheduler
def run_epochs(
    num_epochs, 
    training_dataloader, 
    validation_dataloader, 
    optimizer, 
    model, 
    alpha,
    eta,
    InvCov = None,
    DEVICE="cpu",
    scheduler = None
):
    training_losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        
        print(f'Epoch: {epoch + 1}')
        
        training_losses[epoch] = run_training_epoch(epoch, training_dataloader, optimizer, model, alpha, eta, InvCov, DEVICE)
        validation_losses[epoch] = run_validation_epoch(epoch, validation_dataloader, model, alpha, eta, InvCov, DEVICE)
        
        if scheduler is not None:
            scheduler.step()
        
        # Report
        lr = optimizer.param_groups[0]['lr'] 
        print(f'Learning Rate: {lr:.8f}')
        print(f'Train Loss: {training_losses[epoch]:.3e}')
        print(f'Validation Loss: {validation_losses[epoch]:.3e}', end='\n\n')
        
    return training_losses, validation_losses

def calculate_inv_cov(datafile, rtol = 1e-6, dtype = torch.float32):
    
    samples = np.loadtxt(datafile, delimiter=",")
    samples = samples[:,1:-1]
    Cov = np.cov(samples.T)
    eigvals, eigvecs = np.linalg.eigh(Cov)
    if rtol > 0:
        mask = eigvals >= (rtol * eigvals.max())
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
    eigvals = 1/eigvals
    # eigvals = eigvals / np.max(eigvals)
    eigvals = eigvals / np.exp(np.mean(np.log(eigvals)))
    InvCov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return torch.tensor(InvCov, dtype = dtype)