import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from .loss_functions import vae_loss


# run a single training epoch
def run_training_epoch(
    DEVICE,
    epoch,
    num_epochs,
    dataloader,
    optimizer,
    model,
    alpha,
    eta0,
    eta2,
    eta4,
    **kwargs
):
    
    model.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    
    training_total_loss = 0.0
    training_mse_loss = 0.0
    training_kl_loss = 0.0
    training_negativity_loss_0 = 0.0
    training_negativity_loss_2 = 0.0
    training_negativity_loss_4 = 0.0
    
    for i, x in loop:
        batch_size = x.shape[0]
        Gtau_in = x.reshape(batch_size, -1).to(DEVICE)

        # forward pass
        Gtau_out, poles, residues, mu, logvar = model(Gtau_in)
        
        # annealing parameter
        r = epoch/num_epochs
        
        # calculate loss:
        (total_loss, mse_loss, kl_loss,
            negativity_loss_0, negativity_loss_2, negativity_loss_4
        ) = vae_loss(
            poles, residues,
            Gtau_out, Gtau_in,
            mu, logvar,
            alpha, eta0, eta2, eta4,
            **kwargs
        )
            

        # back propagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        training_total_loss += total_loss.item()
        training_mse_loss += mse_loss.item()
        training_kl_loss += kl_loss.item()
        training_negativity_loss_0 += negativity_loss_0.item()
        training_negativity_loss_2 += negativity_loss_2.item()
        training_negativity_loss_4 += negativity_loss_4.item()
        loop.set_description(f'Epoch [{epoch + 1}]')
        loop.set_postfix(loss=total_loss.item() / batch_size)
        
    training_total_loss /= len(dataloader)
    training_mse_loss /= len(dataloader)
    training_kl_loss /= len(dataloader)
    training_negativity_loss_0 /= len(dataloader)
    training_negativity_loss_2 /= len(dataloader)
    training_negativity_loss_4 /= len(dataloader)
    
    return (
        training_total_loss, training_mse_loss, training_kl_loss,
        training_negativity_loss_0, training_negativity_loss_2, training_negativity_loss_4
    )


# validation epoch
def run_validation_epoch(
    DEVICE,
    epoch,
    num_epochs,
    dataloader,
    model,
    alpha,
    eta0,
    eta2,
    eta4,
    **kwargs
):
    
    model.eval()
    validation_total_loss = 0.0
    validation_mse_loss = 0.0
    validation_kl_loss = 0.0
    validation_negativity_loss_0 = 0.0
    validation_negativity_loss_2 = 0.0
    validation_negativity_loss_4 = 0.0
    
    r = epoch/num_epochs
    with torch.no_grad():
        for x in dataloader:
            
            batch_size = x.shape[0]
            Gtau_in = x.reshape(batch_size, -1).to(DEVICE)
            Gtau_out, poles, residues, mu, logvar = model(Gtau_in)
            
            # calculate loss
            (total_loss, mse_loss, kl_loss,
                negativity_loss_0, negativity_loss_2, negativity_loss_4
            ) = vae_loss(
                poles, residues,
                Gtau_out, Gtau_in,
                mu, logvar,
                alpha, eta0, eta2, eta4,
                **kwargs
            )
            
            validation_total_loss += total_loss.item()
            validation_mse_loss += mse_loss.item()
            validation_kl_loss += kl_loss.item()
            validation_negativity_loss_0 += negativity_loss_0.item()
            validation_negativity_loss_2 += negativity_loss_2.item()
            validation_negativity_loss_4 += negativity_loss_4.item()
            
    validation_total_loss /= len(dataloader)
    validation_mse_loss /= len(dataloader)
    validation_kl_loss /= len(dataloader)
    validation_negativity_loss_0 /= len(dataloader)
    validation_negativity_loss_2 /= len(dataloader)
    validation_negativity_loss_4 /= len(dataloader)
    
    return (
        validation_total_loss, validation_mse_loss, validation_kl_loss,
        validation_negativity_loss_0, validation_negativity_loss_2, validation_negativity_loss_4
    )


# training loop with scheduler
def run_epochs(
    DEVICE,
    scheduler,
    num_epochs, 
    training_dataloader, 
    validation_dataloader, 
    optimizer, 
    model, 
    alpha,
    eta0,
    eta2,
    eta4,
    **kwargs
):
    training_total_losses = np.zeros(num_epochs)
    training_mse_losses = np.zeros(num_epochs)
    training_kl_losses = np.zeros(num_epochs)
    training_negativity_loss_0 = np.zeros(num_epochs)
    training_negativity_loss_2 = np.zeros(num_epochs)
    training_negativity_loss_4 = np.zeros(num_epochs)
    
    validation_total_losses = np.zeros(num_epochs)
    validation_mse_losses = np.zeros(num_epochs)
    validation_kl_losses = np.zeros(num_epochs)
    validation_negativity_loss_0 = np.zeros(num_epochs)
    validation_negativity_loss_2 = np.zeros(num_epochs)
    validation_negativity_loss_4 = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        
        print(f'Epoch: {epoch + 1}')
        
        (training_total_losses[epoch],
         training_mse_losses[epoch],
         training_kl_losses[epoch],
         training_negativity_loss_0[epoch],
         training_negativity_loss_2[epoch],
         training_negativity_loss_4[epoch]
        ) = run_training_epoch(
            DEVICE,
            epoch,
            num_epochs,
            training_dataloader,
            optimizer,
            model,
            alpha,
            eta0,
            eta2,
            eta4,
            **kwargs
        )
        
        (validation_total_losses[epoch],
         validation_mse_losses[epoch],
         validation_kl_losses[epoch],
         validation_negativity_loss_0[epoch],
         validation_negativity_loss_2[epoch],
         validation_negativity_loss_4[epoch]
        ) = run_validation_epoch(
            DEVICE,
            epoch,
            num_epochs,
            validation_dataloader,
            model,
            alpha,
            eta0,
            eta2,
            eta4,
            **kwargs
        )
        
        if scheduler is not None:
            scheduler.step()
        
        # Report
        lr = optimizer.param_groups[0]['lr']
        
        print(f'Learning Rate: {lr:.8f}', end='\n\n')
        
        print(f'Training Total Loss: {training_total_losses[epoch]:.3e}')
        print(f'Training MSE Loss: {training_mse_losses[epoch]:.3e}')
        print(f'Training KL Loss: {training_kl_losses[epoch]:.3e}')
        print(f'Training Negativity Loss 0: {training_negativity_loss_0[epoch]:.3e}')
        print(f'Training Negativity Loss 2: {training_negativity_loss_2[epoch]:.3e}')
        print(f'Training Negativity Loss 4: {training_negativity_loss_4[epoch]:.3e}', end='\n\n')
        
        print(f'Validation Total Loss: {validation_total_losses[epoch]:.3e}')
        print(f'Validation MSE Loss: {validation_mse_losses[epoch]:.3e}')
        print(f'Validation KL Loss: {validation_kl_losses[epoch]:.3e}')
        print(f'Validation Negativity Loss 0: {validation_negativity_loss_0[epoch]:.3e}')
        print(f'Validation Negativity Loss 2: {validation_negativity_loss_2[epoch]:.3e}')
        print(f'Validation Negativity Loss 4: {validation_negativity_loss_4[epoch]:.3e}', end='\n\n')
        
    return (
        training_total_losses, training_mse_losses, training_kl_losses,
        training_negativity_loss_0, training_negativity_loss_2, training_negativity_loss_4,
        validation_total_losses, validation_mse_losses, validation_kl_losses,
        validation_negativity_loss_0, validation_negativity_loss_2, validation_negativity_loss_4
    )