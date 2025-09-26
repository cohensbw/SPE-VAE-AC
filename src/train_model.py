# # %%

# %load_ext autoreload
# %autoreload 2
# %matplotlib widget

# %%

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import integrate
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim

# %%

from utilities.Datasets import JackknifeGreensDataset, BootstrapGreensDataset, SimpleGreensDataset
from utilities.train_model_utils import run_epochs, calculate_inv_cov
from utilities.generate_predictions import generate_predictions
from VAE1 import VAE1
from VAE2 import VAE2
from VAE3 import VAE3

# %%

def spe_spectral(w, poles, residues):
    
    w = np.expand_dims(w, axis=0)
    poles = np.expand_dims(poles, axis=1)
    residues = np.expand_dims(residues, axis=1)
    return np.sum(-np.imag(residues/(w-poles))/np.pi, axis=0)

# %%

# specify dataset

datafile = "./../datasets/half-filled-gaussian/binned_data.csv"
beta = 10.0
dtau = 0.05
Ltau = int(beta / dtau) + 1
A = lambda w: stats.norm.pdf(w, loc = 0.0, scale = 1.0)

# datafile = "./../datasets/two-asymmetric-gaussians/binned_data.csv"
# beta = 10.0
# dtau = 0.05
# Ltau = int(beta / dtau) + 1
# A = lambda w: 0.6*stats.norm.pdf(w, loc = 2.0, scale = 1.0) + 0.4*stats.norm.pdf(w, loc = -2.0, scale = 0.5)

# %%

# load raw datset
dataset = SimpleGreensDataset(datafile)

# load jackknife dataset
jackknife_dataset = JackknifeGreensDataset(datafile)

# load bootstrap dataset
bootstrap_dataset = BootstrapGreensDataset(datafile)

# %%

# calculate inverse covariance matrix
# InvCov = calculate_inv_cov(datafile, rtol = 1e-2, dtype = torch.float32)
InvCov = None

# %%

# set the batch size
batch_size = 50

simple_dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

jackknife_dataloader = DataLoader(dataset = jackknife_dataset, batch_size = batch_size, shuffle = True)

bootstrap_dataloader = DataLoader(dataset = bootstrap_dataset, batch_size = batch_size, shuffle = True)

# %%

# initialize model
model = VAE2(
    beta = beta,
    dtau = dtau,
    num_poles = 4,
    latent_dim = 4*4,
    encoder_channels = [16, 32, 64],
    encoder_kernel_sizes = [11, 9, 7],
    encoder_strides = [2, 2, 2],
    encoder_dilations = [1, 1, 1],
    encoder_paddings = [5, 4, 3],
    encoder_padding_mode = "reflect",
    quadrature_nodes = 256
)

# model = VAE3(
#     beta = beta,
#     dtau = dtau,
#     num_poles = 16, 
#     latent_dim = 16,
#     encoder_channels = [8, 16, 32],
#     encoder_kernel_sizes = [11, 9, 7],
#     encoder_strides = [2, 2, 2],
#     encoder_dilations = [1, 1, 1],
#     encoder_paddings = [5, 4, 3],
#     encoder_padding_mode = "reflect",
#     encoder_dense_outputs = [],
#     decoder_dense_outputs = [],
#     decoder_kernel_sizes = [3],
#     decoder_padding_mode = "zeros",
#     quadrature_nodes = 256
# )

# %%
# configure training procedure
init_learning_rate = 3.0e-3
final_learning_rate = 3.0e-4
num_epochs = 100
weight_decay = 0.0

# initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=init_learning_rate, weight_decay=weight_decay)

# initialize scheduler
if init_learning_rate == final_learning_rate:
    scheduler = None
else:
    gamma = (final_learning_rate/init_learning_rate)**(1/num_epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


# %%
# train the model
training_losses, validation_losses = run_epochs(
    num_epochs = num_epochs,
    training_dataloader = jackknife_dataloader,
    validation_dataloader = bootstrap_dataloader,
    optimizer = optimizer,
    model = model,
    alpha = 0.0,
    eta = 1.0,
    InvCov = InvCov,
    DEVICE = "cpu",
    scheduler = scheduler
)

# %%

plt.figure()

plt.plot(training_losses, c = "b")
plt.plot(validation_losses, c = "r")

plt.tight_layout()
plt.show()


# %%

plt.figure()

plt.plot(np.log(training_losses), c = "b")
plt.plot(np.log(validation_losses), c = "r")

plt.tight_layout()
plt.show()

# %%

# generate predictions
predictions = generate_predictions(model, jackknife_dataset, as_mode = False)
poles = predictions[1]
residues = predictions[2]

# %%

# calculate prediction given mean Green's function
model.eval()
with torch.no_grad():
    Gtau_mean = torch.mean(jackknife_dataset[:,:], axis = 0)
    mean_out = model(torch.unsqueeze(Gtau_mean, dim=0))
    Gtau_mean_out = mean_out[0][0].numpy()
    poles_mean = mean_out[1][0].numpy()
    residues_mean = mean_out[2][0].numpy()

# %%

N_samples = poles.shape[0]
N_expansion = poles.shape[1]
N_omega = 1000
omega_min = -7.0
omega_max = 7.0

omega = np.linspace(omega_min, omega_max, N_omega)
A_omega = np.zeros((N_samples, len(omega)))

# iterate over samples
for s in range(N_samples):
    A_omega[s] = spe_spectral(omega, poles[s], residues[s])
    
# get mean prediction
A_omega_mean = spe_spectral(omega, poles_mean, residues_mean)

# %%

plt.figure()

for i in range(len(poles)):
    plt.plot(omega, spe_spectral(omega, poles[i], residues[i]), color = "black", lw = 0.5, alpha = 0.05)
    
plt.plot(omega, A_omega_mean, color="red")
plt.plot(omega, A(omega), color="magenta")
plt.ylim(-0.025, 0.500)
plt.tight_layout()
plt.show()

# %%

# calculate MSE for spectral function
Aout = lambda w: spe_spectral(w, poles_mean, residues_mean)
MSE_A = lambda w: np.square(Aout(w) - A(w))
spectral_error, tol = integrate.quad(MSE_A, -np.inf, np.inf, epsabs=1e-10)
print("|A_out(w) - A(w)|^2 = ", spectral_error)
