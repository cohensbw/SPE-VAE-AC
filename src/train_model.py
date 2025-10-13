# %%

# %load_ext autoreload
# %autoreload 2
# %matplotlib widget

# %%


import torch

dtype = torch.float32
torch.set_default_dtype(dtype)

import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import integrate

# %%

from utilities.Datasets import JackknifeGreensDataset, BootstrapGreensDataset, SimpleGreensDataset
from utilities.train_model_utils import run_epochs
from utilities.generate_predictions import generate_predictions
from utilities.greens_stats import calculate_cov, calculate_std, calculate_cov_basis_map, calculate_jackknife_cov
from VAE import VAE

# %%

def spe_spectral(w, poles, residues):
    
    w = np.expand_dims(w, axis=0)
    poles = np.expand_dims(poles, axis=1)
    residues = np.expand_dims(residues, axis=1)
    return np.sum(np.imag(residues/(poles-w))/np.pi, axis=0)

# %%

# specify dataset

# datafile = "./../datasets/half-filled-gaussian/binned_data.csv"
# beta = 10.0
# dtau = 0.05
# Ltau = int(beta / dtau) + 1
# A = lambda w: stats.norm.pdf(w, loc = 0.0, scale = 1.0)

datafile = "./../datasets/two-asymmetric-gaussians/binned_data.csv"
beta = 10.0
dtau = 0.05
Ltau = int(beta / dtau) + 1
A = lambda w: 0.6*stats.norm.pdf(w, loc = 2.0, scale = 1.0) + 0.4*stats.norm.pdf(w, loc = -2.0, scale = 0.5)

# datafile = "./../datasets/doped_lorentzian/binned_data.csv"
# beta = 10.0
# dtau = 0.05
# Ltau = int(beta / dtau) + 1
# A = lambda w: stats.cauchy.pdf(w, loc = -1.0, scale = 0.5)

# %%

std = calculate_std(datafile, of_mean = True)
std = torch.tensor(std, dtype = dtype)

# %%

# load raw datset
dataset = SimpleGreensDataset(datafile, dtype = dtype)

# Split deterministically (optional: set generator for reproducibility)
training_dataset, testing_dataset = random_split(dataset, [0.8, 0.2])

# training_dataset = JackknifeGreensDataset(datafile, dtype = dtype)
# testing_dataset = BootstrapGreensDataset(datafile, dtype = dtype)

# %%

batch_size = 50
training_dataloader = DataLoader(dataset = training_dataset, batch_size = batch_size, shuffle = True)
testing_dataloader = DataLoader(dataset = testing_dataset, batch_size = batch_size, shuffle = True)

# %%

# initialize model
model = VAE(
    beta = beta,
    dtau = dtau,
    num_poles = 8,
    latent_dim = 4*8,
    encoder_channels = [16, 32, 64],
    encoder_kernel_sizes = [9, 9, 9],
    encoder_strides = [2, 2, 2],
    encoder_dilations = [1, 1, 1],
    encoder_paddings = [4, 4, 4],
    encoder_padding_mode = "reflect",
    quadrature_nodes = 256,
    matsubara_max = 512,
    dtype = dtype
)

# %%
# configure training procedure
init_learning_rate = 1.0e-3
final_learning_rate = 1.0e-4
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
    training_dataloader = training_dataloader,
    validation_dataloader = testing_dataloader,
    optimizer = optimizer,
    model = model,
    alpha = 0.0,
    eta0 = 1.0,
    eta2 = 1.0,
    std = std,
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

plt.plot(training_losses, c = "b")
plt.plot(validation_losses, c = "r")
plt.yscale("log")
plt.tight_layout()
plt.show()

# %%

# generate predictions
predictions = generate_predictions(model, training_dataset, as_mode = False)
poles = predictions[1]
residues = predictions[2]

# %%

# calculate prediction given mean Green's function
model.eval()
with torch.no_grad():
    data = torch.stack([training_dataset[i] for i in range(len(training_dataset))])
    Gtau_mean = data.mean(dim=0).unsqueeze(0)
    mean_out = model(Gtau_mean)
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
# plt.ylim(-0.025, 0.500)
plt.tight_layout()
plt.show()

# %%

# calculate MSE for spectral function
Aout = lambda w: spe_spectral(w, poles_mean, residues_mean)
MSE_A = lambda w: np.square(Aout(w) - A(w))
spectral_error, tol = integrate.quad(MSE_A, -np.inf, np.inf, epsabs=1e-10)
print("\int_\infty^\infty dw |A_out(w) - A(w)|^2 = ", spectral_error)

# %%
