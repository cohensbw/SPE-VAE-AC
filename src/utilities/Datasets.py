# This script is for loading and processing the dataset
import numpy as np
import torch
from torch.utils.data import Dataset

# construct a dataset of the jackknife sample means of the read in samples
class SimpleGreensDataset(Dataset):
    
    def __init__(self, filepath, transform=None, dtype = torch.float32):
        
        # trim off tau = beta value
        samples = np.loadtxt(filepath, delimiter=",")[:,0:-1]
        self.data = torch.tensor(samples, dtype = dtype)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transormation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
            
        return x

# construct a dataset of the jackknife sample means of the read in samples
class JackknifeGreensDataset(Dataset):
    
    def __init__(self, filepath, transform=None, dtype = torch.float32):
        
        # Load all Green's functions at once, trimming off tau = beta value
        samples = np.loadtxt(filepath, delimiter=",")[:,0:-1]
        
        # get the number of samples
        N_samples = samples.shape[0]
        
        # get the average of all the samples i.e. the sample mean
        sample_mean = np.mean(samples, axis = 0)
        
        # calculate the jackknife sample means. This expression updates the sample
        # mean to reflect removing one sample, doing so for each input sample.
        jackknife_sample_means = (N_samples*sample_mean - samples)/(N_samples - 1)
        
        self.data = torch.tensor(jackknife_sample_means, dtype = dtype)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transormation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
            
        return x
    

# construct a dataset of the means of bootstrap samples generated based on the read in samples
class BootstrapGreensDataset(Dataset):
    
    def __init__(self, filepath, N_bootstrap=None, transform=None, dtype = torch.float32):
        
        # Load all the Green's function as once, trimming off tau = beta value
        samples = np.loadtxt(filepath, delimiter=",")[:,0:-1]
        
        # get the number of samples
        N_samples = samples.shape[0]
        
        # length of imaginary-time axis
        Ltau = samples.shape[1]
        
        # Set the number of bootstrap samples
        if N_bootstrap == None:
            N_bootstrap = N_samples
            
        # initialize array to contain bootstrap sample means
        bootstrap_sample_means = np.empty((N_bootstrap, Ltau))
        
        # iterate over bootstrap samples
        for i in range(N_bootstrap):
            
            # generate bootstrap sample
            indices = np.random.randint(0, N_samples, N_samples)
            bootstrap_sample = samples[indices]
            
            # calculate the mean of current bootstrap sample
            bootstrap_sample_means[i] = bootstrap_sample.mean(axis=0)
            
        self.data = torch.tensor(bootstrap_sample_means, dtype = dtype)
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        x = self.data[idx]
        
        # Perform any transormation on the data
        # Could be normalization, augmentation, or noise addition
        if self.transform:
            x = self.transform(x)
            
        return x