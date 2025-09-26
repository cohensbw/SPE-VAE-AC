import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

# get preditions for trained VAE mode.
# This assumes a mode_forward method is defined for the VAE model.
def generate_predictions(model, dataset, device="cpu", batch_size = 1, as_mode=True, predictions_per_sample=1):
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []

    model.eval()
    with torch.no_grad():
        
        if as_mode: # if getting most probable prediction for each sample
            for x in loader:
                x = x.to(device)
                y = model.mode_forward(x)
                if device == "cpu":
                    outputs.append(y)
                else:
                    outputs.append(tuple(i.cpu() for i in y))
        else: # if sampling random prediction for each sample
            for x in loader:
                x = x.to(device)
                for i in range(predictions_per_sample):
                    y = model(x)
                    if device == "cpu":
                        outputs.append(y)
                    else:
                        outputs.append(tuple(i.cpu() for i in y))
            
    n_outputs = len(outputs[1])
    all_outputs = []
    for i in range(n_outputs):
        all_outputs.append(torch.cat([outputs[n][i] for n in range(len(outputs))]).numpy())
        
    return all_outputs