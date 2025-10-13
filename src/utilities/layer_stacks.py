import numpy as np
import torch
from torch import nn


# create a sequential stack of dense linear layers with activation in between
def make_linear_stack(
    input_dim,
    output_dims,
    activation,
    bias = False
):

    dims = [input_dim] + output_dims
    layers = []
    for i in range(1,len(dims)):
        linear_layer = nn.Linear(dims[i-1], dims[i], bias = bias)
        # nn.init.xavier_uniform_(linear_layer.weight)
        # if bias:
        #     nn.init.zeros_(linear_layer.bias)
        layers.append(linear_layer)
        layers.append(activation)
    
    return nn.Sequential(*layers)


# make sequential stack of convolutions, with instance norm and activation layers in between
def make_conv_stack(
    L,
    out_channels,
    kernel_sizes,
    strides,
    paddings,
    dilations,
    padding_mode,
    activation = nn.LeakyReLU(),
    in_channel = 1,
    bias = False,
):

    layers = []
    for out_channel, k, s, p, d in zip(out_channels, kernel_sizes, strides, paddings, dilations):
        conv_layer = nn.Conv1d(in_channel, out_channel, kernel_size=k, stride=s, padding=p, dilation=d, padding_mode=padding_mode, bias=bias)
        nn.init.xavier_uniform_(conv_layer.weight)
        if bias:
            nn.init.zeros_(conv_layer.bias)
        layers.append(conv_layer)
        layers.append(nn.InstanceNorm1d(out_channel))
        layers.append(activation)
        L = conv1d_output_length(L, k, stride=s, padding=p, dilation=d)
        in_channel = out_channel

    return nn.Sequential(*layers), L


# calculate length outputted by 1D convolutional layer
def conv1d_output_length(L_in, kernel_size, stride=1, padding=0, dilation=1):
    
    return int(np.floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))