import torch        
import torch.nn.functional as F
from torch import nn
from utilities.PoleToGreens import PoleToGaussLegendreGreens
from utilities.layer_stacks import make_conv_stack, make_linear_stack

class VAE3(nn.Module):
    
    def __init__(
        self,
        beta,
        dtau,
        num_poles, 
        latent_dim,
        encoder_channels,
        encoder_kernel_sizes,
        encoder_strides,
        encoder_dilations,
        encoder_paddings,
        encoder_padding_mode,
        encoder_dense_outputs,
        decoder_dense_outputs,
        decoder_kernel_sizes,
        decoder_padding_mode,
        quadrature_nodes
    ):
        
        # call nn.Module __init__() function
        super().__init__()
        
        # record parameters
        self.beta = beta
        self.dtau = dtau
        self.Ltau = int(beta/dtau) + 1
        self.latent_dim = latent_dim
        self.num_poles = num_poles
        
        # ENCODER LAYERS
        
        # initialize encoder convolutational stack
        self.encoder_conv_stack, L = make_conv_stack(
            L = self.Ltau,
            out_channels = encoder_channels,
            kernel_sizes = encoder_kernel_sizes,
            strides = encoder_strides,
            paddings = encoder_paddings,
            dilations = encoder_dilations,
            padding_mode = encoder_padding_mode,
            activation = nn.LeakyReLU(),
            in_channel = 1,
            bias = False
        )
        
        # number of inputs to dense layers that follow
        encoder_dense_inputs = L * encoder_channels[-1]
        
        # initialize encoder dense layers
        encoder_dense_outputs =  encoder_dense_outputs + [2*latent_dim]
        
        # initialize encoder dense stack
        self.encoder_dense_stack = make_linear_stack(
            input_dim = encoder_dense_inputs,
            output_dims = encoder_dense_outputs,
            activation = nn.LeakyReLU(),
            bias = False
        )
        
        # dense layer to return mean of latent space distribution
        self.encoder_dense_to_mu = nn.Linear(latent_dim, latent_dim, bias = False)
        
        # dense layer to return logvar of latent space distribution
        self.encoder_dense_to_logvar = nn.Linear(latent_dim, latent_dim, bias = False)
        
        # DECODER LAYERS
        
        # output for the number of poles
        decoder_dense_outputs = decoder_dense_outputs + [4*num_poles]
        
        # initialize decoder dense stack
        self.decoder_dense_stack = make_linear_stack(
            input_dim = latent_dim,
            output_dims = decoder_dense_outputs,
            activation = nn.LeakyReLU(),
            bias = False
        )
        
        # initialize encoder convolutational stack
        self.decoder_conv_stack, L = make_conv_stack(
            L = num_poles,
            out_channels = len(decoder_kernel_sizes)*[4],
            kernel_sizes = decoder_kernel_sizes,
            strides = len(decoder_kernel_sizes)*[1],
            paddings = [(k-1)//2 for k in decoder_kernel_sizes],
            dilations = len(decoder_kernel_sizes)*[1],
            padding_mode = decoder_padding_mode,
            activation = nn.LeakyReLU(),
            in_channel = 4,
            bias = False
        )
        
        # layer to calculate Green's function given poles and residues
        self.pole_to_greens = PoleToGaussLegendreGreens(beta = beta, Ltau = self.Ltau, N_nodes = quadrature_nodes)
        
    # z = mu + sigma * eps, with eps ~ N(0, I).
    # note a clamp is applied to logvar to avoid overflow issues.
    def reparameterize(self, mu, logvar):
        
        logvar_clamped = torch.clamp(logvar, min=-50, max=50)
        std = torch.exp(0.5 * logvar_clamped)
        r = torch.randn_like(std)
        return mu + std * r
    
    def encode(self, Gtau_in):
        
        x = Gtau_in.unsqueeze(1)
        x = self.encoder_conv_stack(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.encoder_dense_stack(x)
        to_mu, to_logvar = torch.chunk(x, 2, dim = -1)
        mu = self.encoder_dense_to_mu(to_mu)
        logvar = self.encoder_dense_to_mu(to_logvar)
        
        return mu, logvar
    
    def decode(self, z):
        
        x = self.decoder_dense_stack(z)
        x = x.view(x.size(0), 4, self.num_poles)
        x = self.decoder_conv_stack(x)
        channels = torch.unbind(x, dim=1)
        a = torch.square(channels[0])
        a_sum = a.sum(dim=-1, keepdim=True)
        a = a / (a_sum + 1e-12) 
        b = channels[1]
        epsilon = channels[2]
        Gamma = torch.square(channels[3])
        residues = torch.complex(a,b)
        poles = torch.complex(epsilon, -Gamma)
        Gtau_out, norm = self.pole_to_greens(poles, residues)
        
        return Gtau_out, poles, residues
    
    def forward(self, Gtau_in):
        
        mu, logvar = self.encode(Gtau_in)
        z = self.reparameterize(mu, logvar)
        Gtau_out, poles, residues = self.decode(z)
        
        return Gtau_out, poles, residues, mu, logvar
    
    # given an input G(tau) curve, run the model assuming that mode of the latent space
    # distribution was sampled
    def mode_forward(self, Gtau_in):
        
        mu, logvar = self.encode(Gtau_in)
        Gtau_out, poles, residues = self.decode(mu)
        
        return Gtau_out, poles, residues, mu, logvar