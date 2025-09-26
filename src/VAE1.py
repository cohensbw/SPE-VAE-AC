import torch        
import torch.nn.functional as F
from torch import nn
from utilities.PoleToGreens import PoleToGaussLegendreGreens
from utilities.layer_stacks import make_conv_stack

class VAE1(nn.Module):
    
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
        
        # initialize convolutational statck
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
                
        self.encoder_linear_1 = nn.Linear(encoder_channels[-1]*L, encoder_channels[-1]*latent_dim, bias = False)
        self.encoder_linear_2 = nn.Linear(encoder_channels[-1]*latent_dim, latent_dim, bias = False)
        self.encoder_linear_to_mu = nn.Linear(latent_dim, latent_dim, bias = False)
        self.encoder_linear_to_logvar = nn.Linear(latent_dim, latent_dim, bias = False)
        
        # DECODER LAYERS
        
        self.decoder_linear_1 = nn.Linear(latent_dim, 2*latent_dim, bias = False)
        self.decoder_linear_2 = nn.Linear(2*latent_dim, 4*latent_dim, bias = False)
        self.decoder_linear_to_poles = nn.Linear(4*latent_dim, 2*self.num_poles, bias = True)
        self.decoder_linear_to_residues = nn.Linear(4*latent_dim, 2*self.num_poles, bias = False)
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
        x = F.leaky_relu(self.encoder_linear_1(x))
        x = F.leaky_relu(self.encoder_linear_2(x))
        mu = self.encoder_linear_to_mu(x)
        logvar = self.encoder_linear_to_logvar(x)
        
        return mu, logvar
    
    def decode(self, z):
        
        x = F.leaky_relu(self.decoder_linear_1(z))
        x = F.leaky_relu(self.decoder_linear_2(x))
        
        epsilon, gamma = torch.chunk(self.decoder_linear_to_poles(x), 2, dim=-1)
        poles = torch.complex(epsilon, -torch.abs(gamma))
        
        a, b = torch.chunk(self.decoder_linear_to_residues(x), 2, dim=-1)
        a_sum = a.sum(dim=-1, keepdim=True)
        a = a / (a_sum + 1e-12)  
        residues = torch.complex(a, b)
        
        Gtau, norm = self.pole_to_greens(poles, residues)
            
        return Gtau, poles, residues
    
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