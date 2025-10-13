import torch        
import torch.nn.functional as F
from torch import nn
from utilities.PoleToGreens import PoleToGaussLegendreGreens
from utilities.layer_stacks import make_conv_stack

class VAE(nn.Module):
    
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
        quadrature_nodes,
        matsubara_max,
        dtype
    ):
        
        # call nn.Module __init__() function
        super().__init__()
        
        # record parameters
        self.beta = beta
        self.dtau = dtau
        self.Ltau = int(beta/dtau)
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
            activation = nn.Tanh(),
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
        
        self.decoder_linear_to_poles_real = nn.Linear(4*latent_dim, num_poles, bias = True)
        # nn.init.zeros_(self.decoder_linear_to_poles_real.bias)
        
        self.decoder_linear_to_poles_imag = nn.Linear(4*latent_dim, num_poles, bias = True)
        # nn.init.zeros_(self.decoder_linear_to_poles_real.bias)
        
        self.decoder_linear_to_residues_real = nn.Linear(4*latent_dim, num_poles, bias = False)
        
        self.decoder_linear_to_residues_imag = nn.Linear(4*latent_dim, num_poles, bias = False)
        
        self.pole_to_greens = PoleToGaussLegendreGreens(
            beta = beta, dtau = self.dtau, N_nodes = quadrature_nodes, N_iwn = matsubara_max, dtype = dtype
        )
        
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
        x = F.tanh(self.encoder_linear_1(x))
        x = F.tanh(self.encoder_linear_2(x))
        mu = self.encoder_linear_to_mu(x)
        logvar = self.encoder_linear_to_logvar(x)
        
        return mu, logvar
    
    def decode(self, z):
        
        x = F.tanh(self.decoder_linear_1(z))
        x = F.tanh(self.decoder_linear_2(x))
        
        epsilon = self.decoder_linear_to_poles_real(x)
        gamma = self.decoder_linear_to_poles_imag(x)
        gamma = torch.abs(gamma)
        poles = torch.complex(epsilon, -gamma)
        
        a = self.decoder_linear_to_residues_real(x)
        a = torch.abs(a)
        a_sum = a.sum(dim=-1, keepdim=True)
        a = a / (a_sum + 1e-12)
        b = self.decoder_linear_to_residues_imag(x)
        b_mean = b.mean(dim=-1, keepdim=True)
        b = b - b_mean
        residues = torch.complex(a, b)
        
        Gtau = self.pole_to_greens(poles, residues)
            
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