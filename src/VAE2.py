import torch        
import torch.nn.functional as F
from torch import nn
from utilities.PoleToGreens import PoleToGaussLegendreGreens
from utilities.layer_stacks import make_conv_stack

class VAE2(nn.Module):
    
    def __init__(
        self,
        beta,
        dtau,
        num_poles,
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
                
        self.encoder_linear_1 = nn.Linear(L*encoder_channels[-1], encoder_channels[-1], bias = False)
        self.encoder_linear_2 = nn.Linear(encoder_channels[-1], 8*num_poles, bias = False)
        
        self.encoder_linear_a_1 = nn.Linear(8*num_poles, num_poles, bias = False)
        self.encoder_linear_a_2 = nn.Linear(8*num_poles, num_poles, bias = False)
        self.encoder_linear_to_mu_a = nn.Linear(num_poles, num_poles, bias = False)
        self.encoder_linear_to_logvar_a = nn.Linear(num_poles, num_poles, bias = False)
        
        self.encoder_linear_b_1 = nn.Linear(8*num_poles, num_poles, bias = False)
        self.encoder_linear_b_2 = nn.Linear(8*num_poles, num_poles, bias = False)
        self.encoder_linear_to_mu_b = nn.Linear(num_poles, num_poles, bias = False)
        self.encoder_linear_to_logvar_b = nn.Linear(num_poles, num_poles, bias = False)
        
        self.encoder_linear_epsilon_1 = nn.Linear(8*num_poles, num_poles, bias = True)
        self.encoder_linear_epsilon_2 = nn.Linear(8*num_poles, num_poles, bias = True)
        self.encoder_linear_to_mu_epsilon = nn.Linear(num_poles, num_poles, bias = True)
        self.encoder_linear_to_logvar_epsilon = nn.Linear(num_poles, num_poles, bias = True)
        
        self.encoder_linear_gamma_1 = nn.Linear(8*num_poles, num_poles, bias = True)
        self.encoder_linear_gamma_2 = nn.Linear(8*num_poles, num_poles, bias = True)
        self.encoder_linear_to_mu_gamma = nn.Linear(num_poles, num_poles, bias = True)
        self.encoder_linear_to_logvar_gamma = nn.Linear(num_poles, num_poles, bias = True)
        
        # DECODER LAYERS
        
        self.pole_to_greens = PoleToGaussLegendreGreens(
            beta = beta, dtau = self.dtau, N_nodes = quadrature_nodes, N_iwn = matsubara_max, dtype = dtype
        )
        
    # sample poles and residues
    def reparameterize(self,
        mu_a,
        logvar_a,
        mu_b,
        logvar_b,
        mu_epsilon,
        logvar_epsilon,
        mu_gamma,
        logvar_gamma
    ):
        
        logvar_clamped = torch.clamp(logvar_a, min=-50, max=50)
        std = torch.exp(0.5 * logvar_clamped)
        r = torch.randn_like(std)
        a = mu_a + std * r
        a = torch.abs(a)
        a_sum = a.sum(dim=-1, keepdim=True)
        a = a / (a_sum + 1e-12)
        
        logvar_clamped = torch.clamp(logvar_b, min=-50, max=50)
        std = torch.exp(0.5 * logvar_clamped)
        r = torch.randn_like(std)
        b = mu_b + std * r
        b_mean = b.mean(dim=-1, keepdim=True)
        b = b - b_mean
        
        logvar_clamped = torch.clamp(logvar_epsilon, min=-50, max=50)
        std = torch.exp(0.5 * logvar_clamped)
        r = torch.randn_like(std)
        epsilon = mu_epsilon + std * r
        
        logvar_clamped = torch.clamp(logvar_gamma, min=-50, max=50)
        std = torch.exp(0.5 * logvar_clamped)
        r = torch.randn_like(std)
        gamma = mu_gamma + std * r
        gamma = torch.abs(gamma)
        
        poles = torch.complex(epsilon, -gamma)
        residues = torch.complex(a, b)
        
        return poles, residues
    
    def encode(self, Gtau_in):
        
        x = Gtau_in.unsqueeze(1)
        x = self.encoder_conv_stack(x)
        x = torch.flatten(x, start_dim = 1)
        
        x = F.tanh(self.encoder_linear_1(x))
        x = F.tanh(self.encoder_linear_2(x))
        
        x_1 = F.tanh(self.encoder_linear_a_1(x))
        x_2 = F.tanh(self.encoder_linear_a_2(x))
        mu_a = self.encoder_linear_to_mu_a(x_1)
        logvar_a = self.encoder_linear_to_logvar_a(x_2)
        
        x_1 = F.tanh(self.encoder_linear_b_1(x))
        x_2 = F.tanh(self.encoder_linear_b_2(x))
        mu_b = self.encoder_linear_to_mu_b(x_1)
        logvar_b = self.encoder_linear_to_logvar_b(x_2)
        
        x_1 = F.tanh(self.encoder_linear_epsilon_1(x))
        x_2 = F.tanh(self.encoder_linear_epsilon_2(x))
        mu_epsilon = self.encoder_linear_to_mu_epsilon(x_1)
        logvar_epsilon = self.encoder_linear_to_logvar_epsilon(x_2)
        
        x_1 = F.tanh(self.encoder_linear_gamma_1(x))
        x_2 = F.tanh(self.encoder_linear_gamma_2(x))
        mu_gamma = self.encoder_linear_to_mu_gamma(x_1)
        logvar_gamma = self.encoder_linear_to_logvar_gamma(x_2)
        
        return mu_a, logvar_a, mu_b, logvar_b, mu_epsilon, logvar_epsilon, mu_gamma, logvar_gamma
    
    def decode(self, poles, residues):
        
        Gtau = self.pole_to_greens(poles, residues)
            
        return Gtau, poles, residues
    
    def forward(self, Gtau_in):
        
        mu_a, logvar_a, mu_b, logvar_b, mu_epsilon, logvar_epsilon, mu_gamma, logvar_gamma = self.encode(Gtau_in)
        poles, residues = self.reparameterize(mu_a, logvar_a, mu_b, logvar_b, mu_epsilon, logvar_epsilon, mu_gamma, logvar_gamma)
        Gtau_out, poles, residues = self.decode(poles, residues)
        mu = torch.cat([mu_a, mu_b, mu_epsilon, mu_gamma], dim=-1)
        logvar = torch.cat([logvar_a, logvar_b, logvar_epsilon, logvar_gamma], dim=-1)
        
        return Gtau_out, poles, residues, mu, logvar
    
    # given an input G(tau) curve, run the model assuming that mode of the latent space
    # distribution was sampled
    def mode_forward(self, Gtau_in):
        
        mu_a, logvar_a, mu_b, logvar_b, mu_epsilon, logvar_epsilon, mu_gamma, logvar_gamma = self.encode(Gtau_in)
        
        epsilon = mu_epsilon
        gamma = torch.abs(mu_gamma)
        poles = torch.complex(epsilon, gamma)
        
        a = torch.abs(mu_a)
        a_sum = a.sum(dim=-1, keepdim=True)
        a = a / (a_sum + 1e-12)
        b_mean = mu_b.mean(dim=-1, keepdim=True)
        b = mu_b - b_mean
        residues = torch.complex(a, b)
        
        Gtau_out, poles, residues = self.decode(poles, residues)
        mu = torch.cat([mu_a, mu_b, mu_epsilon, mu_gamma], dim=-1)
        logvar = torch.cat([logvar_a, logvar_b, logvar_epsilon, logvar_gamma], dim=-1)
        
        return Gtau_out, poles, residues, mu, logvar