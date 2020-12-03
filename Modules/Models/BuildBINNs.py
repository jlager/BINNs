import torch, pdb
import torch.nn as nn

from Modules.Models.BuildMLP import BuildMLP
from Modules.Activations.SoftplusReLU import SoftplusReLU
from Modules.Utils.Gradient import Gradient

class u_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the solution of the governing PDE. 
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted cell densities non-negative.
    
    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity
    
    Args:
        inputs (torch tensor): x and t pairs with shape (N, 2)
        
    Returns:
        outputs (torch tensor): predicted u values with shape (N, 1)
    '''
    
    def __init__(self, scale=1.7e3):
        
        super().__init__()
        self.scale = scale
        self.mlp = BuildMLP(
            input_features=2, 
            layers=[128, 128, 128, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
    
    def forward(self, inputs):
        
        outputs = self.scale * self.mlp(inputs)
        
        return outputs

class D_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown diffusivity function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted diffusivities non-negative.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        D (torch tensor): predicted diffusivities with shape (N, 1)
    '''
    
    
    def __init__(self, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = 0 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.max = 4000 / (1000**2) / (1/24) # um^2/hr -> mm^2/d
        self.mlp = BuildMLP(
            input_features=input_features, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
        
    def forward(self, u, t=None):
        
        if t is None:
            D = self.mlp(u/self.scale)
        else:
            D = self.mlp(torch.cat([u/self.scale, t], dim=1))    
        D = self.max * D
        
        return D

class G_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown growth function. Includes 
    three hidden layers with 32 sigmoid-activated neurons. Output is linearly 
    activated to allow positive and negative growth values.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        G (torch tensor): predicted growth values with shape (N, 1)
    '''
    
    def __init__(self, input_features=1, scale=1.7e3):
        
        super().__init__()
        self.inputs = input_features
        self.scale = scale
        self.min = -0.02 / (1/24) # 1/hr -> 1/d
        self.max = 0.1 / (1/24) # 1/hr -> 1/d
        self.mlp = BuildMLP(
            input_features=input_features, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=True)
    
    def forward(self, u, t=None):
        
        if t is None:
            G = self.mlp(u/self.scale)
        else:
            G = self.mlp(torch.cat([u/self.scale, t], dim=1))
        G = self.max * G
        
        return G

class NoDelay(nn.Module):
    
    '''
    Trivial delay function.
    
    Args:
        t (torch tensor): time values with shape (N, 1)
        
    Returns:
        T (torch tensor): ones with shape (N, 1)
    '''
    
    
    def __init__(self):
        
        super().__init__()
        
    def forward(self, t):
        
        T = torch.ones_like(t)
        
        return T

class T_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown time delay function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output is 
    linearly sigmoid-activated to constrain outputs to between 0 and 1.
    
    Args:
        t (torch tensor): time values with shape (N, 1)
        
    Returns:
        T (torch tensor): predicted delay values with shape (N, 1)
    '''
    
    
    def __init__(self):
        
        super().__init__()
        self.mlp = BuildMLP(
            input_features=1, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=nn.Sigmoid())
        
    def forward(self, t):
        
        T = self.mlp(t) 
        
        return T
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, delay=False):
        
        super().__init__()
        
        # surface fitter
        self.surface_fitter = u_MLP()
        
        # pde functions
        self.diffusion = D_MLP()
        self.growth = G_MLP()
        self.delay1 = T_MLP() if delay else NoDelay()
        self.delay2 = self.delay1
        
        # parameter extrema
        self.D_min = self.diffusion.min
        self.D_max = self.diffusion.max
        self.G_min = self.growth.min
        self.G_max = self.growth.max
        self.K = 1.7e3
        
        # input extrema
        self.x_min = 0.075 
        self.x_max = 1.875 
        self.t_min = 0.0
        self.t_max = 2.0

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.D_weight = 1e10 / self.D_max
        self.G_weight = 1e10 / self.G_max
        self.T_weight = 1e10 
        self.dDdu_weight = self.D_weight * self.K
        self.dGdu_weight = self.G_weight * self.K
        self.dTdt_weight = self.T_weight * 2.0
        
        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = 'TmlpDmlp_TmlpGmlp' if delay else 'Dmlp_Gmlp'
    
    def forward(self, inputs):
        
        # cache input batch for pde loss
        self.inputs = inputs
        
        return self.surface_fitter(self.inputs)
    
    def gls_loss(self, pred, true):
        
        residual = (pred - true)**2
        
        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 1][:, None]==0, 
                                self.IC_weight*torch.ones_like(pred), 
                                torch.ones_like(pred))
        
        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, return_mean=True):
        
        # unpack inputs
        x = inputs[:, 0][:,None]
        t = inputs[:, 1][:,None]

        # partial derivative computations 
        u = outputs.clone()
        d1 = Gradient(u, inputs, order=1)
        ux = d1[:, 0][:, None]
        ut = d1[:, 1][:, None]

        # diffusion
        if self.diffusion.inputs == 1:
            D = self.diffusion(u)
        else:
            D = self.diffusion(u, t)

        # growth
        if self.growth.inputs == 1:
            G = self.growth(u)
        else:
            G = self.growth(u, t)

        # time delays
        T1 = self.delay1(t)
        T2 = self.delay2(t)

        # Fisher-KPP equation
        LHS = ut
        RHS = T1*Gradient(D*ux, inputs)[:, 0][:,None] + T2*G*u
        pde_loss = (LHS - RHS)**2

        # constraints on learned parameters
        self.D_loss = 0
        self.G_loss = 0
        self.T_loss = 0
        self.D_loss += self.D_weight*torch.where(
            D < self.D_min, (D-self.D_min)**2, torch.zeros_like(D))
        self.D_loss += self.D_weight*torch.where(
            D > self.D_max, (D-self.D_max)**2, torch.zeros_like(D))
        self.G_loss += self.G_weight*torch.where(
            G < self.G_min, (G-self.G_min)**2, torch.zeros_like(G))
        self.G_loss += self.G_weight*torch.where(
            G > self.G_max, (G-self.G_max)**2, torch.zeros_like(G))

        # derivative constraints on eligible parameter terms
        try:
            dDdu = Gradient(D, u, order=1)
            self.D_loss += self.dDdu_weight*torch.where(
                dDdu < 0.0, dDdu**2, torch.zeros_like(dDdu))
        except:
            pass
        try:
            dGdu = Gradient(G, u, order=1)
            self.G_loss += self.dGdu_weight*torch.where(
                dGdu > 0.0, dGdu**2, torch.zeros_like(dGdu))
        except:
            pass
        try:
            dTdt = Gradient(T1, t, order=1)
            self.T_loss += self.dTdt_weight*torch.where(
                dTdt < 0.0, dTdt**2, torch.zeros_like(dTdt))
        except:
            pass
        
        if return_mean:
            return torch.mean(pde_loss + self.D_loss + self.G_loss + self.T_loss)
        else:
            return pde_loss + self.D_loss + self.G_loss + self.T_loss
    
    def loss(self, pred, true):
        
        self.gls_loss_val = 0
        self.pde_loss_val = 0
        
        # load cached inputs from forward pass
        inputs = self.inputs
        
        # randomly sample from input domain
        x = torch.rand(self.num_samples, 1, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)
        
        # predict surface fitter at sampled points
        outputs_rand = self.surface_fitter(inputs_rand)
        
        # compute surface loss
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true)
        
        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand)
        
        return self.gls_loss_val + self.pde_loss_val
