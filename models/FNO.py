"""
Adopted from:
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import math
from .utils import format_tensor_size

       
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel = 1):
        super(MLP, self).__init__()
        if kernel == 3:
            self.mlp1 = nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1)
            self.mlp2 = nn.Conv2d(mid_channels, out_channels, 3, stride=1, padding=1)
        elif kernel == 1:
            self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
            self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        else:
            NotImplementedError

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers=4, in_channel_dim=1, out_channel_dim=1, grid=True, norm = True, padding = 5, kernel=1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.in_channel_dim = in_channel_dim
        self.out_channel_dim = out_channel_dim
        self.grid = grid
        self.norm = norm
        self.padding = padding # pad the domain if input is non-periodic
        if norm:
            self.norm = nn.InstanceNorm2d(width)
        print(in_channel_dim + (2 if grid else 0))
        self.p = nn.Conv2d(in_channel_dim + (2 if grid else 0), self.width, 1) # input channel is 3: (a(x, y), x, y)
        self.conv_layers = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(num_layers)])
        self.mlp_layers = nn.ModuleList([MLP(self.width, self.width, self.width, kernel) for _ in range(num_layers)])
        print(f"kernel:{kernel}")
        if kernel == 1:
            self.w_layers = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(num_layers)])
        elif kernel == 3:
            self.w_layers = nn.ModuleList([nn.Conv2d(self.width, self.width, 3, stride=1, padding=1) for _ in range(num_layers)])
        else:
            NotImplementedError

        self.q = MLP(self.width, 1, self.width * 4, kernel) # output channel is 1: u(x, y)

    def forward(self, x, grid = None):
        if self.grid and grid == None:
            grid = self.get_grid(x.shape, x.device)
        if grid != None:
            x = torch.cat((x, grid), dim=1)
        x = self.p(x)
        x = F.pad(x, [0,self.padding, 0,self.padding]) if self.padding > 0 else x

        for i in range(self.num_layers):
            if self.norm:
                x1 = self.norm(self.conv_layers[i](self.norm(x)))
            else:
                x1 = self.conv_layers[i](x)
            x1 = self.mlp_layers[i](x1)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding] if self.padding > 0 else x
        x = self.q(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1)

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()

            # Check if the tensor is complex
            if param.is_complex():
                nbytes += param.data.element_size() * param.numel() * 2  # Multiply by 2 for complex numbers
                nparams += param.numel()
            else:
                nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters in FNO: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams