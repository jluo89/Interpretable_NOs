import torch
import numpy as np
import torch.nn as nn

from .utils import format_tensor_size
import torch.nn.functional as F

from ipdb import set_trace

# ------------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 kernel_size=3):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.cont_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                   kernel_size=self.kernel_size, stride=1,
                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
        self.sigma = F.leaky_relu
        # conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size, 1, 1)
        # bn = torch.nn.InstanceNorm2d(num_features)
        # relu = torch.nn.ReLU(True)
        # relu = torch.nn.GELU()

        # self.resnet_block = torch.nn.Sequential(
        #    cont_conv,
        #    conv)

    def forward(self, x):
        # out = self.cont_conv(x)
        # out = self.sigma(out)
        p = 0.1
        return self.cont_conv(self.sigma(x)) * p + (1 - p) * x


class Conv2D(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 N_layers,  # Number of layers in the network
                 N_res,
                 kernel_size=3,
                 multiply=32
                 ):  # Number of ResNet Blocks

        super(Conv2D, self).__init__()

        assert N_layers % 2 == 0, "Number of layers myst be even number."

        self.N_layers = N_layers

        #######################################################################################

        self.channel_multiplier = multiply

        in_size = 33
        self.in_size = 33

        self.feature_maps = [in_channels]
        for i in range(0, self.N_layers // 2):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        for i in range(self.N_layers // 2 - 2, -1, -1):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        self.feature_maps.append(1)

        # Define the size of the data for each layer (note that we downsample/usample)
        self.size = []
        for i in range(0, self.N_layers // 2 + 1):
            self.size.append(in_size // (2 ** i))
        for i in range(self.N_layers // 2 - 1, -1, -1):
            self.size.append(in_size // (2 ** i))

        # Define the sizes & number of channels for layers where you x2 upsample and x2 downsample 
        # Note: no changing in the sampling rate in the end of this operation
        # Note: we call this operation size_invariant
        # Note: the size and # of feature maps is the same as before, but we define them for convenience

        self.size_inv = self.size[:-1]
        self.feature_maps_invariant = self.feature_maps[:-1]

        print("size: ", self.size)
        print("channels: ", self.feature_maps)

        assert len(self.feature_maps) == self.N_layers + 1

        self.kernel_size = kernel_size
        self.cont_conv_layers = nn.ModuleList([nn.Conv2d(self.feature_maps[i],
                                                         self.feature_maps[i + 1],
                                                         kernel_size=self.kernel_size, stride=1,
                                                         padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                               for i in range(N_layers)])

        self.cont_conv_layers_invariant = nn.ModuleList([nn.Conv2d(self.feature_maps_invariant[i],
                                                                   self.feature_maps_invariant[i],
                                                                   kernel_size=self.kernel_size, stride=1,
                                                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                                         for i in range(N_layers)])

        self.sigma = F.leaky_relu
        """
        self.cont_conv_layers_invariant = nn.ModuleList([(SynthesisLayer(w_dim=1,
                                                                         is_torgb=False,
                                                                         is_critically_sampled=False,
                                                                         use_fp16=False,

                                                                         # Input & output specifications.
                                                                         in_channels=self.feature_maps_invariant[i],
                                                                         out_channels=self.feature_maps_invariant[i],
                                                                         in_size=self.size_inv[i],
                                                                         out_size=self.size_inv[i],
                                                                         in_sampling_rate=self.size_inv[i],
                                                                         out_sampling_rate=self.size_inv[i],
                                                                         in_cutoff=self.size_inv[i] // cutoff_den,
                                                                         out_cutoff=self.size_inv[i] // cutoff_den,
                                                                         in_half_width=in_half_width,
                                                                         out_half_width=out_half_width))
                                                         for i in range(N_layers)])
        """

        # Define the resnet block --> ##### TO BE DISCUSSED ######

        self.resnet_blocks = []

        # print(self.feature_maps[self.N_layers // 2], )
        for i in range(N_res):
            self.resnet_blocks.append(ResNetBlock(in_channels=self.feature_maps[self.N_layers // 2]))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.N_res = N_res

        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.downsample2 = nn.AvgPool2d(2, stride=2, padding=0)
        self.downsample4 = nn.AvgPool2d(4, stride=4, padding=1)

        self.last = nn.Upsample(size=in_size)

    def forward(self, x):
        # Execute the left part of the network
        # print("")
        for i in range(self.N_layers // 2):
            # print("BEFORE I1",x.shape)
            y = self.cont_conv_layers_invariant[i](x)
            # print("After I1",y.shape)
            y = self.sigma(self.upsample2(y))
            # print("After US1",y.shape)
            x = self.downsample2(y)
            # print("AFTER IS1",x.shape)

            # print("INV DONE")
            y = (self.cont_conv_layers[i](x))
            # print("AFTER CONTCONV", y.shape)
            y = self.upsample2(y)
            # print("AFTER UP", y.shape)
            x = self.downsample4(self.sigma(y))
            # print("AFTER IS2",x.shape)

        for i in range(self.N_res):
            x = self.resnet_blocks[i](x)
            # print("RES",x.shape)

        # Execute the right part of the network
        for i in range(self.N_layers // 2, self.N_layers):
            x = self.downsample2(self.sigma(self.upsample2(self.cont_conv_layers_invariant[i](x))))
            # print("AFTER INV",x.shape)

            x = self.downsample2(self.sigma(self.upsample4(self.cont_conv_layers[i](x))))
            # print("AFTER CONTC",x.shape)

        # print(x.shape[2])
        # print("BEFORE LAST",x.shape)
        # print("-------------")
        # print(" ")

        return self.last(x)

    def get_n_params(self):
        pp = 0

        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class ConvBranch2D(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 N_layers,  # Number of layers in the network
                 N_res,
                 out_channel=1,
                 kernel_size=3,
                 multiply=32,
                 print_bool=False
                 ):  # Number of ResNet Blocks

        super(ConvBranch2D, self).__init__()

        assert N_layers % 2 == 0, "Number of layers myst be even number."

        self.N_layers = N_layers
        self.print_bool = print_bool
        self.channel_multiplier = multiply
        self.feature_maps = [in_channels]
        for i in range(0, self.N_layers):
            self.feature_maps.append(2 ** i * self.channel_multiplier)
        self.feature_maps_invariant = self.feature_maps

        print("channels: ", self.feature_maps)

        assert len(self.feature_maps) == self.N_layers + 1

        self.kernel_size = kernel_size
        self.cont_conv_layers = nn.ModuleList([nn.Conv2d(self.feature_maps[i],
                                                         self.feature_maps[i + 1],
                                                         kernel_size=self.kernel_size, stride=1,
                                                         padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                               for i in range(N_layers)])

        self.cont_conv_layers_invariant = nn.ModuleList([nn.Conv2d(self.feature_maps_invariant[i],
                                                                   self.feature_maps_invariant[i],
                                                                   kernel_size=self.kernel_size, stride=1,
                                                                   padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2))
                                                         for i in range(N_layers)])

        self.sigma = F.leaky_relu

        self.resnet_blocks = []

        for i in range(N_res):
            self.resnet_blocks.append(ResNetBlock(in_channels=self.feature_maps[self.N_layers]))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.N_res = N_res

        self.upsample4 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.downsample2 = nn.AvgPool2d(2, stride=2, padding=0)
        self.downsample4 = nn.AvgPool2d(4, stride=4, padding=1)

        self.flatten_layer = nn.Flatten()
        self.lazy_linear = nn.LazyLinear(out_channel)

    def forward(self, x):
        for i in range(self.N_layers):
            if self.print_bool: print("BEFORE I1", x.shape)
            y = self.cont_conv_layers_invariant[i](x)
            if self.print_bool: print("After I1", y.shape)
            y = self.sigma(self.upsample2(y))
            if self.print_bool: print("After US1", y.shape)
            x = self.downsample2(y)
            if self.print_bool: print("AFTER IS1", x.shape)

            if self.print_bool: print("INV DONE")
            y = self.cont_conv_layers[i](x)
            if self.print_bool: print("AFTER CONTCONV", y.shape)
            y = self.upsample2(y)
            if self.print_bool: print("AFTER UP", y.shape)
            x = self.downsample4(self.sigma(y))
            if self.print_bool: print("AFTER IS2", x.shape)

        for i in range(self.N_res):
            x = self.resnet_blocks[i](x)
            if self.print_bool: print("RES", x.shape)

        x = self.flatten_layer(x)
        if self.print_bool: print("Flattened", x.shape)
        x = self.lazy_linear(x)
        if self.print_bool: print("Linearized", x.shape)
        if self.print_bool: quit()
        return x

    def get_n_params(self):
        pp = 0

        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams

def kaiming_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0.01, nonlinearity='leaky_relu')
        torch.nn.init.zeros_(m.bias.data)


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['swish']:
        return Swish()
    elif name in ['mish']:
        return nn.Mish()
    elif name in ['sin']:
        return Sin()
    else:
        raise ValueError('Unknown activation function')


def init_xavier(model):
    torch.manual_seed(model.retrain)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if model.act_string == "tanh" or model.act_string == "relu" or model.act_string == "leaky_relu":
                gain = nn.init.calculate_gain(model.act_string)
            else:
                gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0)

    model.apply(init_weights)


class FeedForwardNN(nn.Module):

    def __init__(self, input_dimension, output_dimension, layers=8, neurons=256, retrain=4):
        super(FeedForwardNN, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = layers
        self.neurons = neurons
        self.act_string = "leaky_relu"
        # self.retrain = retrain

        # torch.manual_seed(self.retrain)

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        # self.input_layer = nn.Conv2d(in_channels=self.input_dimension,out_channels=self.output_dimension,kernel_size=1)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.batch_layers = nn.ModuleList(
            [nn.BatchNorm2d(self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.activation = activation(self.act_string)

        self.apply(kaiming_init)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for k, (l, b) in enumerate(zip(self.hidden_layers, self.batch_layers)):
            x = b(self.activation(l(x)).permute(0,3,1,2)).permute(0,2,3,1)
        return self.output_layer(x)

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


class DeepOnetNoBiasOrg(nn.Module):
    def __init__(self, branch, trunk):
        super(DeepOnetNoBiasOrg, self).__init__()
        self.branch = branch
        self.trunk = trunk
        self.b0 = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.p = self.trunk.output_dimension

    def forward(self, u_, x_):
        # nx = int(x_.shape[0]**0.5)
        # set_trace()
        weights = self.branch(u_)

        basis = self.trunk(x_.permute(0,2,3,1))
        # set_trace()
        # out = (torch.matmul(weights, basis.T) + self.b0) / self.p ** 0.5
        # out = (torch.bmm(basis, weights) + self.b0) / self.p ** 0.5
        out = (torch.einsum('nxyc,nc->nxy', basis, weights) + self.b0) / self.p ** 0.5
        # set_trace()
        # out = out.reshape(-1, 1, nx, nx)

        return out


    def get_n_params(self):
        pp = 0

        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            if not isinstance(param, torch.nn.parameter.UninitializedParameter):
                nparams += param.numel()
                nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
    
class DON2d(nn.Module):
    def __init__(self,
                 in_channels,
                 N_layers,
                 N_res,
                #  retrain,
                 kernel_size,
                 multiply,
                 basis,
                 N_Fourier_F,
                 trunk_layers,
                 trunk_neurons):

        super(DON2d, self).__init__()
        # print(f"N_layers:{N_layers}")

        branch = ConvBranch2D(in_channels=in_channels,  # Number of input channels.
                              N_layers=N_layers,
                              N_res=N_res,
                              kernel_size=kernel_size,
                              multiply=multiply,
                              out_channel=basis)

        trunk = FeedForwardNN(2 * N_Fourier_F, basis, layers=trunk_layers, neurons=trunk_neurons)

        self.model = DeepOnetNoBiasOrg(branch, trunk)
        self.q = nn.Identity()

    def forward(self,x,grid):
        # grid = grid.reshape(0,-1)
        output = self.model(x,grid)
        output = self.q(output)
        return output

    def get_n_params(self):
        return self.model.get_n_params()

    def print_size(self):
        return self.model.print_size()