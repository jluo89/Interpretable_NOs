from collections import defaultdict

from models.FNO import FNO2d
from models.CNO import CNO2d
from models.GalerkinTransformer import FourierTransformer2D
from models.DON import DON2d
from models.T1 import T1_2d

from ipdb import set_trace

class Model:
    def __init__(self, training_properties):
        if training_properties.which_model == "FNO":
            self.model = FNO2d(training_properties.modes1,
                                training_properties.modes2,
                                training_properties.width,
                                training_properties.layers,
                                training_properties.in_channel_dim,
                                training_properties.out_channel_dim,
                                training_properties.grid,
                                training_properties.norm,
                                kernel=training_properties.kernel,
                                )
        elif training_properties.which_model == "CNO":
            self.model = CNO2d(in_dim  = training_properties.in_channel_dim, 
                            size = training_properties.s, 
                            N_layers = training_properties.N_layers,                    # Number of (D) and (U) Blocks in the network
                            N_res = training_properties.N_res,                          # Number of (R) Blocks per level
                            N_res_neck = training_properties.N_res_neck,
                            channel_multiplier = training_properties.channel_multiplier,
                            out_dim = training_properties.out_channel_dim,
                            grid = training_properties.grid,
                            kernel= training_properties.kernel,
                            )
        elif training_properties.which_model == "DON":
            self.model = DON2d(in_channels=training_properties.in_channels,
                               N_layers=training_properties.N_layers,  # Number of (D) and (U) Blocks in the network
                               N_res=training_properties.N_res,  # Number of (R) Blocks per level
                               kernel_size=training_properties.kernel_size,
                               multiply=training_properties.multiply,
                               basis=training_properties.basis,
                               trunk_layers=training_properties.trunk_layers,
                               trunk_neurons=training_properties.trunk_neurons,
                               N_Fourier_F=training_properties.N_Fourier_F,
                               )
        elif training_properties.which_model == "GalerkinTransformer" or training_properties.which_model == "GT":
            attr_dict = {}
            for attr, value in training_properties.__dict__.items():
                attr_dict[attr] = value
        elif training_properties.which_model == "T1":
            self.model = T1_2d(modes1 = training_properties.modes1,
                               modes2 = training_properties.modes2,
                               width = training_properties.width,
                               nlayers = training_properties.nlayers,
                               signal_resolution = training_properties.signal_resolution,
                               grid = training_properties.grid,
                               )
