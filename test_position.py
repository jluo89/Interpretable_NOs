from timeit import default_timer
import sys
import random
import os
import argparse
from collections import OrderedDict
import torch
import numpy as np

# Import custom utilities and modules.
from utils.utilities3 import *
from settings.properties import Training_Properties
from settings.data_module import *
from settings.model_module import *

import matplotlib.pyplot as plt

# ==============================================================================
# 1. Argument Parsing & Environment Setup
# ==============================================================================

parser = argparse.ArgumentParser(description='Analyze receptive field for pre-trained models.')
parser.add_argument('--which_example', type=str, required=True,
                    help="Specify which example to run (e.g., darcy)")
parser.add_argument('--which_model', type=str, required=True, choices=["FNO", "FNOL", "GFNO", "PTFNO", "RFNO", "CNO", "TS", "T1", "GT", "GalerkinTransformer", "SNO","SFNO","FFNO", "ONO", "GroupFNO", "DON"],
                    help="Specify which model to use (e.g., FNO, GFNO, PTFNO, RFNO)")
parser.add_argument('--random_seed', type=int, required=True, help="Specify the random seed (e.g., 0)")
parser.add_argument('--exp_name', type=str, default=None, help="Specify the purpose of the given experiment")
parser.add_argument('--which_device', type=int, default=-1, help="Specify the device")
parser.add_argument('--which_point', type=str, default='mid', help="Specify the device")

args = parser.parse_args()

EXP_PATH=f"Receptive_{args.which_point}"

if not os.path.isdir(EXP_PATH):
    print("Generated new EXP PATH")
    os.mkdir(EXP_PATH)

which_example, which_model, random_seed, exp_name, which_device = args.which_example, args.which_model, args.random_seed, args.exp_name, args.which_device
folder = f"Receptive_{args.which_point}/{args.which_model}_{args.which_example}_seed_{args.random_seed}" if exp_name == None else f"Receptive_{args.which_point}/{args.which_model}_{args.which_example}_exp_{args.exp_name}_seed_{args.random_seed}"

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
device = torch.device(f'cuda:{which_device}' if torch.cuda.is_available() else 'cpu') if which_device != -1 else 'cpu'

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

# ==============================================================================
# 2. Configuration, Data, and Model Initialization
# ==============================================================================
t_properties = Training_Properties(which_example, which_model, exp_name)
t_properties.norm = False # Assuming normalization is handled outside or not needed for analysis
t_properties.save_as_txt(folder + "/training_properties.json")
use_grid = t_properties.grid

# This script is tailored for 'Wave' but can be adapted.
if "wave" in which_example:
    data = Wave(t_properties)
else:
    raise ValueError(f"Dataset '{which_example}' not configured in this script.")

model = Model(t_properties).model.to(device)
n_params = model.print_size()

if t_properties.y_norm:
    y_normalizer = data.y_normalizer
    y_normalizer.to(device)

plot_loader = data.plot_loader
s = t_properties.s
ntest = t_properties.ntest
which_point = args.which_point
myloss = LpLoss(p=2, size_average=False)


# ==============================================================================
# 3. Load Pre-trained Model
# ==============================================================================
# Construct the path to the pre-trained model file.
# The path logic is kept as in the original script. You may need to adjust it.

model_path = f"Receptive_mid/{args.which_model}_{args.which_example}_seed_{args.random_seed}/model.pth" if exp_name is None else f"Receptive_mid/{args.which_model}_{args.which_example}_exp_{args.exp_name}_seed_{args.random_seed}/model.pth"
print(f"Loading model from: {model_path}")
model_state_dict = torch.load(model_path, map_location=device)

# Handle models saved with nn.DataParallel, which adds a 'module.' prefix.
new_state_dict = OrderedDict()
for k, v in model_state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=True)
model.eval()


# ==============================================================================
# 4. Receptive Field Analysis (Simplified)
# ==============================================================================
def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)

def get_grid(shape, device):
    batchsize, size_x, size_y = shape[0], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
    gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
    return torch.cat((gridx, gridy), dim=1)

# ***** Refactor 1: Define Model Categories *****
MODELS_CONCAT_GRID = ["CNO"]
MODELS_SEPARATE_GRID = ["DON", "T1", "FNO"]

# ***** Refactor 2: Consolidate Layer Selection *****
layer = model.q
if which_model == "CNO":
    layer = model.project

input_H, input_W = t_properties.s, t_properties.s
heatmap = np.zeros([input_H, input_W])
pictures = []
num_picture = 10

if which_point == 'mid': x0, y0 = 2, 2
elif which_point == 'ld': x0, y0 = 3, 1
elif which_point == 'lu': x0, y0 = 1, 1
elif which_point == 'rd': x0, y0 = 3, 3
elif which_point == 'ru': x0, y0 = 1, 3

test_l2 = 0.0

for i, (x, y) in enumerate(plot_loader):
    if i >= num_picture: break

    x, y = x.to(device), y.to(device)
    fmap_block, input_block = [], []
    
    # ***** Refactor 3: Refactor the Forward Pass Logic *****
    x.requires_grad = True
    grid = None
    if use_grid:
        grid = get_grid(x.shape, x.device)
        if which_model in ["FNO"]:
            grid.requires_grad = True
            
    layer.register_forward_hook(forward_hook)

    x_input = x
    if use_grid and which_model in MODELS_CONCAT_GRID:
        x_input = torch.cat((x, grid), dim=1)

    if use_grid and which_model in MODELS_SEPARATE_GRID:
        out = model(x_input, grid)
    else:
        out = model(x_input)
    
    out = out.reshape(1, s, s)

    if t_properties.y_norm:
        out = y_normalizer.decode(out)
    
    test_l2 += myloss(out.reshape(1, -1), y.reshape(1, -1)).item()
    
    if which_model in ["DON", "GT", "T1"]:
        feature_map = fmap_block[0].squeeze()
    else:
        feature_map = fmap_block[0].mean(dim=1, keepdim=False).squeeze()
        
    feature_map[feature_map.shape[0]*x0//4-1, feature_map.shape[1]*y0//4-1].backward(retain_graph=True)

    # ***** Refactor 4: Unify Gradient Processing *****
    grad = torch.abs(x.grad)
    if use_grid and which_model in ["FNO"]:
        grad = torch.cat((grad, torch.abs(grid.grad)), dim=1)
    elif use_grid and which_model == "DON":
        grad = grad[:, 0, ...]

    grad = grad.mean(dim=1, keepdim=False) if which_model != "DON" else grad
    grad = grad.squeeze().cpu().numpy()

    norm_grad = grad / (np.linalg.norm(grad) + 1e-9)
    heatmap += norm_grad
    
    element = (norm_grad, x[:, 0, :, :].cpu().detach().squeeze().numpy(), 
               y.cpu().detach().squeeze().numpy(), out.cpu().detach().squeeze().numpy())
    pictures.append(element)

test_l2 /= len(plot_loader)
print(f"Average L2 loss on plot data: {test_l2}")


# ==============================================================================
# 5. Save Results
# ==============================================================================
def save_image(data, path, cmap='gray'):
    if data.max() == data.min():
        # Avoid division by zero if the image is flat
        data_normalized = np.zeros_like(data)
    else:
        data_normalized = ((data - data.min()) / (data.max() - data.min())) * 255
    plt.imshow(data_normalized, cmap=cmap, vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Save the aggregated heatmap
save_image(heatmap, folder+'/trained_image.png')

# Save images for individual samples
for i, (grad_map, x_img, y_img, pred_img) in enumerate(pictures):
    save_image(grad_map, folder+f'/trained_image_{i}.png')
    save_image(x_img, folder+f'/x_image_{i}.png')
    save_image(y_img, folder+f'/y_image_{i}.png')
    save_image(pred_img, folder+f'/pred_image_{i}.png')

# This is redundant since the model was loaded, not trained. 
# It can be removed or kept to save a cleaned-up version of the state dict.
# torch.save(model.state_dict(), folder + "/model.pth")

print(f"Analysis complete. Results saved in: {folder}")