from timeit import default_timer
import sys
import random
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.utilities3 import *
from settings.properties import Training_Properties
from settings.data_module import *
from settings.model_module import *
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Argument Parsing
# ==============================================================================
parser = argparse.ArgumentParser(description='Train models for various examples.')
parser.add_argument('--which_example', type=str, required=True,
                    help="Specify which example to run (e.g., darcy)")
parser.add_argument('--which_model', type=str, required=True, choices=["FNO", "FNOL", "GFNO", "PTFNO", "RFNO", "CNO", "TS", "T1", "GT", "GalerkinTransformer", "SNO","SFNO","FFNO", "ONO", "GroupFNO", "DON"],
                    help="Specify which model to use (e.g., FNO, GFNO, PTFNO, RFNO)")
parser.add_argument('--random_seed', type=int, required=True, help="Specify the random seed (e.g., 0)")
parser.add_argument('--exp_name', type=str, default=None, help="Specify the purpose of the given experiment")
parser.add_argument('--which_device', type=int, default=-1, help="Specify the device")
parser.add_argument('--which_point', type=str, default='mid', help="Specify the device")

args = parser.parse_args()


# ==============================================================================
# 2. Environment Setup
# ==============================================================================
which_example, which_model, random_seed, exp_name, which_device = args.which_example, args.which_model, args.random_seed, args.exp_name, args.which_device
EXP_PATH = f"./Receptive_{args.which_point}"
if not os.path.isdir(EXP_PATH):
    print("Generated new EXP PATH")
    os.mkdir(EXP_PATH)

folder = f"./Receptive_{args.which_point}/{args.which_model}_{args.which_example}_seed_{args.random_seed}" if exp_name == None else f"./Receptive_{args.which_point}/{args.which_model}_{args.which_example}_exp_{args.exp_name}_seed_{args.random_seed}"

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device(f'cuda:{which_device}' if torch.cuda.is_available() else 'cpu') if which_device != -1 else 'cpu'

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")
print(device)

if not os.path.isdir("TrainedModels"):
    print("Generated new folder TrainedModels")
    os.mkdir("TrainedModels")
if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)


# ==============================================================================
# 3. Config, Data, and Model Initialization
# ==============================================================================
t_properties = Training_Properties(which_example, which_model, exp_name)
t_properties.norm = False
print("Normalization is applied in the model." if t_properties.norm else "No normalization in the model.")
t_properties.save_as_txt(folder + "/training_properties.json")

data = Wave(t_properties)

model = Model(t_properties).model.to(device)

n_params = model.print_size()


# ==============================================================================
# 4. Dataloaders and Training Settings
# ==============================================================================
if t_properties.y_norm:
    y_normalizer = data.y_normalizer
    y_normalizer.to(device)
train_loader = data.train_loader
test_loader = data.test_loader
plot_loader = data.plot_loader

learning_rate = t_properties.learning_rate
iterations = t_properties.iterations
epochs = t_properties.epochs
use_grid = t_properties.grid

batch_size = t_properties.batch_size
s = t_properties.s
ntrain = t_properties.ntrain
ntest = t_properties.ntest
which_point = args.which_point


# ==============================================================================
# 5. Training Loop
# ==============================================================================
print(f"Batch_size:{t_properties.batch_size}\nTrain loader length:{len(train_loader)}\Test loader length:{len(test_loader)}")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(p=2, size_average=False)

f = open(folder + "/record_l2_losses.txt", "w")

def get_grid(shape, device):
    batchsize, size_x, size_y = shape[0], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
    gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
    return torch.cat((gridx, gridy), dim=1)

# Categorize models by how they handle the grid to avoid repeated checks in the loop.
MODELS_CONCAT_GRID = ["CNO"] # These models require concatenating the grid to the input x.
MODELS_SEPARATE_GRID = ["DON", "T1", "FNO"]  # These models take the grid as a separate argument.

best_model_state = None
best_loss = 1e10

for epoch in range(epochs):
    model.train()
    train_l2 = 0
    # Use tqdm for a progress bar
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}") as tepoch:
        for x, y in tepoch:
            x, y = x.to(device), y.to(device)
            
            # Prepare grid input based on model type
            grid = None
            if use_grid:
                grid = get_grid(x.shape, x.device)
                if which_model in MODELS_CONCAT_GRID:
                    x = torch.cat((x, grid), dim=1)

            optimizer.zero_grad()
            
            # Determine the forward call based on model category
            if use_grid and which_model in MODELS_SEPARATE_GRID:
                out = model(x, grid)
            else:
                out = model(x)
            
            out = out.reshape(batch_size, s, s)

            if t_properties.y_norm:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            loss = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

            # Update the progress bar's postfix
            tepoch.set_postfix(train_loss=loss.item())

    # Evaluation phase uses the same simplified logic
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            grid = None
            if use_grid:
                grid = get_grid(x.shape, x.device)
                if which_model in MODELS_CONCAT_GRID:
                    x = torch.cat((x, grid), dim=1)

            if use_grid and which_model in MODELS_SEPARATE_GRID:
                out = model(x, grid)
            else:
                out = model(x)

            out = out.reshape(batch_size, s, s)

            if t_properties.y_norm:
                out = y_normalizer.decode(out)
            test_l2 += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    if test_l2 < best_loss:
        best_loss = test_l2
        best_model_state = model.state_dict()

    print(f"Epoch {epoch} | Relative Train Loss: {train_l2:.4f} | Relative Test Loss: {test_l2:.4f}")
    if epoch % 50 == 0 or epoch == epochs - 1:
        f.write(f"Epoch {epoch}, Train L2: {train_l2}, Test L2: {test_l2}\n")
f.close()

model.load_state_dict(best_model_state, strict=True)


# ==============================================================================
# 6. Receptive Field Analysis
# ==============================================================================
def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)

# ***** Refactor 2: Consolidate Layer Selection *****
# Default layer is model.q, only handle special cases.
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

for i, (x, y) in enumerate(plot_loader):
    if i >= num_picture: break # Only process the number of images we need to save

    x, y = x.to(device), y.to(device)
    fmap_block, input_block = [], []

    # ***** Refactor 3: Refactor the Forward Pass Logic *****
    # Pull out repeated code (like requires_grad, register_forward_hook).
    x.requires_grad = True
    grid = None
    if use_grid:
        grid = get_grid(x.shape, x.device)
        # Some models require the grid to have a gradient as well.
        if which_model in ["FNO"]:
            grid.requires_grad = True
    
    layer.register_forward_hook(forward_hook)

    # Execute the forward pass based on the predefined model categories.
    x_input = x
    if use_grid and which_model in MODELS_CONCAT_GRID:
        x_input = torch.cat((x, grid), dim=1)

    if use_grid and which_model in MODELS_SEPARATE_GRID:
        out = model(x_input, grid)
    else:
        out = model(x_input)
    
    out = out.reshape(1, s, s)

    # Extract feature map
    if which_model in ["DON", "GT", "T1"]:
        feature_map = fmap_block[0].squeeze()
    else:
        feature_map = fmap_block[0].mean(dim=1, keepdim=False).squeeze()

    # Backpropagate from a single point in the feature map
    feature_map[feature_map.shape[0]*x0//4-1, feature_map.shape[1]*y0//4-1].backward(retain_graph=True)

    # ***** Refactor 4: Unify Gradient Processing *****
    grad = torch.abs(x.grad)
    if use_grid and which_model in ["FNO"]:
        grad = torch.cat((grad, torch.abs(grid.grad)), dim=1)
    elif use_grid and which_model == "DON":
        grad = grad[:, 0, ...] # Only take the gradient of the original input, ignoring the grid's.

    # Reduce the dimensionality of the gradient map
    grad = grad.mean(dim=1, keepdim=False) if which_model != "DON" else grad
    grad = grad.squeeze().cpu().numpy()

    # Accumulate into the heatmap
    norm_grad = grad / np.linalg.norm(grad)
    heatmap += norm_grad
    
    # Store images for later saving
    element = (
        norm_grad,
        x[:, 0, :, :].cpu().detach().squeeze().numpy(),
        y.cpu().detach().squeeze().numpy(),
        out.cpu().detach().squeeze().numpy()
    )
    pictures.append(element)

# ==============================================================================
# 7. Save Results
# ==============================================================================
# A helper function can be defined to save images and avoid repetition.
def save_image(data, path, cmap='gray'):
    # Normalize data to the 0-255 range
    data_normalized = ((data - data.min()) / (data.max() - data.min() + 1e-9)) * 255
    plt.imshow(data_normalized, cmap=cmap, vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close() # Close the plot to prevent memory accumulation

# Save the aggregated heatmap
save_image(heatmap, folder+'/trained_image.png')

# Save images for individual samples
for i, (grad_map, x_img, y_img, pred_img) in enumerate(pictures):
    save_image(grad_map, folder+f'/trained_image_{i}.png')
    save_image(x_img, folder+f'/x_image_{i}.png')
    save_image(y_img, folder+f'/y_image_{i}.png')
    save_image(pred_img, folder+f'/pred_image_{i}.png')

torch.save(model.state_dict(), folder + "/model.pth")