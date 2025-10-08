import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from ipdb import set_trace

# -------------------- 1. Load Data --------------------
# Load the .mat file
set_trace()
data = scipy.io.loadmat('wave_64x64_in_2.mat')  # Replace with your filename

# Extract x and y data
# Assuming x is a 3D tensor of shape (num_examples, r, r)
# Assuming y is a 3D tensor of shape (num_examples, r, r)
x_data = data['x']
y_data = data['y']

# Get the total number of samples
num_samples = x_data.shape[0]


# -------------------- 2. Randomly Select and Extract a Sample --------------------
# Randomly select an integer index from the range [0, num_samples-1]
random_index = np.random.randint(0, num_samples)

print(f"Total number of samples: {num_samples}")
print(f"Randomly selected sample index: {random_index}")

# Extract the corresponding x and y samples
sample_x = x_data[random_index]
sample_y = y_data[random_index]


# -------------------- 3. Visualize and Generate Image --------------------
# Create a figure and subplots, 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the left image: the input x sample
im1 = axes[0].imshow(sample_x, cmap='viridis', origin='lower')
axes[0].set_title(f'Input X (Sample #{random_index})')
fig.colorbar(im1, ax=axes[0])  # Add a colorbar for the left plot

# Plot the right image: the corresponding y sample
im2 = axes[1].imshow(sample_y, cmap='viridis', origin='lower')
axes[1].set_title(f'Output Y (Sample #{random_index})')
fig.colorbar(im2, ax=axes[1])  # Add a colorbar for the right plot

# Adjust layout to prevent titles from overlapping
plt.tight_layout()

# (Optional) If you want to save the generated image to a file, uncomment the line below
# plt.savefig(f'wave_sample_{random_index}.png', dpi=300)

# Display the image in a notebook or viewer
plt.show()