from sklearn.decomposition import NMF
import torch
from torchnmf.nmf import NMF
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Path to your dataset
dataset_path = "C:/0001_F/00-Dataset_cut/00_Cropping_Dataset/G-Dataset_crop"

# Custom dataset loader for grayscale .bmp images
class dp_Loader(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith('.bmp')
        ]
        self.transform = transform
        self.num_images = len(self.image_paths)
        self.dataset_name = os.path.basename(os.path.normpath(image_dir))  # Get folder name

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')  # Grayscale
        if self.transform:
            img = self.transform(img)
        return img, self.num_images, self.dataset_name

# Define transformations (grayscale, resize, tensor)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the dataset into a DataLoader
dataset = dp_Loader(dataset_path, transform=transform)
img, total_size, dataset_name = dataset[0]
dataloader = DataLoader(dataset, batch_size=total_size, shuffle=False)

# Plot reconstruction error vs. number of components
def show_k_componts_loss(k_range, reconstruction_errors):
    # Use the KneeLocator to find the elbow point
    knee_locator = KneeLocator(k_range, reconstruction_errors, curve='convex', direction='decreasing')
    knee_point = knee_locator.knee
    print("Optimal number of components:", knee_point)

    # Plot the loss curve
    plt.plot(k_range, reconstruction_errors, 'bx-')
    plt.axvline(knee_point, color='red', linestyle='--', linewidth=3)
    plt.axvspan(knee_point - 4, knee_point + 4, color='green', alpha=0.6,
                label='Recommended region \n (k = {} to k = {})'.format(knee_point - 4, knee_point + 4))
    plt.xticks(k_range)
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error (|V - WH|)')
    plt.title('Reconstruction Loss vs. Number of NMF Components')
    plt.legend()
    return plt, knee_point

# Perform NMF on the dataset for a given number of components using GPU
def NMF_Image_GPU(dataloader, n_components):
    for i, (images, total_images, dataset_name) in enumerate(dataloader):
        # Reshape to 2D matrix: (batch_size, H*W)
        images = images.view(images.shape[0], -1)

        # Ensure non-negativity and move to GPU
        images = torch.relu(images).cuda().t()  # Transpose to shape: [pixels, samples]
        print("NMF input shape:", images.shape)

        # Initialize and fit NMF model
        model = NMF(images.shape, rank=n_components).cuda()
        model.fit(images)

        # Reconstruct the input from learned components
        recon = torch.matmul(model.H, model.W.t())
        print("Reconstructed shape:", recon.shape)

        # Compute mean absolute difference: |V - WH|
        abs_diff = torch.abs(images - recon)
        mean_abs_diff = abs_diff.mean()

        return mean_abs_diff.item()

# Test various component values to find the optimal number
mean_difference = []  # Store mean |V - WH| for each k

for k in range(3, 21):
    diff = NMF_Image_GPU(dataloader, n_components=k)
    mean_difference.append(diff)
    print(f'k={k}; Mean reconstruction error: {diff:.6f}')

# Save the loss curve
mean_difference = np.array(mean_difference, dtype='float32')
loss_path = '{}_K_components_loss_test_gpu.npy'.format(dataset_name)
np.save(loss_path, mean_difference)

# Plot and save elbow chart
k_range = list(range(3, 3 + len(mean_difference)))
plt_k_components, elbow_point = show_k_componts_loss(k_range, mean_difference)
fig_path = '{}.png'.format(dataset_name)
plt_k_components.savefig(fig_path)
plt.close()
