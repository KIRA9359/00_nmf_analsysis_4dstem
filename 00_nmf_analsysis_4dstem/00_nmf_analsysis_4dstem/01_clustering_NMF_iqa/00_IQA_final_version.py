import numpy as np
import torch
from piq import gmsd, ssim, mdsi, psnr
import cv2
from skimage.util import img_as_float
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define basic paths and settings
base_path = 'C:/0001_F/00-Dataset_cut'
eClustering = '00_eClustering'
cropping_dataset = '00_Cropping_Dataset'
img_type = 'bmp'
clusterNum = 12  # Total number of clusters

# List of IQA (Image Quality Assessment) algorithms to use
iqa_list = ['psnr', 'ssim', 'mdsi', 'gmsd']

# Choose dataset
datasetName = "G-Dataset_crop"

# Paths for data and results
dp_folder = os.path.join(base_path, eClustering, datasetName,
                         f"cluster_{clusterNum}", "Diffraction_pattern")
iqa_loss_matrix_path = os.path.join(base_path, eClustering,
                                    datasetName, f"cluster_{clusterNum}", "iqa_loss_matrix")
os.makedirs(iqa_loss_matrix_path, exist_ok=True)


# Compute IQA loss matrix for a given algorithm
def calculate_iqa_loss(alg_name):
    loss_matrix = []
    for i in range(clusterNum):
        for j in range(clusterNum):
            img1 = img_as_float(cv2.imread(os.path.join(dp_folder, f"{i + 1}.{img_type}"), 0))
            img2 = img_as_float(cv2.imread(os.path.join(dp_folder, f"{j + 1}.{img_type}"), 0))

            tensor1 = torch.tensor(img1).unsqueeze(0).unsqueeze(0)
            tensor2 = torch.tensor(img2).unsqueeze(0).unsqueeze(0)

            if alg_name == "ssim":
                loss = ssim(tensor1, tensor2)
            elif alg_name == "gmsd":
                loss = gmsd(tensor1, tensor2)
            elif alg_name == "mdsi":
                loss = mdsi(tensor1, tensor2)
            elif alg_name == "psnr":
                loss = psnr(tensor1, tensor2)
            else:
                raise ValueError(f"Unknown algorithm: {alg_name}")

            loss_matrix.append(loss.item())

    return np.array(loss_matrix, dtype='float32')


# Optional: Save the IQA loss matrices
def get_iqa_matrix(iqa_alg_list):
    for alg_name in iqa_alg_list:
        print(f'Calculating: {alg_name}')
        iqa_loss_matrix = calculate_iqa_loss(alg_name)
        np.save(os.path.join(iqa_loss_matrix_path, f'{datasetName}_{alg_name}_matrix.npy'),
                iqa_loss_matrix)


# Emphasize upper triangle and blur (amplify) lower triangle
def blur_lower_triangle(matrix, blur_factor):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j] *= blur_factor
    return matrix


# Plot the IQA similarity matrix with highlights
def plot_iqa_matrix(iqa_alg_list):
    for alg_name in iqa_alg_list:
        fig, ax = plt.subplots()

        [plt.text(i-clusterNum+8 - 0.5, -1 + 0.4, f'k={i}', rotation=30)
         for i in range(clusterNum-8,clusterNum +1)]

        print(f'Processing algorithm: {alg_name}')
        full_matrix = calculate_iqa_loss(alg_name).reshape(clusterNum, clusterNum)

        # Select a focus window: last 9 clusters
        iqa_matrix = full_matrix[clusterNum - 9:clusterNum, clusterNum - 9:clusterNum]
        iqa_matrix_mean = np.around(np.mean(iqa_matrix, axis=0), 4)

        # Determine the best cluster based on metric behavior
        if alg_name in ['ssim']:
            # Lower SSIM means worse → pick minimum index
            target_index = np.argsort(iqa_matrix_mean)[0]
        else:
            # Higher is better for others
            target_index = np.argsort(iqa_matrix_mean)[-1]

        print(f"{alg_name} best cluster index in cropped matrix: {target_index}")

        # Highlight that column
        rect = patches.Rectangle(
            (target_index - 0.5, -0.5),
            1, len(iqa_matrix),
            edgecolor='red',
            linewidth=3,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Create label list
        k_labels = [f'k={i + 1}' for i in range(clusterNum - 9, clusterNum)]

        # Emphasize lower triangle (arbitrary design choice)
        iqa_matrix = blur_lower_triangle(iqa_matrix, blur_factor=3)

        im = ax.matshow(iqa_matrix, cmap='Blues', interpolation='nearest')
        fig.colorbar(im)

        # Tick labels
        ax.set_xticks(np.arange(len(iqa_matrix)))
        ax.set_xticklabels(iqa_matrix_mean)
        ax.set_yticks(np.arange(len(iqa_matrix)))
        ax.set_yticklabels(k_labels)

        ax.tick_params(axis="x", bottom=True, top=True,
                       labelbottom=True, labeltop=False)
        plt.setp(ax.get_xticklabels(), rotation=30,
                 ha="right", va="center", rotation_mode="anchor")

        fig.tight_layout()
        save_path = f"{datasetName}_{alg_name}_matrixBlur_k={clusterNum}.png"
        plt.savefig(save_path, dpi=fig.dpi, bbox_inches='tight', pad_inches=0.04)
        print(f"Saved figure to: {save_path}")
        plt.close(fig)


# --- Run the plotting ---
plot_iqa_matrix(iqa_list)
# get_iqa_matrix(iqa_list)  # Uncomment to save matrix .npy files
