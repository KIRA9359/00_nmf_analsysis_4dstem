import cv2
import pandas as pd
import torch
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from skimage.util import img_as_float
from piq import gmsd, ssim, mdsi, psnr
import seaborn as sns
import re

# Color dictionary with RGB values
color_map = {
    1: (51, 204, 204),  # Cyan
    2: (255, 0, 0),  # Red
    3: (153, 204, 0),  # Lime Green
    4: (255, 153, 0),  # Orange
    5: (255, 102, 0),  # Dark Orange
    6: (102, 102, 153),  # Greyish Blue
    7: (255, 102, 153),  # Pink
    8: (128, 0, 128),  # Purple
    9: (128, 128, 0),  # Olive
    10: (50, 102, 153),  # Steel Blue
    11: (0, 0, 255),  # Blue
    12: (0, 128, 128),  # Teal
    13: (0, 255, 127),  # Spring Green
    14: (128, 0, 0),  # Maroon
    15: (255, 204, 153),  # Peach
    16: (75, 0, 130),  # Indigo
    17: (139, 69, 19),  # Saddle Brown
    18: (255, 20, 147),  # Deep Pink
    19: (255, 215, 0),  # Gold
    20: (51, 102, 255),  # Dark Blue
    21: (255, 105, 180),  # Hot Pink
    22: (255, 99, 71),  # Tomato
    23: (0, 255, 255),  # Aqua
    24: (128, 128, 128),  # Gray
    25: (255, 165, 0),  # Traditional Orange
    26: (184, 134, 11),  # Dark Goldenrod
    27: (0, 255, 255),  # Turquoise
    28: (250, 128, 114),  # Salmon
    29: (255, 255, 255),  # White
    30: (34, 139, 34),  # Forest Green
    31: (255, 250, 240),  # Floral White
    32: (255, 239, 0),  # Yellow
    33: (0, 0, 139),  # Dark Blue
    34: (255, 228, 196),  # Bisque
    35: (135, 206, 235),  # Sky Blue
    36: (255, 228, 181),  # Moccasin
    37: (50, 205, 50),  # Lime Green
    38: (255, 248, 220),  # Cornsilk
    39: (64, 224, 208),  # Turquoise
    40: (153, 204, 255),  # Light Blue
    41: (255, 182, 193),  # Light Pink
    42: (70, 130, 180),  # Steel Blue
    43: (189, 183, 107),  # Dark Khaki
    44: (255, 127, 80),  # Coral
    45: (75, 0, 130),  # Indigo
    46: (72, 61, 139),  # Dark Slate Blue
    47: (255, 215, 0),  # Gold
    48: (219, 112, 147),  # Pale Violet Red
    49: (0, 191, 255),  # Deep Sky Blue
    50: (255, 99, 71),  # Tomato
    "black": (0, 0, 0)  # Black
}

# Dataset dimensions
row = 95 # replace it by your dataset
col = 165 # replace it by your dataset
cluster = 8  # Once the ideal number of orientations is detected by nmf

threshold = 0.65
seuillage = 75
background = 3
datasetName = "G-Dataset_crop"
basePath = "C:/0001_F/00-Dataset_cut"
cropping_dataset = '00_Cropping_Dataset'
eClustering = "00_eClustering"
wh_matrix = 'wh_matrix'
diffraction_pattern = 'Diffraction_pattern'
dp_with_cadre = 'Dp_with_cadre'
first_index_mapping = 'First_Index_mapping'
overlapping = 'Overlapping'
raw_matched_nmf = 'raw_matched_nmf'
raw_dp_clustering_mean = 'raw_dp_clustering_mean'
raw_nmf_iqa = 'raw_nmf_iqa'

# Path configurations
# dp_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster), diffraction_pattern)
# dp_with_cadre_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster), dp_with_cadre)
# first_index_mapping_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster),
#                                         first_index_mapping)
overlapping_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster), overlapping)
raw_matched_nmf_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster), raw_matched_nmf)
raw_dp_clustering_mean_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster),
                                           raw_matched_nmf, 'raw_dp_clustering_mean')
raw_nmf_iqa_path = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster), raw_matched_nmf,
                                raw_nmf_iqa)
diffraction_pattern_with_cadre = os.path.join(basePath, eClustering, datasetName, 'cluster_{}'.format(cluster), 'Dp_with_cadre')
# 'C:/0001_F/00-Dataset_cut/00_eClustering/G-Dataset_crop/cluster_8/Diffraction_pattern'
nmf_dp_path = os.path.join(basePath, eClustering,
                           datasetName, 'cluster_{}'.format(cluster),
                           diffraction_pattern)
# Create directories if they don't exist
if not os.path.exists(overlapping_path):
    os.makedirs(overlapping_path)

if not os.path.exists(raw_matched_nmf_path):
    os.makedirs(raw_matched_nmf_path)

if not os.path.exists(raw_dp_clustering_mean_path):
    os.makedirs(raw_dp_clustering_mean_path)

if not os.path.exists(raw_nmf_iqa_path):
    os.makedirs(raw_nmf_iqa_path)

# Path to raw diffraction pattern dataset
raw_dp_dataset = os.path.join(basePath, cropping_dataset, datasetName)

# Get all BMP image paths
image_paths = [os.path.join(raw_dp_dataset, f)
               for f in os.listdir(raw_dp_dataset) if f.lower().endswith(('.bmp'))]


def min_max_normalization_2d(data):
    """Normalize 2D data using min-max scaling"""
    data = np.array(data, dtype=np.float32)
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def get_matrix_weight():

    # load the npy file
    """Load and normalize weight matrix from .npy files"""
    eClustering_folder_path = os.path.join(basePath, eClustering)
    componet = "cluster_{}".format(cluster)
    wh_path = os.path.join(eClustering_folder_path, datasetName, componet, wh_matrix)

    # Load all .npy files
    file_paths = glob.glob(os.path.join(wh_path, '*.npy'))
    file_paths.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    list_weights = [np.load(f) for f in file_paths]
    # list_weights = [np.load(f) for f in file_paths]
    k_com = len(list_weights)

    # Normalize weights
    normalized_list_weights = min_max_normalization_2d(list_weights)
    normalized_matrix_weight = normalized_list_weights.reshape(k_com, row, col)

    # Flatten the weights
    w_matrix = []
    for i in range(row):
        for j in range(col):
            for k in range(k_com):
                weight = normalized_matrix_weight[k][i][j]
                w_matrix.append(weight)
    normalized_matrix_weight = np.array(w_matrix).reshape(row * col, k_com)

    return normalized_matrix_weight, k_com


def get_argsort_matrix_weight(matrix_weight):
    """Get indices of sorted weights (descending order)"""
    argsort = np.argsort(matrix_weight)
    first_index = argsort[:, -1]
    second_index = argsort[:, -2]
    third_index = argsort[:, -3]
    return first_index, second_index, third_index


def determination_firstIndex_secondIndex(row, col,
                                         first_index, second_index, third_index,
                                         matrix_weight, threshold,
                                         seuillage):
    """Determine first and second indices based on thresholding"""
    for i, fis_index, sec_index, thirdIndex in zip(range(row * col), first_index, second_index, third_index):
        firstWeight_max = matrix_weight[i][fis_index]
        secWeight_max = matrix_weight[i][sec_index]
        thirdWeight_max = matrix_weight[i][thirdIndex]

        if fis_index == background:  # Background case
            if secWeight_max / firstWeight_max >= threshold:
                first_index[i] = sec_index
                if thirdWeight_max / secWeight_max >= seuillage / 100:
                    second_index[i] = thirdIndex
                else:
                    second_index[i] = 3000
            else:
                second_index[i] = background
        else:
            if sec_index != background:
                if secWeight_max / firstWeight_max < seuillage / 100:
                    second_index[i] = 3000
            else:
                second_index[i] = 3000
    return first_index, second_index


def distribut_color_firstIndexMapping_singleOverlapping(first_index, second_index, row, col,
                                                        background_index=background + 1):
    """Assign colors to first and second indices for visualization"""
    first_color_map, overlapping_color_map = assign_colors(background_index)
    first_index_row_col = np.array(first_index).reshape(row, col)
    second_index_row_col = np.array(second_index).reshape(row, col)
    second_index_row_col[(second_index_row_col != background_index - 1) & (second_index_row_col != 3000)] = 100

    # Create 3D color arrays
    first_index_3d = np.ndarray(shape=(row, col, 3), dtype=int)
    second_index_3d = first_index_3d.copy()

    for i in range(0, row):
        for j in range(0, col):
            first_index_3d[i][j] = first_color_map[first_index_row_col[i][j]]
            second_index_3d[i][j] = overlapping_color_map[second_index_row_col[i][j]]

    # Add black borders for visualization
    second_index_3d = add_black_border_to_array(second_index_3d, 3)
    return first_index_3d, second_index_3d


def distribut_color_75_90_overlapping(second_index, row, col, seuil, background_index=background + 1):
    """Assign colors for overlapping visualization with different thresholds"""
    overlapping_color_map = assign_75_95_overlapping_colors(background_index, seuil)
    second_index_row_col = np.array(second_index).reshape(row, col)
    second_index_row_col[(second_index_row_col != background_index - 1) & (second_index_row_col != 3000)] = 100
    second_index_3d = np.ndarray(shape=(row, col, 3), dtype=int)

    for i in range(0, row):
        for j in range(0, col):
            second_index_3d[i][j] = overlapping_color_map[second_index_row_col[i][j]]

    # Add black borders for visualization
    second_index_3d = add_black_border_to_array(second_index_3d, 5)
    return second_index_3d


def assign_colors(background_index):
    """Create color maps for first index and overlapping regions"""
    firstIndex_color_map = []
    color_keys = [k for k in color_map.keys() if k != "black"]
    color_index = 0

    # Assign colors sequentially, skipping the background index
    for i in range(1, 51):
        if i == background_index:
            firstIndex_color_map.append(color_map["black"])
        else:
            firstIndex_color_map.append(color_map[color_keys[color_index]])
            color_index += 1

    # Color map for overlapping regions
    overlapping_color_map = {
        100: np.array([0, 0, 255]),  # Blue for overlapping
        3000: np.array([255, 255, 255]),  # White for no overlapping
        background_index - 1: np.array([0, 0, 0])  # Black for background
    }

    return firstIndex_color_map, overlapping_color_map


def assign_75_95_overlapping_colors(background_index, seuil):
    """Assign different colors based on overlapping threshold"""
    if seuil == 75:
        return {
            100: np.array([255, 0, 0]),  # Red
            3000: np.array([255, 255, 255]),  # White
            background_index - 1: np.array([0, 0, 0])  # Black
        }
    elif seuil == 80:
        return {
            100: np.array([255, 165, 0]),  # Orange
            3000: np.array([255, 255, 255]),  # White
            background_index - 1: np.array([0, 0, 0])  # Black
        }
    elif seuil == 85:
        return {
            100: np.array([255, 239, 0]),  # Yellow
            3000: np.array([255, 255, 255]),  # White
            background_index - 1: np.array([0, 0, 0])  # Black
        }
    elif seuil == 90:
        return {
            100: np.array([153, 204, 0]),  # Lime Green
            3000: np.array([255, 255, 255]),  # White
            background_index - 1: np.array([0, 0, 0])  # Black
        }
    else:
        return {
            100: np.array([0, 0, 255]),  # Blue (default)
            3000: np.array([255, 255, 255]),  # White
            background_index - 1: np.array([0, 0, 0])  # Black
        }


def show_firstindexMapping(first_index_3d):
    """Display and save first index mapping visualization"""
    fig, ax = plt.subplots(1, 1)
    plt.title('K{}_FirstMaxWeight-threshold-{}'.format(cluster, threshold), x=0.5, y=1.08)
    ax.imshow(first_index_3d)
    ax.axis(False)
    plt.savefig(os.path.join('{}_K{}_FirstMaxWeight-threshold-{}.png'.format(datasetName, cluster, threshold)),
                bbox_inches='tight',
                dpi=fig.dpi, pad_inches=0.05)


def show_overlapping(second_index_3d, seuillage):
    """Display and save overlapping visualization"""
    fig, ax = plt.subplots(1, 1)
    ax.imshow(second_index_3d, interpolation='nearest')
    ax.axis(False)
    fig_path = os.path.join('K_{}_Overlapping_thresholding_{}%.png'.format(cluster, seuillage))
    plt.savefig(fig_path,
                bbox_inches='tight',
                dpi=fig.dpi, pad_inches=0.05)


def add_cadre(cluster, src, dst, background_index=background + 1):
    """Add colored borders to images based on cluster assignment"""
    first_color_map, overlapping_color_map = assign_colors(background_index)
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for i in range(cluster):
        image_border(os.path.join(src, '{}.bmp'.format(i + 1)),
                     os.path.join(dst, '{}.bmp'.format(i + 1)),
                     'a', 21,
                     color=first_color_map[i])


def get_overlapping_evolution_75_to_95():
    """Generate visualization showing overlapping evolution from 75% to 95% threshold"""
    fig, ax = plt.subplots(figsize=(15, 15))
    overlapping_evolution = []

    # Process each threshold from 75 to 95 in steps of 5
    for seuil in range(75, 100, 5):
        matrix_weight, k_com = get_matrix_weight()
        first_index_temp, second_index_temp, third_index_temp = get_argsort_matrix_weight(matrix_weight)
        first_index, second_index = determination_firstIndex_secondIndex(row, col,
                                                                         first_index_temp,
                                                                         second_index_temp,
                                                                         third_index_temp,
                                                                         matrix_weight, threshold, seuil)
        second_index_3d = distribut_color_75_90_overlapping(second_index, row, col, seuil)
        overlapping_evolution.append(second_index_3d)

    # Combine overlapping visualizations
    mask_overlapping_evolution = get_mask_overlapping_evolution(overlapping_evolution)
    for i in range(1, len(mask_overlapping_evolution)):
        overlapping_evolution[0][mask_overlapping_evolution[i]] = overlapping_evolution[i][
            mask_overlapping_evolution[i]]

    ax.imshow(overlapping_evolution[0], interpolation='nearest')
    fig_path = 'K_{}_Overlapping_75%_to_95%.png'.format(cluster)
    plt.savefig(fig_path,
                bbox_inches='tight',
                dpi=fig.dpi, pad_inches=0.05)


def get_mask_overlapping_evolution(overlapping_evolution):
    """Create masks for overlapping evolution visualization"""
    mask_overlapping_evolution = []
    for idx, overlapping in enumerate(overlapping_evolution):
        mask = ((overlapping != [0, 0, 0]).any(axis=-1) &
                (overlapping != [255, 255, 255]).any(axis=-1))
        mask_overlapping_evolution.append(mask)
    return mask_overlapping_evolution


def image_border(src, dst, loc='a', width=35, color=(255, 0, 0)):
    """
    Add border to an image

    Parameters:
    - src: Path to source image
    - dst: Path to save bordered image
    - loc: Border location ('a'=all, 't'=top, 'r'=right, 'b'=bottom, 'l'=left)
    - width: Border width in pixels
    - color: Border color as RGB tuple
    """
    img_ori = Image.open(src)
    w, h = img_ori.size

    # Create new image with border based on location
    if loc in ['a', 'all']:
        w += 2 * width
        h += 2 * width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, width))  # Center original image
    elif loc in ['t', 'top']:
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, width))  # Move original down
    elif loc in ['r', 'right']:
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0))  # Keep original at left
    elif loc in ['b', 'bottom']:
        h += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (0, 0))  # Keep original at top
    elif loc in ['l', 'left']:
        w += width
        img_new = Image.new('RGB', (w, h), color)
        img_new.paste(img_ori, (width, 0))  # Shift original right
    else:
        raise ValueError("Invalid location specified. Use 'a', 't', 'r', 'b', or 'l'.")

    img_new.save(dst)


def add_black_border_to_array(image_array, border_thickness):
    """Add black border to a 3D image array"""
    original_height, original_width, channels = image_array.shape
    new_height = original_height + 2 * border_thickness
    new_width = original_width + 2 * border_thickness

    # Create black array and place original image in center
    new_image_array = np.zeros((new_height, new_width, channels), dtype=image_array.dtype)
    new_image_array[border_thickness:border_thickness + original_height,
    border_thickness:border_thickness + original_width, :] = image_array
    return new_image_array


def get_final_first_and_second_index():
    """Get final first and second indices after thresholding"""
    matrix_weight, k_com = get_matrix_weight()
    first_index_temp, second_index_temp, third_index = get_argsort_matrix_weight(matrix_weight)
    first_index, second_index = determination_firstIndex_secondIndex(row, col,
                                                                     first_index_temp, second_index_temp,
                                                                     third_index,
                                                                     matrix_weight, threshold, seuillage)
    return first_index, second_index


def get_each_dp_name(image_paths):
    """Extract image names from paths"""
    return [os.path.split(path)[-1] for path in image_paths]


def get_raw_dp_clustering_mean(image_dir, current_clustering_number, raw_dp_clustering_mean_path):
    """Calculate mean diffraction pattern for a cluster"""
    current_clustering_dp_files = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]
    current_dp_arrays = []

    # Load all images in cluster
    for dp_name in current_clustering_dp_files:
        img_path = os.path.join(image_dir, dp_name)
        img = cv2.imread(img_path, 0)
        current_dp_arrays.append(np.array(img))

    # Compute mean and save
    stacked_images = np.stack(current_dp_arrays, axis=0)
    mean_raw_dp = np.mean(stacked_images, axis=0)
    mean_raw_dp_clamped = np.clip(mean_raw_dp, 0, 255)
    mean_raw_dp_uint8 = mean_raw_dp_clamped.astype(np.uint8)
    cv2.imwrite(os.path.join(raw_dp_clustering_mean_path, '{}.bmp'.format(current_clustering_number + 1)),
                mean_raw_dp_uint8)


def retour_dp(image_paths, first_indx, second_indx,
              background_index, row, col, rawdp_dataset, raw_matched_nmf_path,
              clustering_number, raw_dp_clustering_mean_path):
    """Organize diffraction patterns into cluster folders and handle overlapping cases"""
    raw_matched_nmf_path_list = creat_MattchedRaw_overlapping_foler(raw_matched_nmf_path, clustering_number)
    image_name = get_each_dp_name(image_paths)
    image_name = np.array(image_name).reshape(row, col)
    first_indx = np.array(first_indx).reshape(row, col)
    second_indx = np.array(second_indx).reshape(row, col)

    # Copy each DP to its assigned cluster folder
    for i in range(0, row):
        for j in range(0, col):
            cluster = first_indx[i][j] + 1
            current_dp_name = image_name[i, j]
            current_second_indx = second_indx[i][j]

            name_extracted = current_dp_name.rsplit('.', 1)[0]
            copy_name = '{}_row_{}_col_{}_cluster_{}.bmp'.format(name_extracted, i, j, cluster)
            src_path = os.path.join(rawdp_dataset, current_dp_name)
            dst_path = os.path.join(raw_matched_nmf_path_list[cluster - 1], copy_name)
            shutil.copyfile(src_path, dst_path)

            # Handle overlapping cases
            if current_second_indx != background_index - 1 and current_second_indx != 3000:
                each_overlapping_folder = os.path.join(overlapping_path, 'overlapping_dp',
                                                       copy_name.rsplit('.', 1)[0])
                if not os.path.exists(each_overlapping_folder):
                    os.makedirs(each_overlapping_folder)
                overlapping = current_second_indx + 1
                overlapping_dst_path = os.path.join(each_overlapping_folder, copy_name)
                shutil.copyfile(src_path, overlapping_dst_path)
                shutil.copyfile(os.path.join(diffraction_pattern_with_cadre, '{}.bmp'.format(cluster)),
                                os.path.join(each_overlapping_folder, '{}.bmp'.format(cluster)))
                shutil.copyfile(os.path.join(diffraction_pattern_with_cadre, '{}.bmp'.format(overlapping)),
                                os.path.join(each_overlapping_folder, '{}.bmp'.format(overlapping)))

    # Calculate mean DP for each cluster
    for current_cluser in range(0, clustering_number):
        get_raw_dp_clustering_mean(raw_matched_nmf_path_list[current_cluser],
                                   current_cluser,
                                   raw_dp_clustering_mean_path)


def creat_MattchedRaw_overlapping_foler(raw_matched_nmf_path, clustering_number):
    """Create folders for each cluster to store matched diffraction patterns"""
    raw_dp_clustering_path_list = []
    for count in range(clustering_number):
        raw_dp_clustering_path = os.path.join(raw_matched_nmf_path,
                                              'raw_dp_clustering_{}_k_{}'.format(clustering_number, count + 1))
        raw_dp_clustering_path_list.append(raw_dp_clustering_path)
        if not os.path.isdir(raw_dp_clustering_path):
            os.makedirs(raw_dp_clustering_path)
    return raw_dp_clustering_path_list


def get_iqa_value_nmf_raw(nmf_dp_path, raw_dp_path):
    """Calculate image quality metrics between NMF and raw diffraction patterns"""
    nmf_dp_path_list = sorted([f for f in os.listdir(nmf_dp_path) if f.endswith('.bmp')])
    raw_dp_path_list = sorted([f for f in os.listdir(raw_dp_path) if f.endswith('.bmp')])

    assert len(nmf_dp_path_list) == len(raw_dp_path_list), "Folders must contain the same number of images."

    # Initialize metric lists
    ssim_loss_matrix = []
    mdsi_loss_matrix = []
    gmsd_loss_matrix = []
    psnr_loss_matrix = []

    # Calculate metrics for each pair
    for nmf_dp, raw_dp in zip(nmf_dp_path_list, raw_dp_path_list):
        current_nmf_dp_path = os.path.join(nmf_dp_path, nmf_dp)
        current_raw_dp_path = os.path.join(raw_dp_path, raw_dp)

        # Convert images to tensors
        img_row = img_as_float(cv2.imread(current_raw_dp_path, 0))
        tor_row = torch.unsqueeze(torch.from_numpy(img_row), 0)
        tor_row = torch.unsqueeze(tor_row, 0)

        img_col = img_as_float(cv2.imread(current_nmf_dp_path, 0))
        tor_col = torch.unsqueeze(torch.from_numpy(img_col), 0)
        tor_col = torch.unsqueeze(tor_col, 0)

        # Calculate metrics
        ssim_loss_matrix.append(ssim(tor_row, tor_col))
        mdsi_loss_matrix.append(mdsi(tor_row, tor_col))
        gmsd_loss_matrix.append(gmsd(tor_row, tor_col))
        psnr_loss_matrix.append(psnr(tor_row, tor_col))

    # Calculate mean values
    psnr_loss_mean = np.mean(np.array(psnr_loss_matrix))
    psnr_iqa_value = np.append(psnr_loss_matrix, psnr_loss_mean)

    mdsi_loss_mean = np.mean(np.array(mdsi_loss_matrix))
    mdsi_iqa_value = np.append(mdsi_loss_matrix, mdsi_loss_mean)

    gmsd_loss_mean = np.mean(np.array(gmsd_loss_matrix))
    gmsd_iqa_value = np.append(gmsd_loss_matrix, gmsd_loss_mean)

    ssim_loss_mean = np.mean(np.array(ssim_loss_matrix))
    ssim_iqa_value = np.append(ssim_loss_matrix, ssim_loss_mean)

    # Stack all metrics
    nmf_raw_iqa_stack = np.stack((psnr_iqa_value,
                                  mdsi_iqa_value,
                                  gmsd_iqa_value,
                                  ssim_iqa_value), axis=0)
    return nmf_raw_iqa_stack


def plot_nmf_raw_iqa(iqa_stack, cluster, raw_nmf_iqa_path, datasetname):
    """Create and save heatmap visualization of IQA metrics"""
    dp_list = ["Raw_{}_{}/NMF_{}_{}".format(cluster, x + 1, cluster, x + 1) for x in range(cluster)]
    dp_list.append("IQA")

    # Create DataFrame for visualization
    df2 = pd.DataFrame(iqa_stack,
                       columns=dp_list,
                       index=["psnr", "mdsi", 'gmsd', 'ssim'])

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = sns.cm.rocket_r  # Reversed color map
    ax = sns.heatmap(df2, annot=True, ax=ax, fmt='.3f', cmap=cmap)

    plt.xlabel('Raw vs NMF')
    plt.ylabel('IQA Algorithm')
    plt.title('{}_Cluster_{} NMF vs Raw'.format(datasetname, cluster))
    plt.xticks(fontsize=7, rotation=20)
    plt.yticks(fontsize=9, rotation=20)

    raw_nmf_fig_path = '{}_Cluster_{}_RAW_NMF_IQA.png'.format(datasetname, cluster)
    plt.savefig(raw_nmf_fig_path,
                dpi=fig.dpi,
                bbox_inches='tight',
                pad_inches=0.04)


# Main execution (commented out)
get_overlapping_evolution_75_to_95()

first_index, second_index = get_final_first_and_second_index()
first_index_3d, second_index_3d = distribut_color_firstIndexMapping_singleOverlapping(first_index, second_index, row, col)
show_firstindexMapping(first_index_3d)
show_overlapping(second_index_3d, seuillage)
retour_dp(image_paths, first_index, second_index,
          row=row, col=col,
          background_index=background + 1,
          rawdp_dataset=raw_dp_dataset,
          raw_matched_nmf_path=raw_matched_nmf_path,
          raw_dp_clustering_mean_path=raw_dp_clustering_mean_path,
          clustering_number=cluster)


add_cadre(cluster,
          src=raw_dp_clustering_mean_path,
          dst='raw_matched_nmf/raw_dp_clustering_cadre')


nmf_raw_iqa_stack = get_iqa_value_nmf_raw(nmf_dp_path,
                                          raw_dp_path=raw_dp_clustering_mean_path)
plot_nmf_raw_iqa(nmf_raw_iqa_stack,
                 cluster=cluster,
                 raw_nmf_iqa_path=raw_nmf_iqa_path, datasetname=datasetName)