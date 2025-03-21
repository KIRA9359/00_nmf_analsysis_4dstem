from sklearn.decomposition import NMF


import torch
from torchnmf.nmf import NMF
from skimage.io import imread_collection
import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
from kneed import KneeLocator
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = ImageFolder(root="C:/0001_F/00-Dataset_cut/00_Cropping_Dataset/G-Dataset_crop_torch", transform=transform)
dataloader = DataLoader(dataset, batch_size=15675, shuffle=False)

def show_k_componts_loss(k_range, reconstruction_errors):
    # Find the knee point using the 'kneed' library
    knee_locator = KneeLocator(k_range,
                               reconstruction_errors,
                               curve='convex', direction='decreasing')
    # Get the knee point
    knee_point = knee_locator.knee
    print("Optimal number of components:", knee_point)
    plt.plot(k_range, reconstruction_errors, 'bx-')
    plt.axvline(knee_point, color='red', linestyle='--', linewidth=3)
    plt.axvspan(knee_point - 4, knee_point + 4, color='green', alpha=0.6,
                label='Recommended region \n (k = {} to k = {})'.format(knee_point - 4, knee_point + 4))
    plt.xticks(k_range)
    plt.xlabel('number of components')
    plt.ylabel('k_components_loss')
    plt.title('loss tendency with the number of components')
    plt.legend()
    return plt, knee_point


def NMF_Image_GPU(dataloader, n_components):
    for i, (images, labels) in enumerate(dataloader):
        print("原始图片张量形状:", images.shape)  # [batch, 1, 128, 128]
        #
        #     # 4️⃣ 变成 (batch_size, H*W) 的 2D 矩阵
        images = images.view(images.shape[0], -1)  # [batch, 128*128]
        #
        #     # 5️⃣ 确保数据非负
        #     images = torch.relu(images)  # 保险起见，去掉可能的负值
        images = torch.relu(images).cuda().t()  # 保险起见，去掉可能的负值
        #
        print("NMF input shape :", images.shape)  # 预期: [batch_size, 128*128]
        #
        #     X = torch.rand(512 * 512 , 15675)
        #
        model = NMF(images.shape, rank=n_components).cuda()
        #
        #     X = X.cuda()
        #     model = model.cuda()
        #
        model.fit(images)
        print("W SIZE", model.W.size())
        print("H SIZE", model.H.size())
        #
        recon = torch.matmul(model.H, model.W.t())
        #
        #     print("W SIZE : ", model.W)
        #     print("H SIZE : ", model.H)
        #
        print("Recon", recon.shape)
        print("Recon : ", recon)

        # 逐像素计算绝对值
        abs_diff = torch.abs(images - recon)

        # 计算总和
        # sum_abs_diff = abs_diff.sum()

        # mean_absolut_difference
        mean_abs_diff = abs_diff.mean()

        # print("逐像素绝对值差异:\n", abs_diff)
        # print("绝对值差异总和:", sum_abs_diff.item())
        # print("绝对值差异均值:", mean_abs_diff.item())

        return mean_abs_diff.item()


mean_difference = []  # The absolute value between the difference of V-V'(w * h)

for k in range(3, 21, 1):
    diffence = NMF_Image_GPU(dataloader, n_components=k)
    mean_difference.append(diffence)
    print('iter: {}; mean_|v-w@h|:{}'.format(iter, diffence))
mean_difference = np.array(mean_difference, dtype='float32')
current_dataset_k_loss_path = os.path.join('{}_K_components_loss_test_gpu.npy'.format('G-Dataset_crop'))
np.save(current_dataset_k_loss_path, mean_difference)

len_k_loss = len(mean_difference)
k_range = [i for i in range(3, 3 + len_k_loss, 1)]
plt_k_componts, elbow_point = show_k_componts_loss(k_range, mean_difference)
fig_path = '{}.png'.format('G-Dataset_crop')
plt_k_componts.savefig(fig_path)
plt.close()
