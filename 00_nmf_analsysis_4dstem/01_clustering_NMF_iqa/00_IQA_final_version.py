import numpy as np
import torch
from piq import gmsd, ssim, mdsi, srsim, dss, haarpsi, psnr
import cv2
from skimage.util import img_as_float
import os
import matplotlib.pyplot as plt

base_path = 'C:/0001_F/00-Dataset_cut'
eClustering = '00_eClustering'
cropping_dataset = '00_Cropping_Dataset'
type_image = '*.bmp'
img_type = 'bmp'
clusterNum = 12
# iqa_list = ['ssim', 'mdsi', 'gmsd']
iqa_list = ['psnr', 'ssim', 'mdsi', 'gmsd']
import matplotlib.patches as patches


# datasetName = "WIthout_anneal_crop_cluster_14"
# datasetName = "2_5_Registration_Reconstruction_cropping"
datasetName = "G-Dataset_crop"
# datasetName = "NMC-811_zone4_20ms_EDX_Row_222_Col_102_dim_384_384"

dp_folder = os.path.join(base_path, eClustering,
                         datasetName ,
                         "cluster_{}".format(clusterNum), "Diffraction_pattern")

iqa_loss_matrix_path = os.path.join(base_path, eClustering,
                                    datasetName,
                                    "cluster_{}".format(clusterNum), "iqa_loss_matrix")

if not os.path.isdir(iqa_loss_matrix_path):
    os.makedirs(iqa_loss_matrix_path)




def calculate_iqa_loss(alg_name):
    loss_matrix = []
    for i in range(clusterNum):
        for j in range(clusterNum):
            img_row = img_as_float(cv2.imread(os.path.join(dp_folder,"{}.{}".format(i + 1, img_type)), 0))
            # print('img_row : ', np.max(img_row), np.min(img_row))
            # print("img row shape", img_row.shape)
            tor_row = torch.unsqueeze(torch.from_numpy(img_row), 0)
            # print("tor row shape", tor_row.shape)
            tor_row = torch.unsqueeze(tor_row, 0)
            # print("tor row shape", tor_row.shape)

            img_col = img_as_float(cv2.imread(os.path.join(dp_folder, "{}.{}".format(j + 1, img_type)), 0))
            # print('img_col : ', np.max(img_col), np.min(img_col))
            tor_col = torch.unsqueeze(torch.from_numpy(img_col), 0)
            tor_col = torch.unsqueeze(tor_col, 0)

            if alg_name == "ssim":
                loss = ssim(tor_row, tor_col)

                loss_matrix.append(loss)
            if alg_name == "gmsd":
                loss = gmsd(tor_row, tor_col)

                loss_matrix.append(loss)
            if alg_name == "mdsi":
                loss = mdsi(tor_row, tor_col)

                loss_matrix.append(loss)
            if alg_name == "psnr":
                loss = psnr(tor_row, tor_col)

                loss_matrix.append(loss)

    return loss_matrix


def get_iqa_matrix(iqa_alg_list):
    for alg_name in iqa_alg_list:
        print('alg_name : ', alg_name)
        iqa_loss_matrix = calculate_iqa_loss(alg_name)
        iqa_loss_matrix = np.array(iqa_loss_matrix, dtype='float32')
        if not os.path.isdir(iqa_loss_matrix_path):
            os.makedirs(iqa_loss_matrix_path)
        np.save('{}/{}_{}_matrix.npy'.format(iqa_loss_matrix_path,
                                             datasetName, alg_name), iqa_loss_matrix)

def blur_lower_triangle(matrix, blur_factor):
    n = len(matrix)
    print("n",n)
    for i in range(n):
        for j in range(i, n):
            print("current j : ", j)
            print("matrix[i][j] : ", i, j)
            # if i==j:
            #     matrix[i][j] = 1
            matrix[i][j] *= blur_factor
    return matrix

def plot_iqa_matrix(iqa_alg_list):
    for alg_name in iqa_alg_list:
        fig = plt.figure()
        ax = plt.gca()

        [plt.text(i-clusterNum+8 - 0.5, -1 + 0.4, f'k={i}', rotation=30)
         for i in range(clusterNum-8,clusterNum +1)]

        print('alg_name : ', alg_name)
        iqa_loss_matrix = calculate_iqa_loss(alg_name)
        iqa_loss_matrix = np.array(iqa_loss_matrix, dtype='float32').reshape(clusterNum, clusterNum)

        # iqa_matrix = iqa_loss_matrix[5:clusterNum, 5:clusterNum]
        iqa_matrix = iqa_loss_matrix[clusterNum - 8 - 1:clusterNum,
                     clusterNum - 8 - 1:clusterNum]
        iqa_matrix_mean = np.mean(iqa_matrix, axis=0)


        iqa_matrix_mean = np.around(iqa_matrix_mean, 4)
        print(iqa_matrix_mean)
        # np.argsort is decreasing from min to max
        if alg_name == 'gmsd' or alg_name == 'mdsi' or alg_name == 'psnr':
            index_value_iqa = np.argsort(iqa_matrix_mean)
            print(index_value_iqa)
            print('{}'.format(alg_name), index_value_iqa, 'max index',index_value_iqa[-1])
            # Add a rectangle (cadre) around the specific column
            rect = patches.Rectangle(
                (index_value_iqa[-1] - 0.5, -0.5),  # Bottom-left corner (adjusted for matrix grid alignment)
                1,  # Width of the column
                iqa_matrix.shape[0],  # Height of the rectangle (equal to number of rows)
                edgecolor='red',  # Outline color
                linewidth=5,  # Thickness of the outline
                facecolor='none'  # No fill
            )
            ax.add_patch(rect)

            k_names = ['k={}'.format(i + 1) for i in range(clusterNum - 8 - 1, clusterNum)]
            print(k_names)

            iqa_matrix = blur_lower_triangle(iqa_matrix, blur_factor=3)
            im = ax.matshow(iqa_matrix, cmap='Blues', interpolation='nearest')
            fig.colorbar(im)

            ax.set_xticks(np.arange(len(iqa_matrix)))
            ax.set_xticklabels(iqa_matrix_mean)
            # ax.set_xticklabels(ssim_matrix_mean)
            ax.set_yticks(np.arange(len(iqa_matrix)))
            ax.set_yticklabels(k_names)

            # Set ticks on both sides of axes on
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            # Rotate and align top ticklabels
            # plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
            #          ha="left", va="center", rotation_mode="anchor")
            # Rotate and align bottom ticklabels
            plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()],
                     rotation=30,
                     ha="right", va="center", rotation_mode="anchor")

            # ax.set_title("{}_{}_matrix_k={}".format(matrix_name, alg_name, k), pad=55)
            fig.tight_layout()
            # ax.title.set_text('{}_matrixBlur_k={}'.format(alg_name, clusterNum))
            # plt.title('{}_matrixBlur_k={}'.format(alg_name, clusterNum))
            # plt.savefig('{}/{}_{}_matrixBlur_k={}.png'.format(iqa_loss_matrix_path,
            #                                                   datasetName, alg_name, clusterNum),
            #             dpi=fig.dpi,
            #             bbox_inches='tight',
            #             pad_inches=0.04)
            plt.savefig('{}_{}_matrixBlur_k={}.png'.format(
                                                              datasetName, alg_name, clusterNum),
                        dpi=fig.dpi,
                        bbox_inches='tight',
                        pad_inches=0.04)
            # plt.show()
        if alg_name == 'ssim':
            index_value_iqa = np.argsort(iqa_matrix_mean)
            print(index_value_iqa)
            print('{}'.format(alg_name), index_value_iqa, 'min index',index_value_iqa[0])

            # Add a rectangle (cadre) around the specific column
            rect = patches.Rectangle(
                (index_value_iqa[0]- 0.5, -0.5),  # Bottom-left corner (adjusted for matrix grid alignment)
                1,  # Width of the column
                iqa_matrix.shape[0],  # Height of the rectangle (equal to number of rows)
                edgecolor='red',  # Outline color
                linewidth=5,  # Thickness of the outline
                facecolor='none'  # No fill
            )
            ax.add_patch(rect)

            k_names = ['k={}'.format(i + 1) for i in range(clusterNum, clusterNum)]
            print(k_names)

            iqa_matrix = blur_lower_triangle(iqa_matrix, blur_factor=3)
            im = ax.matshow(iqa_matrix, cmap='Blues', interpolation='nearest')
            fig.colorbar(im)

            ax.set_xticks(np.arange(len(iqa_matrix)))
            ax.set_xticklabels(iqa_matrix_mean)
            # ax.set_xticklabels(ssim_matrix_mean)
            ax.set_yticks(np.arange(len(iqa_matrix)))
            ax.set_yticklabels(k_names)

            # Set ticks on both sides of axes on
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            # Rotate and align top ticklabels
            # plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
            #          ha="left", va="center", rotation_mode="anchor")
            # Rotate and align bottom ticklabels
            plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()],
                     rotation=30,
                     ha="right", va="center", rotation_mode="anchor")

            # ax.set_title("{}_{}_matrix_k={}".format(matrix_name, alg_name, k), pad=55)
            fig.tight_layout()
            # ax.title.set_text('{}_matrixBlur_k={}'.format(alg_name, clusterNum))
            # plt.title('{}_matrixBlur_k={}'.format(alg_name, clusterNum))
            print('{}/{}_{}_matrixBlur_k={}.png'.format(iqa_loss_matrix_path,
                                                              datasetName, alg_name, clusterNum))
            # plt.savefig('{}/{}_{}_matrixBlur_k={}.png'.format(iqa_loss_matrix_path,
            #                                                   datasetName, alg_name, clusterNum),
            #             dpi=fig.dpi,
            #             bbox_inches='tight',
            #             pad_inches=0.04)
            plt.savefig('{}_{}_matrixBlur_k={}.png'.format(
                                                              datasetName, alg_name, clusterNum),
                        dpi=fig.dpi,
                        bbox_inches='tight',
                        pad_inches=0.04)
            # plt.show()

plot_iqa_matrix(iqa_list)
# get_iqa_matrix(iqa_list)