import os.path
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread_collection
from sklearn.decomposition import NMF
import argparse as ap
from imutils import paths
import time
import cv2
import os
from datetime import timedelta

start = time.time()
k = int(input('your k : '))
# Faire configurer les parametrers
# print(imageName_path)
type_image = '*.bmp'

# datasetName = '700_24h_O2_crop'
# datasetName = '700_4h_Air_crop'
# datasetName = '1_0_Original_cropping'
# datasetName = '2_1_Mean(3x3)_cropping'
# datasetName = '2_5_Registration_Reconstruction_cropping'

# 'C:/0001_F/00-Dataset_cut/00_Cropping_Dataset/NMC-811_zone4_20ms_EDX_Row_222_Col_102_dim_384_384'
datasetName = 'NMC-811_zone4_20ms_EDX_Row_222_Col_102_dim_384_384'


# Location of cluster for each dataset
# imageName_path = 'E:/gozde/00-Dataset_cut/{}'.format(datasetName)
imageName_path = 'C:/0001_F/00-Dataset_cut/00_eClustering/{}'.format(datasetName)

# Location of each dataset
# DP_dataset = 'E:/gozde/00-cropping-dataset/{}'.format(datasetName)
DP_dataset = 'C:/0001_F/00-Dataset_cut/00_Cropping_Dataset/{}'.format(datasetName)

image_dir = os.path.join(DP_dataset, type_image)
print(image_dir)
directory_cluster = '{}/cluster_{}'.format(imageName_path, k)
print(directory_cluster)

parser = ap.ArgumentParser()
parser.add_argument('-imagePath', '--image_dir',
                    type=str,
                    default=image_dir)
parser.add_argument('-i', '--imageName_path', type=str, default=DP_dataset)
for i in range(k):
    parser.add_argument('-d{}'.format(i + 1),
                        '--directory_K_{}'.format(i + 1),
                        type=str,
                        default=os.path.join(directory_cluster,
                                             'Composant_{}_K_{}'.format(k, i + 1)))
parser.add_argument('-wh', '--directory_wh', type=str,
                    default=os.path.join(directory_cluster, 'wh_matrix'))
args = vars(parser.parse_args())
print(args)

for count in range(k):
    save_path = args['directory_K_{}'.format(count + 1)]
    print(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
wh_path = args['directory_wh']
print(wh_path)
if not os.path.isdir(wh_path):
    os.makedirs(wh_path)

imagePaths = list(paths.list_images(args["imageName_path"]))
print(imagePaths[0])
print(os.path.split(imagePaths[0])[-1])

image_name = []
for path in imagePaths:
    img_name = os.path.split(path)[-1]
    image_name.append(img_name)
print(len(image_name))
# creating a collection with the available images
image_dataset = imread_collection(args['image_dir'])
print(type(image_dataset))
# image_dataset = np.transpose(np.array(image_dataset, dtype="float32").reshape(9984, -1))
image_dataset = np.array(image_dataset, dtype="float32")
print(image_dataset.shape)
simples, ligne, colume = image_dataset.shape
image_dataset = image_dataset.reshape(simples, -1)
print(image_dataset.shape)
image_dataset = np.transpose(image_dataset)
print(image_dataset.shape)


# Nous pouvons faire décomposition en utilisant NMF
def NMF_Image(image_dataset, n_components):
    w, h = image_dataset.shape
    # new_img = image_dataset.copy()
    nmf = NMF(n_components=n_components,
              init='nndsvd',
              max_iter=1500)
    # W matrice
    W = nmf.fit_transform(image_dataset)
    print("W Matrice :")
    # print(W)
    print(W.shape)
    # H matrice
    H = nmf.components_
    print("H Matrice :")
    # print(H)
    print(H.shape)
    # print('diffence')
    # diffience = image_dataset.reshape(w, h) - W @ H
    # new_img = np.clip(W @ H, 0, 255)
    # print(new_img)
    return {'W': W, 'H': H}


def appler_NMF_Image(k):
    new_image_dateset = []
    for i in [k]:
        print('Number of components:', i)
        out = NMF_Image(image_dataset, n_components=i)
        # new_image = out['new_image']
        W = out['W']
        H = out['H']

        for count, (w_k, k_h) in enumerate(zip(np.transpose(W), H)):
            # k_d = np.float32(w_k @ k_h)
            print(len(k_h))
            print(k_h.shape)
            # np.save('C:/0001_F/00-Dataset_cut/G-Dataset_crop_clu/wh_matrix/{}'.format(count+1), k_d)
            np.save('{}/wh_matrix/{}'.format(directory_cluster, count+1), k_h)
            print('Réussit à créer wh_matrix_{}'.format(count+1))
            w_k = np.expand_dims(w_k, axis=1)
            k_h = np.expand_dims(k_h, axis=0)
            k_d = np.float32(w_k @ k_h)
            print('k_d-shape : ', k_d.shape)
            for simple in range(simples):
                print(simple)
                name = image_name[simple]
                print(name)
                new_img = k_d[:, simple]
                new_img = new_img.reshape(ligne, colume)
                save_path = os.path.join(args['directory_K_{}'.format(count + 1)], name)
                cv2.imwrite(save_path, new_img)
            print('Réussit à créer le nouvel dataset_{}'.format(count + 1))
        print(len(new_image_dateset))


new_image_dataset = appler_NMF_Image(k=k)


# end time
end = time.time()
t = timedelta(seconds=(end - start))
print(str(t))

# total time taken
print("Runtime of the {}_{} is {} s".format(datasetName, k, str(t)))
