import os.path
import numpy as np
from skimage.io import imread_collection
from skimage.util import img_as_float
import imageio
import argparse as ap
import cv2
import matplotlib.pyplot as plt
k = int(input('your k : '))

base_path = 'C:/0001_F/00-Dataset_cut'
eClustering = '00_eClustering'
cropping_dataset = '00_Cropping_Dataset'
type_image = '*.bmp'
# datasetName = '700_24h_O2_crop'
# datasetName = '700_4h_Air_crop'
datasetName = 'NMC-811_zone4_20ms_EDX_Row_222_Col_102_dim_384_384'

# C:/0001_F/00-Dataset_cut/00_eClustering/NMC-811_zone4_20ms_EDX_Row_222_Col_102_dim_384_384/cluster_14/Composant_14_K_1
imageName_path =  os.path.join(base_path, eClustering, datasetName)
# imageName_path = 'E:/gozde/00-Dataset_cut/{}'.format(datasetName)

# DP_dataset = 'C:/0001_F/00-Dataset_cut/00_Cropping_Dataset/{}'.format(datasetName)
DP_dataset = os.path.join(base_path, cropping_dataset, datasetName)

# directory_cluster = '{}/cluster_{}'.format(imageName_path, k)
directory_cluster = '{}/cluster_{}'.format(imageName_path, k)
print(directory_cluster)

# Faire configurer les parametrers
parser = ap.ArgumentParser()
for i in range(k):
    parser.add_argument('-d{}'.format(i + 1), '--directory_K_{}'.format(i + 1),
                        type=str,
                        default=os.path.join(directory_cluster, 'Composant_{}_K_{}'.format(k, i + 1)))
    parser.add_argument('-rd{}'.format(i + 1), '--raw_directory_K_{}'.format(i + 1),
                        type=str,
                        default=os.path.join(directory_cluster, 'Composant_{}_K_{}'.format(k, i + 1)))
parser.add_argument('-dp', '--Dataset_cut', type=str,
                    default=os.path.join(directory_cluster))
args = vars(parser.parse_args())
print(args)

Diffraction_pattern = os.path.join(args['Dataset_cut'], 'Diffraction_pattern')
print(Diffraction_pattern)


# remplir notre liste de image-nparray en utilisant for-boucle
def visualiser_rapidement_img(directoryPath, count):
    # définir une liste
    data = []
    # creating a collection with the available images
    image_dataset = imread_collection(os.path.join(directoryPath, '*.bmp'))
    image_dataset = np.array(image_dataset, dtype="float32")
    print(image_dataset.shape)
    for signal in image_dataset:
        img_float = img_as_float(signal)
        # Calculer gris-valeur pour chaque image
        img_gris_valeur = np.mean(img_float)
        # l'ajouter dans liste
        data.append(img_gris_valeur)

    np_data_gris_valeur = np.array(data, dtype='float32')
    max_light = np.argmax(np_data_gris_valeur)
    value = image_dataset[max_light]
    print(value)

    if not os.path.isdir(Diffraction_pattern):
        os.makedirs(Diffraction_pattern)
    # imageio.imsave(os.path.join(Diffraction_pattern, '{}.bmp'.format(count + 1)), value)
    cv2.imwrite(os.path.join(Diffraction_pattern, '{}.bmp'.format(count + 1)), value)


for i in range(k):
    k_path = args['raw_directory_K_{}'.format(i + 1)]
    visualiser_rapidement_img(k_path, i)
