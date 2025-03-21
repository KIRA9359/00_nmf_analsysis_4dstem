import os.path

import argparse as ap
from matplotlib import pyplot as plt
import cv2

# Faire configuer les parametres
k = 9
# dp_folder = 'E:/Junhao/Dataset_cut/dataset_nanomegas/cluster_{}'.format(k)
# dp_folder = 'C:/00_F/Junhao/Dataset_cut/2-5/Registration_Reconstruction-Overlapping-5%_95%/Overlapping'
dp_folder = 'C:/0001_F/00-Dataset_cut/G-Dataset_figure'
parser = ap.ArgumentParser()
for i in range(k):
    parser.add_argument('-p{}'.format(i + 1),
                        '--pattern_{}'.format(i + 1),
                        type=str,
                        default=os.path.join(dp_folder, 'K-{}_dp-cadre', '{}-cadre.png'.format(k, i + 1)))
    # parser.add_argument('-d{}'.format(i + 1),
    #                     '--directory_K_{}'.format(i + 1),
    #                     type=str,
    #                     default=os.path.join(dp_folder, 'Mapping_jpg/{}.jpg'.format(i + 1)))
    parser.add_argument('-m{}'.format(i + 1),
                        '--directory-raw_K_{}'.format(i + 1),
                        type=str,
                        default=os.path.join(dp_folder, 'raw_Mapping_jpg/{}.jpg'.format(i + 1)))
    # parser.add_argument('-w{}'.format(i + 1),
    #                     '--dis_K_{}'.format(i + 1),
    #                     type=str,
    #                     default='E:/Junhao/Dataset_cut/2_5_Registration_Reconstruction_cropping/w_distribution/Weight_Composant=8_k={}.jpg'.format(
    #                         i + 1))
# for index in range(8):
#     parser.add_argument('-o_{}'.format(index), '--ol_{}'.format(index),
#                         type=str,
#                         default=os.path.join(dp_folder,
#                                              'SecondMaxWeight-{}%_index.png'.format(int((index+1) * 5))))

# parser.add_argument('-c', '--colorbar', type=str, default='2_5_reg_recon.png')
parser.add_argument('-m', '--montagne', type=str, default='G-Montage_dp_mapping.jpg')
parser.add_argument('-dp', '--Dataset_cut', type=str, default=dp_folder)
args = vars(parser.parse_args())
print(args)
fig = plt.figure(figsize=(10, 2))
# setting values to rows and column variables
rows = 2
columns = 9

montagne = os.path.join(args['Dataset_cut'], 'montagne')
print(montagne)
# plt.title('G-dataset')
for i in range(k):
    fig.add_subplot(rows, columns, i + 1)
    # showing image
    mapping_nmf = cv2.imread(args['directory_K_{}'.format(i + 1)], 1)
    mapping_nmf = cv2.cvtColor(mapping_nmf, cv2.COLOR_BGR2RGB)
    plt.imshow(mapping_nmf)
    plt.axis('off')
    plt.title("K={}".format(i + 1))

    fig.add_subplot(rows, columns, i + k + 1)
    # showing image
    # mapping_raw = cv2.imread(args['directory-raw_K_{}'.format(i + 1)], 0)
    mapping_raw = cv2.imread(args['directory_raw_K_{}'.format(i + 1)], 1)
    mapping_raw = cv2.cvtColor(mapping_raw, cv2.COLOR_BGR2RGB)
    plt.imshow(mapping_raw)
    plt.axis('off')

    fig.add_subplot(rows, columns, i + 2 * k + 1)
    # showing image
    pattern = cv2.imread(args['pattern_{}'.format(i + 1)], 1)
    pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
    plt.imshow(pattern)
    plt.axis('off')
    plt.savefig('G-Montage_dp_mapping.jpg',
                dpi=fig.dpi,
                bbox_inches='tight', pad_inches=0.05)
    # plt.title("Com4_K={}".format(i+1+5))

# for i in range(8):
#     fig.add_subplot(rows, columns, i + 1)
#     # showing image
#     # image = cv2.imread(args['ol_{}'.format(i)], 1)
#     image = plt.imread(args['ol_{}'.format(i)], 1)
#     plt.imshow(image)
#     plt.axis('off')
# plt.title("K={}".format(i + 1))


# fig.tight_layout()
plt.tight_layout()
if not os.path.isdir(montagne):
    os.makedirs(montagne)
# plt.savefig(os.path.join(montagne, args["montagne"]),
#             dpi=fig.dpi,
#             bbox_inches='tight', pad_inches=0.05)
plt.show()
