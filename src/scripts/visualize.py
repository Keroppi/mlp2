import numpy as np
import os
import nibabel as nib
from nilearn import image
from sklearn.decomposition import PCA
from fourier_transform import FourierTransform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from nilearn import plotting

train_filenames = np.array(["../data/set_train/train_" + str(i) + ".nii" for i in range(1,279)])


def get_targets():
    target_data = ""
    with open('../data/targets.csv', 'r') as fo:
        target_data = fo.read()

    numbers = target_data.split('\n')
    numbers.pop(len(numbers) - 1)

    vectorized_int = np.vectorize(int)
    targets = vectorized_int(numbers)

    return targets

def visualize():
    healthy_total = np.zeros((176, 208, 176), dtype='f')
    sick_total = np.zeros((176, 208, 176), dtype='f')

    num_healthy = 211
    num_sick = 67

    targets = get_targets()

    healthy_files = train_filenames[np.nonzero(targets)]
    sick_files = train_filenames[(targets == 0).nonzero()]

    save_path = "../data/visualize/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    healthy_avg = image.mean_img(healthy_files, verbose=2)
    sick_avg = image.mean_img(sick_files, verbose=2)

    nib.save(healthy_avg, save_path + 'healthy.nii')
    nib.save(sick_avg, save_path + 'sick.nii')

visualize()
