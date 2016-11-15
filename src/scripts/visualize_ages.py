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
    with open('../data/target_ages.csv', 'r') as fo:
        target_data = fo.read()

    numbers = target_data.split('\n')
    numbers.pop(len(numbers) - 1)

    vectorized_int = np.vectorize(int)
    targets = vectorized_int(numbers)

    return targets

def visualize():
    healthy_total = np.zeros((176, 208, 176), dtype='f')
    sick_total = np.zeros((176, 208, 176), dtype='f')

    targets = get_targets()

    healthy_files = train_filenames[(targets <= 53).nonzero()]
    sick_files = train_filenames[(targets > 53).nonzero()]

    save_path = "../data/visualize/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    healthy_avg = image.mean_img(healthy_files, verbose=2)
    sick_avg = image.mean_img(sick_files, verbose=2)
    '''
    counter = 0
    for f in train_filenames:
        img = image.load_img(f)
        img_array = np.array(img.dataobj[:, :, :, 0], dtype='f')

        if targets[counter] == 1:
            healthy_total = healthy_total + img_array
        else:
            sick_total = sick_total + img_array

        counter = counter + 1
    '''
    print("Done with calculating mean.")

    nib.save(healthy_avg, save_path + 'young.nii')
    nib.save(sick_avg, save_path + 'old.nii')

visualize()