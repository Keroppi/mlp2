import numpy as np
import os
import nibabel as nib
from nilearn import image
from nilearn.image import resample_img

train_filenames = np.array(["./data/set_train/train_" + str(i) + ".nii" for i in range(1,279)])

def visualize():
    save_path = "./data/visualize/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cropped = image.load_img(train_filenames[0])

    nib.save(cropped, save_path + 'crop1.nii')


