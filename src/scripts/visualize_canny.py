import numpy as np
import os
import nibabel
from nilearn import image
from fourier_transform import FourierTransform
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
import sys
from skimage import feature
import matplotlib.pyplot as plt

image_path = "../../data/set_train/train_100.nii"

# Visualize various values of sigma.
def show_canny():
    img = image.load_img(image_path)
    im = np.array(img.dataobj[:, 103, :, 0], dtype='f')

    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im)
    edges2 = feature.canny(im, sigma=1.75)

    # display results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

    #print(im[103, :, :].shape)
    #print(im[:, 103, :].shape)
    #print(im[:, :, 103].shape)

    ax1.imshow(im, cmap=plt.cm.jet)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)

    ax2.imshow(edges1, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

    ax3.imshow(edges2, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Canny filter, $\sigma=1.75$', fontsize=20)

    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                        bottom=0.02, left=0.02, right=0.98)

    plt.show()

show_canny()