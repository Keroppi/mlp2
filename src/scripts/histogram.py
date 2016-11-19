import numpy as np
import os
import nibabel
from nilearn import image
from sklearn.decomposition import PCA
from fourier_transform import FourierTransform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import neural_net as nn


def histogram(X, y, num_bins):
    hist, bin_edges = np.histogram(X, bins=num_bins, range=(0, 5000), density=False)

    return hist