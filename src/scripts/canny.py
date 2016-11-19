import numpy as np
import os
import nibabel
from nilearn import image
from fourier_transform import FourierTransform
import pickle
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
import sys
from skimage import feature
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import math

def canny_filter(X, y, slices=1):
    x_size, y_size, z_size = X.shape

    # Fix X
    x_edges = np.array([])
    for slice_idx in range(1, slices + 1):
        x_edges = np.append(x_edges, (feature.canny(X[int(math.floor(x_size / (slices + 1.0) * slice_idx)), :, :], sigma=1.75)).astype(int))

    # Fix Y
    y_edges = np.array([])
    for slice_idx in range(1, slices + 1):
        y_edges = np.append(y_edges, (feature.canny(X[:, int(math.floor(y_size / (slices + 1.0) * slice_idx)), :], sigma=1.75)).astype(int))

    # Fix Z
    z_edges = np.array([])
    for slice_idx in range(1, slices + 1):
        z_edges = np.append(z_edges, (feature.canny(X[:, :, int(math.floor(z_size / (slices + 1.0) * slice_idx))], sigma=1.75)).astype(int))

    canny_vector = np.concatenate((x_edges, y_edges, z_edges))

    return canny_vector

