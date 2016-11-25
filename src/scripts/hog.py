import numpy as np
from skimage import feature
import math
from skimage import feature

def hog(X, y, slices=1):
    x_size, y_size, z_size = X.shape

    # Fix X
    x_edges = np.array([])
    for slice_idx in range(1, slices + 1):
        x_edges = np.append(x_edges, (feature.hog(X[int(math.floor(x_size / (slices + 1.0) * slice_idx)), :, :], orientations=8, pixels_per_cell=(5, 5), \
                                                  cells_per_block=(1, 1), feature_vector=True)))

    # Fix Y
    y_edges = np.array([])
    for slice_idx in range(1, slices + 1):
        y_edges = np.append(y_edges, (feature.hog(X[:, int(math.floor(y_size / (slices + 1.0) * slice_idx)), :], orientations=8, pixels_per_cell=(5, 5), \
                                                  cells_per_block=(1, 1), feature_vector=True)))

    # Fix Z
    z_edges = np.array([])
    for slice_idx in range(1, slices + 1):
        z_edges = np.append(z_edges, (feature.hog(X[:, :, int(math.floor(z_size / (slices + 1.0) * slice_idx))], orientations=8, pixels_per_cell=(5, 5), \
                                                  cells_per_block=(1, 1), feature_vector=True)))

    hog_vector = np.concatenate((x_edges, y_edges, z_edges))

    return hog_vector

