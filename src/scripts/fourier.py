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

def fourier(X, y):
    fft = FourierTransform(X)
    fourier_transform_result = ((fft.return_output_array()).flatten()).real

    return fourier_transform_result
