import sys

# Whether or not this code will run on the cluster.
cluster_run = True
cluster_username = 'vli'

# This is to point to local packages on Euler cluster.
if cluster_run:
    sys.path.insert(1, '/cluster/home/' + cluster_username + '/.local/lib64/python2.7/site-packages/sklearn')
    sys.path.insert(1, '/cluster/home/' + cluster_username + '/.local/lib64/python2.7/site-packages/')

import numpy as np
import os
import nibabel
from nilearn import image
import pickle
import src.scripts.get_targets as targets
import src.scripts.fourier as fourier
import src.scripts.svm as svm
import src.scripts.neural_net as nn
import src.scripts.elastic_net as regr
import src.scripts.crop as crop
import src.scripts.histogram as hist
import src.scripts.average as avg

if cluster_run:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Get the data.
NUM_TRAIN_DATA = 278
NUM_TEST_DATA = 138
train_filenames = ["./data/set_train/train_" + str(i) + ".nii" for i in range(1, NUM_TRAIN_DATA + 1)]
test_filenames = ["./data/set_test/test_" + str(i) + ".nii" for i in range(1, NUM_TEST_DATA + 1)]
y = targets.get_targets()

#y = y[:NUM_TRAIN_DATA] # ONLY FOR DEBUGGING WITH SMALLER SETS

### PREPROCESSING ###

# Crop the images (automatically crops the black borders, offsets are starting from the actual brain).
x_crop = 5
y_crop = 5
z_crop = 5
crop_size_str = "_" + str(x_crop) + "_" + str(y_crop) + "_" + str(z_crop) # for name of directory to save to
train_filenames, test_filenames = crop.crop_images(train_filenames, test_filenames, x_crop, y_crop, z_crop, cluster_run, cluster_username)

# TO DO: Break up images into 3x3x3 (or some other size) grids.

### FEATURES ###

# Fourier (with PCA and ANOVA)
fourier_train_feat, fourier_test_feat = fourier.fourier(train_filenames, test_filenames, y, crop_size_str, pca_dim=10, k_best=10, cluster_run=cluster_run, cluster_username=cluster_username)

# Histogram (with ANOVA)
num_bins = 45
hist_train_feat, hist_test_feat = hist.histogram(num_bins, train_filenames, test_filenames, y, crop_size_str, k_best=10, cluster_run=cluster_run, cluster_username=cluster_username)

# Canny filter?
# Watershed?
# Template matching?

# Find hyperparameters for various models.
errors = []           # store the cross-validation errors (needed for averaging)
prediction_files = [] # store the paths to the prediction files (needed for averaging)

### SVM ###

# SVM with fourier features
f_svm_cross_val_error, f_svm_stddev = svm.find_params(fourier_train_feat, y, fourier_test_feat, 'fourier')
errors.append(f_svm_cross_val_error + f_svm_stddev) # weight according to error + stddev to punish high stddev
prediction_files.append('./src/predictions/fourier_svm_pred.csv')

# SVM with histogram features
h_svm_cross_val_error, h_svm_stddev = svm.find_params(hist_train_feat, y, hist_test_feat, 'hist')
errors.append(h_svm_cross_val_error + h_svm_stddev)
prediction_files.append('./src/predictions/hist_svm_pred.csv')

### ElasticNet regression ###

f_regr_cross_val_error, f_regr_stddev = regr.find_params(fourier_train_feat, y, fourier_test_feat, 'regr')
errors.append(f_regr_cross_val_error + f_regr_stddev)
prediction_files.append('./src/predictions/fourier_regr_pred.csv')

h_regr_cross_val_error, h_regr_stddev = regr.find_params(hist_train_feat, y, hist_test_feat, 'regr')
errors.append(h_regr_cross_val_error + h_regr_stddev)
prediction_files.append('./src/predictions/hist_regr_pred.csv')

# seems like neural nets are not good - extremely high error
# f_nn_cross_val_error, f_nn_stddev = nn.find_params(fourier_train_feat, y, fourier_test_feat, 'fourier')
# h_nn_cross_val_error, h_nn_stddev = nn.find_params(hist_train_feat, y, hist_test_feat, 'hist')

# Searchlight? http://nilearn.github.io/auto_examples/02_decoding/plot_haxby_searchlight.html#sphx-glr-auto-examples-02-decoding-plot-haxby-searchlight-py

# Average predictions according to cross-validation error and stddev.
avg.average_predictions(errors, prediction_files, num_test_examples=NUM_TEST_DATA)