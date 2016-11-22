import sys

# Whether or not this code will run on the cluster.
cluster_run = True
cluster_username = 'vli'

# This is to point to local packages on Euler cluster.
if cluster_run:
    sys.path.append('/cluster/home/' + cluster_username + '/.local/lib64/python2.7/site-packages/skimage')
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
import src.scripts.random_forest as rf
import src.scripts.crop as crop
import src.scripts.histogram as hist
import src.scripts.average as avg
import src.scripts.canny as canny
import src.scripts.logistic_regr as logreg
import src.scripts.features as feat

if cluster_run:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Threshold for rejecting a model (error + stddev has to be lower than this).
MAX_ERR_PLUS_STDDEV = 0.45

# Get the data.
NUM_TRAIN_DATA = 278
NUM_TEST_DATA = 138
train_filenames = ["./data/set_train/train_" + str(i) + ".nii" for i in range(1, NUM_TRAIN_DATA + 1)]
test_filenames = ["./data/set_test/test_" + str(i) + ".nii" for i in range(1, NUM_TEST_DATA + 1)]
y = targets.get_targets()

#y = y[:NUM_TRAIN_DATA] # ONLY FOR DEBUGGING WITH SMALLER SETS

errors = []           # store the cross-validation errors (needed for averaging)
prediction_files = [] # store the paths to the prediction files (needed for averaging)

### PREPROCESSING ###

# Crop the images (automatically crops the black borders, offsets are starting from the actual brain).

x_crop = 2
y_crop = 2
z_crop = 2
crop_size_str = str(x_crop) + "_" + str(y_crop) + "_" + str(z_crop) # for name of directory to save to
train_filenames, test_filenames = crop.crop_images(train_filenames, test_filenames, x_crop, y_crop, z_crop, cluster_run, cluster_username)

# Break the brain into a voxel grid and compute features.
for grid_size in (1, 3):
    ###############

    ### Fourier ###

    params = {}

    pca_param = 7
    kbest_param = 7

    if grid_size > 1:
        pca_param = 2
        kbest_param = 2

    feature_function = fourier.fourier
    fourier_train_feat, fourier_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size, 'fourier', feature_function, params, crop_size_str, cluster_run, cluster_username, pca_dim = pca_param, k_best = kbest_param)

    ###############

    ### Histogram ###

    kbest_param = 10

    if grid_size > 1:
        kbest_param = 3

    feature_function = hist.histogram
    params = {'num_bins': 50}
    hist_train_feat, hist_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size, 'hist', feature_function, params, crop_size_str, cluster_run, cluster_username, k_best = kbest_param)

    ###############

    ### Canny Filter ###

    n_dim_param = 400
    params = {'slices': 5}

    if grid_size > 1:
        n_dim_param = 100
        params = {'slices': 1}

    if grid_size > 3:
        n_dim_param = 25

    feature_function = canny.canny_filter
    canny_train_feat, canny_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size, 'canny', feature_function, params, crop_size_str, cluster_run, cluster_username, n_dim = n_dim_param)

    ###############

    # Find hyperparameters for various models and output prediction.

    ### SVM ###

    # SVM with fourier features
    f_svm_cross_val_error, f_svm_stddev = svm.find_params(fourier_train_feat, y, fourier_test_feat, 'fourier', grid_size)

    if f_svm_cross_val_error + f_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(f_svm_cross_val_error + f_svm_stddev) # weight according to error + stddev to punish high stddev
        prediction_files.append('./src/predictions/fourier_grid_' + str(grid_size) + '_svm_pred.csv')

    # SVM with histogram features
    h_svm_cross_val_error, h_svm_stddev = svm.find_params(hist_train_feat, y, hist_test_feat, 'hist', grid_size)

    if h_svm_cross_val_error + h_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(h_svm_cross_val_error + h_svm_stddev)
        prediction_files.append('./src/predictions/hist_grid_' + str(grid_size) + '_svm_pred.csv')

    # SVM with canny filter features
    c_svm_cross_val_error, c_svm_stddev = svm.find_params(canny_train_feat, y, canny_test_feat, 'canny', grid_size)

    if c_svm_cross_val_error + c_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(c_svm_cross_val_error + c_svm_stddev)
        prediction_files.append('./src/predictions/canny_grid_' + str(grid_size) + '_svm_pred.csv')

    ###############

    ### Logistic Regression ###

    # with fourier features
    f_logreg_cross_val_error, f_logreg_stddev = logreg.find_params(fourier_train_feat, y, fourier_test_feat, 'fourier', grid_size)

    if f_logreg_cross_val_error + f_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(f_logreg_cross_val_error + f_logreg_stddev)
        prediction_files.append('./src/predictions/fourier_grid_' + str(grid_size) + '_logreg_pred.csv')

    # with histogram features
    h_logreg_cross_val_error, h_logreg_stddev = logreg.find_params(hist_train_feat, y, hist_test_feat, 'hist', grid_size)

    if h_logreg_cross_val_error + h_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(h_logreg_cross_val_error + h_logreg_stddev)
        prediction_files.append('./src/predictions/hist_grid_' + str(grid_size) + '_logreg_pred.csv')

    # with canny filter features
    c_logreg_cross_val_error, c_logreg_stddev = logreg.find_params(canny_train_feat, y, canny_test_feat, 'canny', grid_size)

    if c_logreg_cross_val_error + c_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(c_logreg_cross_val_error + c_logreg_stddev)
        prediction_files.append('./src/predictions/canny_grid_' + str(grid_size) + '_logreg_pred.csv')

    ###############

    ###  Random Forest Classifier ###

    f_dt_cross_val_error, f_dt_stddev = rf.find_params(fourier_train_feat, y, fourier_test_feat, 'fourier', grid_size)

    if f_dt_cross_val_error + f_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(f_dt_cross_val_error + f_dt_stddev)
        prediction_files.append('./src/predictions/fourier_grid_' + str(grid_size) + '_rf_pred.csv')

    h_dt_cross_val_error, h_dt_stddev = rf.find_params(hist_train_feat, y, hist_test_feat, 'hist', grid_size)

    if h_dt_cross_val_error + h_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(h_dt_cross_val_error + h_dt_stddev)
        prediction_files.append('./src/predictions/hist_grid_' + str(grid_size) + '_rf_pred.csv')

    c_dt_cross_val_error, c_dt_stddev = rf.find_params(canny_train_feat, y, canny_test_feat, 'canny', grid_size)

    if c_dt_cross_val_error + c_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(c_dt_cross_val_error + c_dt_stddev)
        prediction_files.append('./src/predictions/canny_grid_' + str(grid_size) + '_rf_pred.csv')

    ###############

# Average predictions according to cross-validation error and stddev.
avg.average_predictions(errors, prediction_files, num_test_examples=NUM_TEST_DATA)
