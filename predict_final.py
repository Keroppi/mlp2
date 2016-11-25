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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from   src.scripts.model import Model
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

x_min = 2 # pixels to crop from the left ear
x_max = 2 # pixels to crop from the right ear
y_min = 2
y_max = 2
z_min = 2
z_max = 2
crop_str = str(x_min) + "," + str(x_max) + "_" + str(y_min) + "," + str(y_max) + "_" + str(z_min) + "," + str(z_max) # for name of directory or files to save
train_filenames, test_filenames = crop.crop_images(train_filenames, test_filenames, x_min, x_max, y_min, y_max, z_min, z_max, cluster_run, cluster_username)

# Break the brain into a voxel grid and compute features.
for grid_size in (1,):
    ###############

    ### Fourier ###

    params = {}

    pca_param = 7
    kbest_param = 7

    if grid_size > 1:
        pca_param = 2
        kbest_param = 2

    feature_function = fourier.fourier
    fourier_train_feat, fourier_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size,
                                                                       crop_str, 'fourier', feature_function, params,
                                                                       cluster_run, cluster_username, pca_dim=pca_param,
                                                                       k_best=kbest_param)

    ###############

    ### Histogram ###

    kbest_param = 10

    if grid_size > 1:
        kbest_param = 3

    feature_function = hist.histogram
    params = {'num_bins': 50}
    hist_train_feat, hist_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size,
                                                                 crop_str, 'hist', feature_function, params,
                                                                 cluster_run, cluster_username, k_best=kbest_param)

    ###############

    ### Canny Filter ###

    n_dim_param = 400
    params = {'slices': 5}

    if grid_size > 1:
        n_dim_param = 100
        params = {'slices': 1}

    feature_function = canny.canny_filter
    canny_train_feat, canny_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size,
                                                                   crop_str, 'canny', feature_function, params,
                                                                   cluster_run, cluster_username, n_dim=n_dim_param)

    ###############

    ### Histogram of Oriented Gradients ###

    kbest_param = 10
    params = {'slices': 5}

    if grid_size > 1:
        kbest_param = 3
        params = {'slices': 1}

    feature_function = canny.canny_filter
    hog_train_feat, hog_test_feat = feat.compute_grid_features(train_filenames, test_filenames, y, grid_size,
                                                               crop_str, 'hog', feature_function, params,
                                                               cluster_run, cluster_username, k_best=kbest_param)

    ###############

    # Find hyperparameters for various models and output prediction.

    ##### SVM #####

    ### SVM with fourier features ###
    # Parameters to try.
    param_grid = {"probability": [True],
                  "C": [0.01, 0.1, 1, 10],
                  "degree": [3],
                  "kernel": ['poly', 'linear', 'rbf', 'sigmoid'],
                  "tol": [0.001, 0.01, 0.1]}

    svm_model = Model(SVC, param_grid, 'fourier', grid_size, crop_str)

    # Grid search all hyperparameters.
    f_svm_cross_val_error, f_svm_stddev = svm_model.find_hyperparams(fourier_train_feat, y)

    # Predict.
    prediction_file_path = svm_model.output_predictions(fourier_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if f_svm_cross_val_error + f_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(f_svm_cross_val_error + f_svm_stddev)
        prediction_files.append(prediction_file_path)

    ### SVM with histogram features ###
    svm_model = Model(SVC, param_grid, 'hist', grid_size, crop_str)

    # Grid search all hyperparameters.
    h_svm_cross_val_error, h_svm_stddev = svm_model.find_hyperparams(hist_train_feat, y)

    # Predict.
    prediction_file_path = svm_model.output_predictions(hist_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if h_svm_cross_val_error + h_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(h_svm_cross_val_error + h_svm_stddev)
        prediction_files.append(prediction_file_path)

    ### SVM with canny filter features ###
    svm_model = Model(SVC, param_grid, 'canny', grid_size, crop_str)

    # Grid search all hyperparameters.
    c_svm_cross_val_error, c_svm_stddev = svm_model.find_hyperparams(canny_train_feat, y)

    # Predict.
    prediction_file_path = svm_model.output_predictions(canny_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if c_svm_cross_val_error + c_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(c_svm_cross_val_error + c_svm_stddev)
        prediction_files.append(prediction_file_path)

    ### SVM with histogram of oriented gradient features ###
    svm_model = Model(SVC, param_grid, 'hog', grid_size, crop_str)

    # Grid search all hyperparameters.
    hog_svm_cross_val_error, hog_svm_stddev = svm_model.find_hyperparams(hog_train_feat, y)

    # Predict.
    prediction_file_path = svm_model.output_predictions(hog_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if hog_svm_cross_val_error + hog_svm_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(hog_svm_cross_val_error + hog_svm_stddev)
        prediction_files.append(prediction_file_path)

    ###############

    ##### Logistic Regression #####

    ### Logistic Regression with Fourier features ###
    # Parameters to try.
    param_grid = {"penalty": ['l1', 'l2'],
                  "C": [0.001, 0.005, 0.01, 0.05] + [0.1 * x for x in range(1, 50)],
                  "max_iter": [300],
                  "solver": ['liblinear'],
                  "tol": [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
                 }

    lr_model = Model(LogisticRegression, param_grid, 'fourier', grid_size, crop_str)

    # Grid search all hyperparameters.
    f_logreg_cross_val_error, f_logreg_stddev = lr_model.find_hyperparams(fourier_train_feat, y)

    # Predict.
    prediction_file_path = lr_model.output_predictions(fourier_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if f_logreg_cross_val_error + f_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(f_logreg_cross_val_error + f_logreg_stddev)
        prediction_files.append(prediction_file_path)

    ### Logistic Regression with histogram features ###
    lr_model = Model(LogisticRegression, param_grid, 'hist', grid_size, crop_str)

    # Grid search all hyperparameters.
    h_logreg_cross_val_error, h_logreg_stddev = lr_model.find_hyperparams(hist_train_feat, y)

    # Predict.
    prediction_file_path = lr_model.output_predictions(hist_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if h_logreg_cross_val_error + h_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(h_logreg_cross_val_error + h_logreg_stddev)
        prediction_files.append(prediction_file_path)

    ### Logistic Regression with canny filter features ###
    lr_model = Model(LogisticRegression, param_grid, 'canny', grid_size, crop_str)

    # Grid search all hyperparameters.
    c_logreg_cross_val_error, c_logreg_stddev = lr_model.find_hyperparams(canny_train_feat, y)

    # Predict.
    prediction_file_path = lr_model.output_predictions(canny_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if c_logreg_cross_val_error + c_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(c_logreg_cross_val_error + c_logreg_stddev)
        prediction_files.append(prediction_file_path)

    ### Logistic Regression with histogram of oriented gradient features ###
    lr_model = Model(LogisticRegression, param_grid, 'hog', grid_size, crop_str)

    # Grid search all hyperparameters.
    hog_logreg_cross_val_error, hog_logreg_stddev = lr_model.find_hyperparams(hog_train_feat, y)

    # Predict.
    prediction_file_path = lr_model.output_predictions(hog_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if hog_logreg_cross_val_error + hog_logreg_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(hog_logreg_cross_val_error + hog_logreg_stddev)
        prediction_files.append(prediction_file_path)

    ###############

    #####  Random Forest Classifier #####

    ### Random Forest with Fourier features ###
    # Parameters to try.
    param_grid = {"n_estimators": [20],
                  "max_features": [None, "auto", "log2"],
                  "min_samples_split": [2**x / 100.0 for x in range (0, 7)]}

    rf_model = Model(RandomForestClassifier, param_grid, 'fourier', grid_size, crop_str)

    # Grid search all hyperparameters.
    f_dt_cross_val_error, f_dt_stddev = rf_model.find_hyperparams(fourier_train_feat, y)

    # Predict.
    prediction_file_path = rf_model.output_predictions(fourier_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if f_dt_cross_val_error + f_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(f_dt_cross_val_error + f_dt_stddev)
        prediction_files.append(prediction_file_path)

    ### Random Forest with histogram features ###

    rf_model = Model(RandomForestClassifier, param_grid, 'hist', grid_size, crop_str)

    # Grid search all hyperparameters.
    h_dt_cross_val_error, h_dt_stddev = rf_model.find_hyperparams(hist_train_feat, y)

    # Predict.
    prediction_file_path = rf_model.output_predictions(hist_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if h_dt_cross_val_error + h_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(h_dt_cross_val_error + h_dt_stddev)
        prediction_files.append(prediction_file_path)

    ### Random Forest with canny filter features ###

    rf_model = Model(RandomForestClassifier, param_grid, 'canny', grid_size, crop_str)

    # Grid search all hyperparameters.
    c_dt_cross_val_error, c_dt_stddev = rf_model.find_hyperparams(canny_train_feat, y)

    # Predict.
    prediction_file_path = rf_model.output_predictions(canny_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if c_dt_cross_val_error + c_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(c_dt_cross_val_error + c_dt_stddev)
        prediction_files.append(prediction_file_path)

    ### Random Forest with histogram of oriented gradient features ###

    rf_model = Model(RandomForestClassifier, param_grid, 'hog', grid_size, crop_str)

    # Grid search all hyperparameters.
    hog_dt_cross_val_error, hog_dt_stddev = rf_model.find_hyperparams(hog_train_feat, y)

    # Predict.
    prediction_file_path = rf_model.output_predictions(hog_test_feat)

    # Add to the final weighting if the error + stddev is good enough.
    if hog_dt_cross_val_error + hog_dt_stddev < MAX_ERR_PLUS_STDDEV:
        errors.append(hog_dt_cross_val_error + hog_dt_stddev)
        prediction_files.append(prediction_file_path)

    ###############

# Average predictions according to cross-validation error and stddev.
avg.average_predictions(errors, prediction_files, num_test_examples=NUM_TEST_DATA)
