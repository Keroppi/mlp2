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
from sklearn.feature_selection import SelectKBest

# Every function returns a tuple (train_feature_vectors, test_feature_vectors)
# Tries to load from disk if possible, otherwise it saves them to disk.

def histogram(num_of_bins, train_filenames, test_filenames, y, crop_size_str, k_best, cluster_run, cluster_username='vli'):
    # Train features
    train_feature_vectors = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/train_features/histogram" + crop_size_str + "_" + str(num_of_bins) + "bin/"
    else:
        save_path = "./src/train_features/histogram" + crop_size_str + "_" + str(num_of_bins) + "bin/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    missing_feature = False
    for i in range(1, len(train_filenames) + 1):
        try:
            vector = np.load(save_path + "feature_vector_" + str(i) + ".npy")
        except:
            missing_feature = True
            break
        train_feature_vectors.append(vector)

    if missing_feature:
        train_feature_vectors = []
        i = 1
        for f in train_filenames:
            #img = image.load_img(f)
            img_array = np.load(f) #np.array(img.dataobj[:, :, :, 0], dtype='f')
            hist, bin_edges = np.histogram(img_array, bins=num_of_bins, range=(0, 5000), density=False)

            np.save(save_path + "feature_vector_" + str(i), hist)
            train_feature_vectors.append(hist)
            print("Saved histogram train feature vector #" + str(i))
            i = i + 1

    # Test features
    test_feature_vectors = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/test_features/histogram" + crop_size_str + "_" + str(num_of_bins) + "bin/"
    else:
        save_path = "./src/test_features/histogram" + crop_size_str + "_" + str(num_of_bins) + "bin/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    missing_feature = False
    for i in range(1, len(test_filenames) + 1):
        try:
            vector = np.load(save_path + "feature_vector_" + str(i) + ".npy")
        except:
            missing_feature = True
            break
        test_feature_vectors.append(vector)

    if missing_feature:
        test_feature_vectors = []
        i = 1
        for f in test_filenames:
            #img = image.load_img(f)
            img_array = np.load(f) #np.array(img.dataobj[:, :, :, 0], dtype='f')
            hist, bin_edges = np.histogram(img_array, bins=num_of_bins, range=(0, 5000), density=False)

            np.save(save_path + "feature_vector_" + str(i), hist)
            test_feature_vectors.append(hist)
            print("Saved histogram test feature vector #" + str(i))
            i = i + 1

    selection = SelectKBest(k=k_best)

    # Use combined features to transform dataset:
    selection.fit(train_feature_vectors, y)
    reduced_train_feature_vectors = selection.transform(train_feature_vectors)
    reduced_test_feature_vectors = selection.transform(test_feature_vectors)

    return (reduced_train_feature_vectors, reduced_test_feature_vectors)