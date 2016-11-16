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

def fourier(train_filenames, test_filenames, y, crop_size_str, pca_dim, k_best, cluster_run, cluster_username='vli'):
    # Train features
    train_feature_vectors = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/train_features/fourier" + crop_size_str + "/"
    else:
        save_path = "./src/train_features/fourier" + crop_size_str + "/"
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
            fft = FourierTransform(img_array)
            fourier_transform_result = ((fft.return_output_array()).flatten()).real

            np.save(save_path + "feature_vector_" + str(i), fourier_transform_result)
            train_feature_vectors.append(fourier_transform_result)
            print("Saved fourier train feature vector #" + str(i))
            i = i + 1

    if cluster_run:
        sys.stdout.flush()

    # Test features
    test_feature_vectors = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/test_features/fourier" + crop_size_str + "/"
    else:
        save_path = "./src/test_features/fourier" + crop_size_str + "/"

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
            fft = FourierTransform(img_array)
            fourier_transform_result = ((fft.return_output_array()).flatten()).real

            np.save(save_path + "feature_vector_" + str(i), fourier_transform_result)
            test_feature_vectors.append(fourier_transform_result)
            print("Saved fourier test feature vector #" + str(i))
            i = i + 1

    if cluster_run:
        sys.stdout.flush()

    # Scale all data to mean 0, variance 1.
    X_scaler = StandardScaler()
    X_scaler.fit(np.array(train_feature_vectors)) #X_scaler.fit(np.array(train_feature_vectors + test_feature_vectors))
    train_feature_vectors = X_scaler.transform(train_feature_vectors)
    test_feature_vectors = X_scaler.transform(test_feature_vectors)

    # Reduce dimensionality combining PCA and ANOVA.
    pca = PCA(n_components=pca_dim)
    selection = SelectKBest(k=k_best)
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Use combined features to transform dataset:
    combined_features.fit(train_feature_vectors, y)
    reduced_train_feature_vectors = combined_features.transform(train_feature_vectors)
    reduced_test_feature_vectors = combined_features.transform(test_feature_vectors)

    return (reduced_train_feature_vectors, reduced_test_feature_vectors)
