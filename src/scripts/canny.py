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

def canny_filter(train_filenames, test_filenames, y, crop_size_str, k_best, cluster_run, cluster_username='vli'):
    # Train features
    train_feature_vectors = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/train_features/canny" + crop_size_str + "/"
    else:
        save_path = "./src/train_features/canny" + crop_size_str + "/"
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
            img_array = np.load(f)

            x_size, y_size, z_size = img_array.shape

            # Fix X
            x_edges = (feature.canny(img_array[x_size / 2, :, :], sigma=1.75)).astype(int)

            # Fix Y
            y_edges = (feature.canny(img_array[:, y_size / 2, :], sigma=1.75)).astype(int)

            # Fix Z
            z_edges = (feature.canny(img_array[:, :, z_size / 2], sigma=1.75)).astype(int)

            '''
            # display results
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

            ax1.imshow(x_edges, cmap=plt.cm.gray)
            ax1.axis('off')
            ax1.set_title('X', fontsize=20)

            ax2.imshow(y_edges, cmap=plt.cm.gray)
            ax2.axis('off')
            ax2.set_title('Y', fontsize=20)

            ax3.imshow(z_edges, cmap=plt.cm.gray)
            ax3.axis('off')
            ax3.set_title('Z', fontsize=20)

            fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                                bottom=0.02, left=0.02, right=0.98)

            plt.show()
            '''

            canny_vector = np.concatenate((x_edges.flatten(), y_edges.flatten(), z_edges.flatten()))

            np.save(save_path + "feature_vector_" + str(i), canny_vector)
            train_feature_vectors.append(canny_vector)
            print("Saved canny train feature vector #" + str(i))
            i = i + 1

    if cluster_run:
        sys.stdout.flush()

    # Test features
    test_feature_vectors = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/test_features/canny" + crop_size_str + "/"
    else:
        save_path = "./src/test_features/canny" + crop_size_str + "/"

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
            img_array = np.load(f)

            x_size, y_size, z_size = img_array.shape

            # Fix X
            x_edges = feature.canny(img_array[x_size / 2, :, :], sigma=1.75)

            # Fix Y
            y_edges = feature.canny(img_array[:, y_size / 2, :], sigma=1.75)

            # Fix Z
            z_edges = feature.canny(img_array[:, :, z_size / 2], sigma=1.75)

            canny_vector = np.concatenate((x_edges.flatten(), y_edges.flatten(), z_edges.flatten()))

            np.save(save_path + "feature_vector_" + str(i), canny_vector)
            test_feature_vectors.append(canny_vector)
            print("Saved canny test feature vector #" + str(i))
            i = i + 1

    if cluster_run:
        sys.stdout.flush()

    selection = SelectKBest(k=k_best)

    # Use combined features to transform dataset:
    selection.fit(train_feature_vectors, y)
    reduced_train_feature_vectors = selection.transform(train_feature_vectors)
    reduced_test_feature_vectors = selection.transform(test_feature_vectors)

    return (reduced_train_feature_vectors, reduced_test_feature_vectors)
