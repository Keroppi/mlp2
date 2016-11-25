import os, sys, pickle
import numpy as np
from src.scripts.voxel_grid import VoxelGrid
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD

def reduce_dimensions(feature_name, X, targets, X_test, pca_dim = 3, k_best = 3, n_dim = 100):
    if feature_name == 'fourier':
        # Scale all data to mean 0, variance 1.
        X_scaler = StandardScaler()
        X_scaler.fit(X)
        X_scaled = X_scaler.transform(X)
        X_test_scaled = X_scaler.transform(X_test)

        # Reduce dimensionality combining PCA and ANOVA.
        pca = PCA(n_components=pca_dim)
        selection = SelectKBest(k=k_best)
        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

        # Use combined features to transform dataset:
        combined_features.fit(X_scaled, targets)
        reduced_X = combined_features.transform(X_scaled)
        reduced_X_test = combined_features.transform(X_test_scaled)
    elif feature_name == 'hist' or feature_name == 'hog':
        selection = SelectKBest(k=k_best)
        selection.fit(X, targets)
        reduced_X = selection.transform(X)
        reduced_X_test = selection.transform(X_test)
    elif feature_name == 'canny':
        lsa = TruncatedSVD(n_components=n_dim, n_iter=7)

        # Use combined features to transform dataset:
        lsa.fit(X)
        reduced_X = lsa.transform(X)
        reduced_X_test = lsa.transform(X_test)

    return (reduced_X, reduced_X_test)


def compute_grid_features(train_filenames, test_filenames, targets, grid_size, crop_size_str, feature_name, feature_function, params, cluster_run, cluster_username='vli', pca_dim=3, k_best=3, n_dim=100):
    # Train features
    train_grids = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/train_features/" + feature_name + "/crop_" + crop_size_str + "/grid_" + str(grid_size) + "/"
    else:
        save_path = "./src/train_features/" + feature_name + "/crop_" + crop_size_str + "/grid_" + str(grid_size) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    missing_grid = False
    for i in range(1, len(train_filenames) + 1):
        try:
            grid = pickle.load(open(save_path + 'grid_' + str(i) + '.pkl', 'rb'))
        except:
            missing_grid = True
            break
        train_grids.append(grid)

    if missing_grid:
        train_grids = []
        i = 1
        for f in train_filenames:
            img_array = np.load(f)

            grid = VoxelGrid(img_array, targets, feature_function, params, grid_size)

            pickle.dump(grid, open(save_path + 'grid_' + str(i) + '.pkl', 'wb'))

            train_grids.append(grid)
            print("Saved " + feature_name + " train grid #" + str(i))

            if cluster_run:
                sys.stdout.flush()

            i = i + 1

    # Test features
    test_grids = []

    if cluster_run:
        save_path = "/cluster/scratch/" + cluster_username + "/src/test_features/" + feature_name + "/crop_" + crop_size_str + "/grid_" + str(grid_size) + "/"
    else:
        save_path = "./src/test_features/" + feature_name + "/crop_" + crop_size_str + "/grid_" + str(grid_size) + "/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    missing_grid = False
    for i in range(1, len(test_filenames) + 1):
        try:
            grid = pickle.load(open(save_path + 'grid_' + str(i) + '.pkl', 'rb'))
        except:
            missing_grid = True
            break
        test_grids.append(grid)

    if missing_grid:
        test_grids = []
        i = 1
        for f in test_filenames:
            img_array = np.load(f)

            grid = VoxelGrid(img_array, targets, feature_function, params, grid_size)

            pickle.dump(grid, open(save_path + 'grid_' + str(i) + '.pkl', 'wb'))

            test_grids.append(grid)
            print("Saved " + feature_name + " test grid #" + str(i))

            if cluster_run:
                sys.stdout.flush()

            i = i + 1

    train_feature_vectors = np.array([])
    test_feature_vectors = np.array([])

    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                train_voxel = []
                for i in range(len(train_filenames)):
                    train_voxel.append(train_grids[i].get_feature_vector(x, y, z))

                test_voxel = []
                for i in range(len(test_filenames)):
                    test_voxel.append(test_grids[i].get_feature_vector(x, y, z))

                train_voxel = np.array(train_voxel)
                test_voxel = np.array(test_voxel)

                reduced_X, reduced_X_test = reduce_dimensions(feature_name, train_voxel, targets, test_voxel, pca_dim, k_best, n_dim)

                if (x == 0 and y == 0 and z == 0):
                    train_feature_vectors = reduced_X
                    test_feature_vectors = reduced_X_test
                else:
                    train_feature_vectors = np.concatenate((train_feature_vectors, reduced_X), axis=1)
                    test_feature_vectors = np.concatenate((test_feature_vectors, reduced_X_test), axis=1)

                print("Finished voxel " + str(x) + ", " + str(y) + ", " + str(z) + ".")
                if cluster_run:
                    sys.stdout.flush()

    return (train_feature_vectors, test_feature_vectors)