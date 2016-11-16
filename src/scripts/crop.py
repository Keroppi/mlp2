import numpy as np
import os
from nilearn import image
import sys

# Approximate indices of where the brain starts and ends.
# x,y,z sizes = 139x173x147 with 1 pixel black boundary
Z_MIN = 6 # bottom of brain stem
Z_MAX = 153

Y_MIN = 17
Y_MAX = 190 # forehead

X_MIN = 20 # ear to ear
X_MAX = 159


# If x_diff is 20, then we select the image from [x_min + 20, x_max - 20)
# Saves sliced data to ./src/cropped_data/set_train_<x_diff>_<y_diff>_<z_diff>/train_<i>.npy
# and                  ./src/cropped_data/set_test_<x_diff>_<y_diff>_<z_diff>/test_<i>.npy
# Returns paths to train files and test files.

def crop_images(train_filenames, test_filenames, x_diff, y_diff, z_diff, cluster_run, cluster_username='vli'):
    # Crop train images.
    if cluster_run:
        train_save_path = "/cluster/scratch/" + cluster_username + "/src/cropped_data/set_train_" + str(x_diff) + "_" + str(y_diff) + "_" + str(z_diff) + "/"
    else:
        train_save_path = "./src/cropped_data/set_train_" + str(x_diff) + "_" + str(y_diff) + "_" + str(z_diff) + "/"

    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)

    missing = False
    for i in range(1, len(train_filenames) + 1):
        if not os.path.isfile(train_save_path + "train_" + str(i) + ".npy"):
            missing = True
            break

    if missing:
        i = 1
        for f in train_filenames:
            img = image.load_img(f)
            img_array = np.array(img.dataobj[X_MIN + x_diff:X_MAX - x_diff, Y_MIN + y_diff:Y_MAX - y_diff, Z_MIN + z_diff:Z_MAX - z_diff, 0], dtype='f')
            np.save(train_save_path + "train_" + str(i), img_array)
            print("Saved cropped train image #" + str(i))
            i = i + 1

    # Crop test images.
    if cluster_run:
        test_save_path = "/cluster/scratch/" + cluster_username + "/src/cropped_data/set_test_" + str(x_diff) + "_" + str(y_diff) + "_" + str(z_diff) + "/"
    else:
        test_save_path = "./src/cropped_data/set_test_" + str(x_diff) + "_" + str(y_diff) + "_" + str(z_diff) + "/"

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    missing = False
    for i in range(1, len(test_filenames) + 1):
        if not os.path.isfile(test_save_path + "test_" + str(i) + ".npy"):
            missing = True
            break

    if missing:
        i = 1
        for f in test_filenames:
            img = image.load_img(f)
            img_array = np.array(img.dataobj[X_MIN + x_diff:X_MAX - x_diff, Y_MIN + y_diff:Y_MAX - y_diff, Z_MIN + z_diff:Z_MAX - z_diff, 0], dtype='f')
            np.save(test_save_path + "test_" + str(i), img_array)
            print("Saved cropped test image #" + str(i))
            i = i + 1

    if cluster_run:
        sys.stdout.flush()

    train_filenames = [train_save_path + "train_" + str(i) + ".npy" for i in range(1, len(train_filenames) + 1)]
    test_filenames = [test_save_path + "test_" + str(i) + ".npy" for i in range(1, len(test_filenames) + 1)]

    return (train_filenames, test_filenames)

