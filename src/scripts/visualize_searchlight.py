"""
Searchlight analysis of face vs house recognition
==================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the fMRI (see the generated figures).

"""

def get_targets():
    target_data = ""
    with open('../../data/targets.csv', 'r') as fo:
        target_data = fo.read()

    numbers = target_data.split('\n')
    numbers.pop(len(numbers) - 1)

    vectorized_int = np.vectorize(int)
    targets = vectorized_int(numbers)

    return targets

#########################################################################
import numpy as np
from nilearn import datasets
from nilearn.image import new_img_like, load_img

# Get the data.
NUM_TRAIN_DATA = 278
NUM_TEST_DATA = 138
train_filenames = ["../../data/set_train/train_" + str(i) + ".nii" for i in range(1, NUM_TRAIN_DATA + 1)]
test_filenames = ["../../data/set_test/test_" + str(i) + ".nii" for i in range(1, NUM_TEST_DATA + 1)]
y = get_targets()

#########################################################################
# Prepare masks
#
# - mask_img is the original mask
# - process_mask_img is a subset of mask_img, it contains the voxels that
#   should be processed (we only keep the slice z = 26 and the back of the
#   brain to speed up computation)
from nilearn.image import mean_img
from nilearn.masking import compute_background_mask

mean_img = mean_img(train_filenames, verbose=2)
mask_img = compute_background_mask(mean_img)

from nilearn.image import concat_imgs

train_4d = concat_imgs(train_filenames)
print(train_4d)

#########################################################################
# Searchlight computation
print("Searchlight computation.")

# Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session
from sklearn.cross_validation import KFold
cv = KFold(y.size, n_folds=10)

import nilearn.decoding
import pickle, os

# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = nilearn.decoding.SearchLight(
    mask_img,
    radius=5.6, n_jobs=-1,
    verbose=1, cv=cv)
searchlight.fit(train_4d, y)

# Pickle searchlight.
save_path = "../searchlight/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

pickle.dump(searchlight, open(save_path + 'searchlight.pkl', 'wb'))

#########################################################################
# F-scores computation
print("F-scores computation.")

from nilearn.input_data import NiftiMasker

# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_img,
                           standardize=True, memory='nilearn_cache',
                           memory_level=1)
fmri_masked = nifti_masker.fit_transform(train_4d)

from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(fmri_masked, y)
p_values = -np.log10(p_values)
p_values[p_values > 10] = 10
p_unmasked = nifti_masker.inverse_transform(p_values).get_data()

#########################################################################
# Visualization
print("Visualization.")

# Use the fmri mean image as a surrogate of anatomical data
from nilearn import image

from nilearn.plotting import plot_stat_map, plot_img, show
searchlight_img = new_img_like(mean_img, searchlight.scores_)

# Because scores are not a zero-center test statistics, we cannot use
# plot_stat_map
plot_img(searchlight_img, bg_img=mean_img,
         title="Searchlight", display_mode="z", cut_coords=[-90],
         vmin=.42, cmap='hot', threshold=.2, black_bg=True)

# F_score results
f_score_img = new_img_like(mean_img, p_unmasked)
plot_stat_map(f_score_img, mean_img,
              title="F-scores", display_mode="z",
              cut_coords=[-73],
              colorbar=False, output_file='../searchlight/z_display.png')

plot_stat_map(f_score_img, mean_img,
              title="F-scores", display_mode="y",
              cut_coords=[-86],
              colorbar=False, output_file='../searchlight/y_display.png')

plot_stat_map(f_score_img, mean_img,
              title="F-scores", display_mode="x",
              cut_coords=[-69],
              colorbar=False, output_file='../searchlight/x_display.png')

show()
