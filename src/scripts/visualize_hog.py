import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from nilearn import image
import numpy as np

image_path = "../../data/set_train/train_100.nii"

img = image.load_img(image_path)
im = np.array(img.dataobj[80, :, :, 0], dtype='f')

fd, hog_image = hog(im, orientations=8, pixels_per_cell=(5, 5),
                    cells_per_block=(1, 1), visualise=True, feature_vector=True)

print(fd[np.nonzero(fd)].shape)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(im, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()