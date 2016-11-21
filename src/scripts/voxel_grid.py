import numpy as np

class VoxelGrid:
    def __init__(self, img_array, targets, feature_function, params, size=3):
        self.voxelSize = size

        size_x, size_y, size_z = img_array.shape

        size_x /= size
        size_y /= size
        size_z /= size

        self.grid = np.array([])

        for x in range(0, size):
            for y in range(0, size):
                for z in range(0, size):
                    feature_vector = feature_function(X=img_array[x * size_x: (x + 1) * size_x, \
                                                                  y * size_y: (y + 1) * size_y, \
                                                                  z * size_z: (z + 1) * size_z], \
                                                      y=targets, **params)
                    if (x == 0 and y == 0 and z == 0):
                        self.grid = np.zeros((size, size, size, feature_vector.shape[0]))
                    self.grid[x, y, z] = feature_vector

    def get_feature_vector(self, x, y, z):
        return self.grid[x, y, z]
