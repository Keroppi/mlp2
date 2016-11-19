import numpy as np

class VoxelGrid:
    def __init__(self, img_array, targets, feature_function, params, size=3):
        self.voxelSize = size

        size_x, size_y, size_z = img_array.shape

        size_x /= size
        size_y /= size
        size_z /= size

        self.grid = np.array([])

        for x in range(0, size_x):
            for y in range(0, size_y):
                for z in range(0, size_z):
                    feature_vector = feature_function(X=img_array[x * size: (x + 1) * size, y * size: (y + 1) * size, z * size: (z + 1) * size], \
                                                      y=targets, **params)
                    if (x == 0 and y == 0 and z == 0):
                        self.grid = np.zeros((size_x, size_y, size_z, feature_vector.shape[0]))
                    self.grid[x, y, z] = feature_vector

    def __str__(self):

        size_x, size_y, size_z = self.grid.shape
        result = "Voxel Grid:\n\t- X dimension: " + str(size_x) \
                 + "\n\t- Y dimension: " + str(size_y) \
                 + "\n\t- Z dimension: " + str(size_z) \
                 + "\n\t[\n"
        for x in range(0, size_x):
            result += "\t\t[\n"
            for y in range(0, size_y):
                result += "\t\t\t["
                for z in range(0, size_z):
                    result += str(self.grid[x][y][z])
                    if z < size_z - 1:
                        result += " "
                result += "]\n"
            result += "\t\t]\n"
        result += "\t]"

        return result

    def get_feature_vector(self, x, y, z):
        return self.grid[x, y, z]
