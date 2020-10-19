import pinkie

import numpy as np
import torch

from pinkie_transform_python import rotation_matrix

x = torch.tensor([1.0, 0.0, 0.0])
axis = torch.tensor([0.0, 1.0, 0.0])
matrix = rotation_matrix(axis, 30.0 / 180.0 * np.pi)
print(matrix)
print(torch.matmul(matrix, x))
