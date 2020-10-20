import pinkie

import numpy as np
import torch

from pinkie_image_python import Frame
from pinkie_transform_python import rotation_matrix

frame = Frame()
matrix = rotation_matrix(torch.tensor([0.0, 1.0, 0.0]), 30.0 / 180.0 * np.pi)
axes = torch.matmul(matrix, frame.axes())
frame.set_axes(axes)
print(frame.world_to_voxel(torch.tensor([1.0,2.0,3.0])))


