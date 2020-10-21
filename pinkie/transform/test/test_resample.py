import pinkie
import torch

import numpy as np
import time

from pinkie.image.python.io import read_image, write_image
from pinkie_transform_python import rotate, resample_trilinear


path = '/data2/home/shuheng_zhang/test.png'
path_out = '/data2/home/shuheng_zhang/test_out.jpeg'

image, _ = read_image(path)
image.cast_(torch.float)
size = image.size()
spacing = image.spacing()
frame = image.frame().clone()

rotate_axis = torch.tensor([0.0, 0.0, 1.0])
rotate_matrix = rotate(rotate_axis, 90.0 / 180.0 * np.pi)
origin = frame.origin() + size.float() / 2.0 * frame.spacing()
spacing = spacing * 3 / 2
for i in range(3):
  axis = torch.matmul(rotate_matrix, image.axis(i))
  frame.set_axis(axis, i)
  origin -= (axis * size[i] / 2 * spacing[i])
frame.set_origin(origin)
frame.set_spacing(spacing)

print(image)
print(frame)

time_start = time.time()
image_resampled = resample_trilinear(image, frame, size, 0.0)
time_end = time.time()

image_resampled.cast_(torch.uint8)
write_image(image_resampled, path_out)

print(time_end - time_start)

