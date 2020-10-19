import pinkie
import torch

from pinkie_image_python import Frame

frame = Frame()
print(frame)

print(frame.voxel_to_world(torch.tensor([1.0,2.0,3.0])))


