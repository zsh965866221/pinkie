import pinkie
import torch

from pinkie_image_python import Frame

frame = Frame()
frame.set_axis(torch.tensor([1.0,1.0,1.0]), 0)
frame.to_(torch.device("cuda"))
print(frame)
