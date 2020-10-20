import pinkie

import torch

from pinkie_image_python import Frame, Image


image = Image(10, 10, 10)
image.to_(torch.device("cuda"))
print(image)
image.set_data(torch.randn(100, 100, 100))
print(image)
