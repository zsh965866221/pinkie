import pinkie
from pinkie_image_python import Frame

import numpy as np
import os
import SimpleITK as sitk
import torch 

def read_image(path, dtype=None):
  if dtype is None:
    dtype = sitk.sitkUnknown
  else:
    if dtype == np.uint8:
      dtype = sitk.sitkUInt8
    elif dtype == np.int32:
      dtype = sitk.sitkInt32
    elif dtype == np.float:
      dtype = sitk.sitkFloat32
    else:
      raise f"Unsupportted dtype: {dtype}"

  if os.path.isdir(path):
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.SetOutputPixelType(dtype)
    image_itk = reader.Execute()
  else:
    image_itk = sitk.ReadImage(path, dtype)

  tags = {}
  keys = image_itk.GetMetaDataKeys()
  for key in keys:
    tags[key] = image_itk.GetMetaData(key)

  tmp_size = image_itk.GetSize()
  tmp_spacing = image_itk.GetSpacing()
  tmp_origin = image_itk.GetOrigin()
  tmp_axes = image_itk.GetDirection()

  size = torch.ones(3)
  spacing = torch.ones(3)
  origin = torch.zeros(3)
  axes = torch.zeros((3, 3))

  L = len(tmp_size)
  for i in range(L):
    size[i] = tmp_size[i]
    spacing[i] = tmp_spacing[i]
    origin[i] = tmp_origin[i]
    for j in range(L):
      axes[i, j] = tmp_axes[j * L + i]

  frame = Frame()
  frame.set_origin(origin)
  frame.set_spacing(spacing)
  frame.set_axes(axes)
  
  return frame, tags


if __name__ == "__main__":
  path = '/data2/home/shuheng_zhang/test.png'
  image, tags = read_image(path)
  print(image)