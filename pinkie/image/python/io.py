
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
  return image_itk, tags


if __name__ == "__main__":
  path = '/data2/home/shuheng_zhang/a'
  image, tags = read_image(path)
  print(image)