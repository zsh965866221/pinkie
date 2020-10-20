import pinkie
import torch
from pinkie_image_python import Frame, Image

import numpy as np
import os
import SimpleITK as sitk

from pinkie.image.python.image import itk_to_image, image_to_itk


def read_image(path, dtype=None):
  if dtype is None:
    dtype = sitk.sitkUnknown
  else:
    if dtype == np.uint8 or dtype == torch.uint8:
      dtype = sitk.sitkUInt8
    elif dtype == np.int32 or dtype == torch.int32:
      dtype = sitk.sitkInt32
    elif dtype == np.float or dtype == torch.float:
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

  image = itk_to_image(image_itk)
  
  return image, tags


def write_image(image, path, compression=True):
  itk = image_to_itk(image)
  sitk.WriteImage(itk, path, compression)




if __name__ == "__main__":
  path = '/data2/home/shuheng_zhang/projects/parser18+/merge/AI01_0001_01_L_diast_best_0.50mm.mhd'

  image, _ = read_image(path)
  image.to_(torch.device('cuda'))
  print(image)

  write_image(image, '/data2/home/shuheng_zhang/out.mhd')
