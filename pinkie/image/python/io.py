# coding=utf-8

import numpy as np
import os
import SimpleITK as sitk

from pinkie.image.python.image_itk import itk_to_image, image_to_itk
from pinkie.image.python.pixel_type import dtype_itk


def read_image(path, dtype=None):
  if dtype is None:
    dtype = sitk.sitkUnknown
  else:
    assert dtype in dtype_itk
    dtype = dtype_itk[dtype]

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
  
  return image, tags, image_itk


def write_image(image, path, compression=True):
  itk = image_to_itk(image)
  sitk.WriteImage(itk, path, compression)



if __name__ == "__main__":
  path = r'E:\work\git\test.jpg'
  path_out = r'E:\work\git\test_out.png'
  image, _, _ = read_image(path)
  print(image)
  data = image.to_numpy()
  image.set_data(data[:1,:,:])
  write_image(image, path_out)


