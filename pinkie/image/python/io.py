# coding=utf-8

import numpy as np
import os
import SimpleITK as sitk

from pinkie.image.python.image_itk import itk_to_image, image_to_itk


def read_image(path, dtype=None):
  if dtype is None:
    dtype = sitk.sitkUnknown
  else:
    if dtype == np.int8:
      dtype = sitk.sitkInt8
    if dtype == np.uint8:
      dtype = sitk.sitkUInt8
    elif dtype == np.int32:
      dtype = sitk.sitkInt32
    elif dtype == np.uint32:
      dtype = sitk.sitkUInt32
    elif dtype == np.int64:
      dtype = sitk.sitkInt64
    elif dtype == np.uint64:
      dtype = sitk.sitkUInt64
    elif dtype == np.float32:
      dtype = sitk.sitkFloat32
    elif dtype == np.float64:
      dtype = sitk.sitkFloat64
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
  path = r'E:\work\git\test.jpg'
  path_out = r'E:\work\git\test_out.png'
  image, _ = read_image(path)
  print(image)
  data = image.to_numpy()
  image.set_data(data[:1,:,:])
  write_image(image, path_out)


