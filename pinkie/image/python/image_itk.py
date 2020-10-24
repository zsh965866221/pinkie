# coding=utf-8

import numpy as np
import os
import SimpleITK as sitk

from pinkie.image.python.frame import Frame
from pinkie.image.python.image import Image


def image_to_itk(image: Image):
  is_2d = image.is_2d()
  L = 2 if is_2d is True else 3

  axes = image.frame().axes()
  spacing = image.frame().spacing()
  origin = image.frame().origin()

  direction_out = []
  spacing_out = []
  origin_out = []
  for i in range(L):
    origin_out.append(float(origin[i]))
    spacing_out.append(float(spacing[i]))
    for j in range(L):
      direction_out.append(float(axes[i, j]))
  
  data = image.to_numpy()
  if is_2d is True:
    data = data.transpose(2, 1, 0)
  itk = sitk.GetImageFromArray(
    data, 
    isVector=True if is_2d is True else False
  )
  itk.SetDirection(direction_out)
  itk.SetSpacing(spacing_out)
  itk.SetOrigin(origin_out)

  return itk


def itk_to_image(itk) -> Image:
  tmp_size = itk.GetSize()
  tmp_spacing = itk.GetSpacing()
  tmp_origin = itk.GetOrigin()
  tmp_axes = itk.GetDirection()

  tmp_data = sitk.GetArrayFromImage(itk)

  size = np.ones(3)
  spacing = np.ones(3)
  origin = np.zeros(3)
  axes = np.zeros((3, 3))

  L = len(tmp_size)
  assert(L == 2 or L == 3)
  is_2d = True if L == 2 else False
  for i in range(L):
    size[i] = tmp_size[i]
    spacing[i] = tmp_spacing[i]
    origin[i] = tmp_origin[i]
    for j in range(L):
      axes[i, j] = tmp_axes[i * L + j]
  
  if L == 2:
    axes[2, 2] = 1.0
  
  if len(tmp_data.shape) == 2:
    tmp_data = tmp_data.reshape(tmp_data.shape[0], tmp_data.shape[1], -1)
  
  size[0] = tmp_data.shape[0]
  size[1] = tmp_data.shape[1]
  size[2] = tmp_data.shape[2]

  frame = Frame()
  frame.set_origin(origin)
  frame.set_spacing(spacing)
  frame.set_axes(axes)

  if is_2d is True:
    tmp_data = tmp_data.transpose(2, 1, 0)
  image = Image.from_numpy(tmp_data, is_2d)
  image.set_frame(frame)

  return image
