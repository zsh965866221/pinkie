import pinkie
import torch
from pinkie_image_python import Frame, Image

import numpy as np
import os
import SimpleITK as sitk


def image_to_itk(image: Image):
  is_2d = image.is_2d()
  L = 2 if is_2d is True else 3

  axes = image.axes().cpu().numpy()
  spacing = image.spacing().cpu().numpy()
  origin = image.origin().cpu().numpy()

  direction_out = []
  spacing_out = []
  origin_out = []
  for i in range(L):
    origin_out.append(float(origin[i]))
    spacing_out.append(float(spacing[i]))
    for j in range(L):
      direction_out.append(float(axes[j, i]))
  
  data = image.data().cpu().numpy()
  if is_2d is not True:
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

  size = torch.ones(3)
  spacing = torch.ones(3)
  origin = torch.zeros(3)
  axes = torch.zeros((3, 3))

  L = len(tmp_size)
  assert(L == 2 or L == 3)
  is_2d = True if L == 2 else False
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

  image = Image(is_2d=is_2d)
  image.set_frame(frame)
  if is_2d is False:
    tmp_data = tmp_data.transpose(2, 1, 0)
  image.set_data(torch.from_numpy(tmp_data))

  return image
