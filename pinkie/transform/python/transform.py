# coding=utf-8

import ctypes
import numpy as np
import pinkie

from pinkie.image.python.frame import Frame
from pinkie.image.python.image import Image
from pinkie.utils.python.ctypes import find_dll

def load_lib():
  lib_path = find_dll('pinkie_pytransform')
  if lib_path is None:
    raise 'lib not exists'
  lib = ctypes.cdll.LoadLibrary(lib_path)

  lib.transform_rotate.argtypes = [
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
    ctypes.c_float,
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=2,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.transform_rotate.restype = None

  lib.transform_resample_trilinear.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_int32,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
    ctypes.c_float
  ]
  lib.transform_resample_trilinear.restype = ctypes.c_void_p

  
  return lib


lib = load_lib()


def rotate(axis, theta):
  axis = np.array(
    axis, dtype=np.float32, copy=False, order='F'
  )
  ret = np.zeros(
    (3, 3), dtype=np.float32, order='F'
  )
  lib.transform_rotate(axis, theta, ret)
  return ret

def resample_trilinear(
  image, frame, size, padding_value=0.0
):
  padding_value = float(padding_value)
  size = np.array(
    size, dtype=np.int32, copy=False, order='F'
  )
  return Image(ptr=lib.transform_resample_trilinear(
    image.ptr, frame.ptr, size, padding_value
  ))


if __name__ == '__main__':
  from pinkie.image.python.io import read_image, write_image

  path = r'E:\work\git\test.jpg'
  path_out = r'E:\work\git\test_out.jpg'

  image, _ = read_image(path)
  size = image.size()
  spacing = image.frame().spacing()
  frame = image.frame().copy()

  rotate_axis = np.array([0.0, 0.0, 1.0])
  rotate_matrix = rotate(rotate_axis, 30.0 / 180.0 * np.pi)
  origin = frame.origin()
  spacing = spacing * 3 / 2
  for i in range(3):
    axis = np.dot(rotate_matrix, image.frame().axis(i))
    frame.set_axis(axis, i)
  frame.set_origin(origin)
  frame.set_spacing(spacing)

  print(image)
  print(frame)

  image_resampled = resample_trilinear(image, frame, size)
  write_image(image_resampled, path_out)


