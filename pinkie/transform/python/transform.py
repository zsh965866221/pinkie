# coding=utf-8

import ctypes
import numpy as np

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


if __name__ == '__main__':
  axis = np.array([0, 0, 1])
  theta = 90.0 / 180.0 * np.pi
  print(rotate(axis, theta))