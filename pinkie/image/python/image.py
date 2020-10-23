# coding=utf-8

import ctypes
import numpy as np

from pinkie.image.python.frame import Frame
from pinkie.image.python.pixel_type import dtype_dict, dtype_list
from pinkie.utils.python.ctypes import find_dll

def load_lib():
  lib_path = find_dll('pinkie_pyimage')
  if lib_path is None:
    raise 'lib not exists'
  lib = ctypes.cdll.LoadLibrary(lib_path)

  lib.image_new.argtypes = [ctypes.c_int, ctypes.c_bool]
  lib.image_new.restype = ctypes.c_void_p

  lib.image_clone.argtypes = [ctypes.c_void_p, ctypes.c_bool]
  lib.image_clone.restype = ctypes.c_void_p

  lib.image_new_owned.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_bool
  ]
  lib.image_new_owned.restype = ctypes.c_void_p

  lib.image_delete.argtypes = [ctypes.c_void_p]
  lib.image_delete.restype = None

  lib.image_size.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_int,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_size.restype = None

  lib.image_frame.argtypes = [ctypes.c_void_p]
  lib.image_frame.restype = ctypes.c_void_p

  lib.image_set_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  lib.image_set_frame.restype = None

  lib.image_data.argtypes = [ctypes.c_void_p]
  lib.image_data.restype = ctypes.c_void_p

  lib.image_size.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      ndim=3,
      flags='C_CONTIGUOUS'
    ),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.c_int, ctypes.c_bool, ctypes.c_bool
  ]
  lib.image_size.restype = None

  lib.image_set_zero.argtypes = [ctypes.c_void_p]
  lib.image_set_zero.restype = None

  lib.image_is_2d.argtypes = [ctypes.c_void_p]
  lib.image_is_2d.restype = ctypes.c_bool

  lib.image_set_2d.argtypes = [ctypes.c_void_p, ctypes.c_bool]
  lib.image_set_2d.restype = None

  lib.image_dtype.argtypes = [ctypes.c_void_p]
  lib.image_dtype.restype = ctypes.c_bool

  lib.image_cast.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.image_cast.restype = ctypes.c_void_p

  lib.image_cast_.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.image_cast_.restype = None
  
  return lib


lib = load_lib()

class Image:
  def __init__(self, dtype=np.float32, is_2d=False, ptr=None):
    if ptr is None:
      self.ptr = lib.image_new(
        dtype_list[dtype], 
        is_2d
      )
    else:
      self.ptr = ptr
  
  def copy(self, copy=True):
    return Image(lib.image_clone(self.ptr, copy))

  @staticmethod
  def new(height, width, depth, dtype=np.float32, is_2d=False):
    return Image(
      lib.image_new_owned(
        height, width, depth, 
        dtype_dict[dtype], is_2d
      )
    )
  
  def __del__(self):
    lib.image_delete(self.ptr)
  
  def size(self):
    ret = np.zeros((3), order='F', dtype=np.int)
    lib.image_size(self.ptr, ret)
    return ret
  
  def frame(self):
    return Frame(ptr=lib.image_frame(self.ptr))
  
  def set_frame(self, frame: Frame):
    lib.image_set_frame(self.ptr, frame.ptr)

  def to_numpy(self):
    size = self.size()
    ret = np.zeros(
      (size[0], size[1], size[2]),
      dtype=self.dtype(), order='C'
    )
  