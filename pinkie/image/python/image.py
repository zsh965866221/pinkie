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

  lib.image_data.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(
      ndim=3,
      flags='C_CONTIGUOUS'
    ),
  ]
  lib.image_data.restype = None

  lib.image_set_data.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      ndim=3,
      flags='C_CONTIGUOUS'
    ),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, 
    ctypes.c_int, ctypes.c_bool, ctypes.c_bool
  ]
  lib.image_set_data.restype = None

  lib.image_set_zero.argtypes = [ctypes.c_void_p]
  lib.image_set_zero.restype = None

  lib.image_is_2d.argtypes = [ctypes.c_void_p]
  lib.image_is_2d.restype = ctypes.c_bool

  lib.image_set_2d.argtypes = [ctypes.c_void_p, ctypes.c_bool]
  lib.image_set_2d.restype = None

  lib.image_dtype.argtypes = [ctypes.c_void_p]
  lib.image_dtype.restype = ctypes.c_int

  lib.image_cast.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.image_cast.restype = ctypes.c_void_p

  lib.image_cast_.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.image_cast_.restype = None

  lib.image_origin.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_origin.restype = None

  lib.image_spacing.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_spacing.restype = None

  lib.image_axes.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=2,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_axes.restype = None

  lib.image_axis.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_uint,
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_axis.restype = None

  lib.image_set_origin.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_set_origin.restype = None

  lib.image_set_spacing.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_set_spacing.restype = None

  lib.image_set_axes.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=2,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_set_axes.restype = None

  lib.image_set_axis.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
    ctypes.c_uint,
  ]
  lib.image_set_axis.restype = None

  lib.image_world_to_voxel.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_world_to_voxel.restype = None

  lib.image_voxel_to_world.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.image_voxel_to_world.restype = None
  
  return lib


lib = load_lib()

class Image:
  def __init__(self, dtype=np.float32, is_2d=False, ptr=None):
    if ptr is None:
      self.ptr = lib.image_new(
        dtype_dict[dtype], 
        is_2d
      )
    else:
      self.ptr = ptr
  
  def copy(self, copy=True):
    return Image(ptr=lib.image_clone(self.ptr, copy))

  @staticmethod
  def new(height, width, depth, dtype=np.float32, is_2d=False):
    return Image(
      lib.image_new_owned(
        height, width, depth, 
        dtype_dict[dtype], is_2d
      )
    )
  
  def __del__(self):
    if self.ptr is not None:
      lib.image_delete(self.ptr)
  
  def size(self):
    ret = np.zeros((3), order='F', dtype=np.int)
    lib.image_size(self.ptr, ret)
    return ret
  
  def frame(self):
    return Frame(ptr=lib.image_frame(self.ptr), owned=False)
  
  def set_frame(self, frame: Frame):
    lib.image_set_frame(self.ptr, frame.ptr)
  
  def dtype(self):
    return dtype_list[
      lib.image_dtype(self.ptr)
    ]
  
  def to_numpy(self):
    size = self.size()
    ret = np.zeros(
      (size[2], size[1], size[0]),
      dtype=self.dtype(), order='C'
    )
    lib.image_data(self.ptr, ret)
    return ret

  def set_data(self, data):
    data = np.array(data, order='C', copy=False)
    lib.image_set_data(
      self.ptr, 
      data,
      data.shape[2],
      data.shape[1],
      data.shape[0],
      dtype_dict[data.dtype.type],
      self.is_2d(),
      True
    )
  
  @staticmethod
  def from_numpy(data, is_2d=False):
    data = np.array(data, order='C', copy=False)
    image = Image(is_2d=is_2d)
    lib.image_set_data(
      image.ptr, 
      data,
      data.shape[2],
      data.shape[1],
      data.shape[0],
      dtype_dict[data.dtype.type],
      is_2d,
      True
    )
    return image

  def set_zero(self):
    lib.image_set_zero(self.ptr)
  
  def is_2d(self):
    return lib.image_is_2d(self.ptr)
  
  def set_2d(self, p):
    lib.image_set_2d(self.ptr, p)

  def cast(self, dtype):
    return Image(
      ptr=lib.image_cast(self.ptr, dtype_dict[dtype])
    )
  
  def cast_(self, dtype):
    lib.image_cast_(self.ptr, dtype_dict[dtype])
  
  def origin(self):
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.image_origin(self.ptr, ret)
    return ret

  def spacing(self):
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.image_spacing(self.ptr, ret)
    return ret
  
  def axes(self):
    ret = np.zeros((3, 3), order='F', dtype=np.float32)
    lib.image_axes(self.ptr, ret)
    return ret
  
  def axis(self, index):
    assert index < 3
    assert index >= 0

    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.image_axis(self.ptr, ctypes.c_uint(index), ret)
    return ret
  
  def set_origin(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.image_set_origin(self.ptr, data)
  
  def set_spacing(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.image_set_spacing(self.ptr, data)
  
  def set_axes(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.image_set_axes(self.ptr, data)
  
  def set_axis(self, data, index):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.image_set_axis(self.ptr, data, ctypes.c_uint(index))
  
  def world_to_voxel(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.image_world_to_voxel(self.ptr, data, ret)
    return ret

  def voxel_to_world(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.image_voxel_to_world(self.ptr, data, ret)
    return ret

  def __repr__(self):
    return \
      f"size: {self.size()}\n"\
      f"2d: {self.is_2d()}\n"\
      f"dtype: {self.dtype()}\n"\
      f"{self.frame()}\n"


if __name__ == '__main__':
  image = Image(is_2d=True)
  print(image)
  image.cast_(dtype=np.int32)
  print(image)
  