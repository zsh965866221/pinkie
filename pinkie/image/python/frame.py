# coding=utf-8

import ctypes
import numpy as np

from pinkie.utils.python.ctypes import find_dll

def load_lib():
  lib_path = find_dll('pinkie_pyframe')
  if lib_path is None:
    raise 'lib not exists'
  lib = ctypes.cdll.LoadLibrary(lib_path)

  lib.frame_new.argtypes = []
  lib.frame_new.restype = ctypes.c_void_p

  lib.frame_clone.argtypes = [ctypes.c_void_p]
  lib.frame_clone.restype = ctypes.c_void_p

  lib.frame_delete.argtypes = [ctypes.c_void_p]
  lib.frame_delete.restype = None

  lib.frame_origin.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_origin.restype = None

  lib.frame_spacing.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_spacing.restype = None

  lib.frame_axes.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=2,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_axes.restype = None

  lib.frame_axis.argtypes = [
    ctypes.c_void_p, 
    ctypes.c_uint,
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_axis.restype = None

  lib.frame_set_origin.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_set_origin.restype = None

  lib.frame_set_spacing.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_set_spacing.restype = None

  lib.frame_set_axes.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=2,
      flags='F_CONTIGUOUS'
    ),
  ]
  lib.frame_set_axes.restype = None

  lib.frame_set_axis.argtypes = [
    ctypes.c_void_p, 
    np.ctypeslib.ndpointer(
      dtype=ctypes.c_float,
      ndim=1,
      flags='F_CONTIGUOUS'
    ),
    ctypes.c_uint,
  ]
  lib.frame_set_axis.restype = None

  lib.frame_world_to_voxel.argtypes = [
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
  lib.frame_world_to_voxel.restype = None

  lib.frame_voxel_to_world.argtypes = [
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
  lib.frame_voxel_to_world.restype = None
  
  return lib


lib = load_lib()

class Frame:
  def __init__(self, frame=None):
    if frame is None:
      self.ptr = lib.frame_new()
    else:
      assert isinstance(frame, Frame)
      self.ptr = lib.frame_clone(frame.ptr)
  
  def copy(self):
    return Frame(self)

  def __del__(self):
    lib.frame_delete(self.ptr)
  
  def origin(self):
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.frame_origin(self.ptr, ret)
    return ret

  def spacing(self):
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.frame_spacing(self.ptr, ret)
    return ret
  
  def axes(self):
    ret = np.zeros((3, 3), order='F', dtype=np.float32)
    lib.frame_axes(self.ptr, ret)
    return ret
  
  def axis(self, index):
    assert index < 3
    assert index >= 0

    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.frame_axis(self.ptr, ctypes.c_uint(index), ret)
    return ret
  
  def set_origin(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.frame_set_origin(self.ptr, data)
  
  def set_spacing(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.frame_set_spacing(self.ptr, data)
  
  def set_axes(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.frame_set_axes(self.ptr, data)
  
  def set_axis(self, data, index):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    lib.frame_set_axis(self.ptr, data, ctypes.c_uint(index))
  
  def world_to_voxel(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.frame_world_to_voxel(self.ptr, data, ret)
    return ret

  def voxel_to_world(self, data):
    data = np.array(data, order='F', dtype=np.float32, copy=False)
    ret = np.zeros((3), order='F', dtype=np.float32)
    lib.frame_voxel_to_world(self.ptr, data, ret)
    return ret

  def __str__(self):
    return \
      f"origin: {self.origin()}\n"\
      f"spacing: {self.spacing()}\n"\
      f"axis_x: {self.axis(0)}\n"\
      f"axis_y: {self.axis(1)}\n"\
      f"axis_z: {self.axis(2)}"
  

if __name__ == '__main__':
  frame = Frame()
  frame.set_axes([
    [1,2,3],
    [4,5,6],
    [7,8,9]
  ])
  print(frame)