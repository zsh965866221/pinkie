# coding=utf-8

import pinkie

import os
import platform
import sys

def find_dll(name):
  if platform.system() == 'Linux':
    name = f'lib{name}.so'
  elif platform.system() == 'Darwin':
    name = f'lib{name}.dylib'
  elif platform.system() == 'Windows':
    name = f'{name}.dll'
  else:
    raise OSError('Unsupported os system!!!')

  out_path = None
  # find in the system path
  for dir_c in sys.path:
    if os.path.isdir(dir_c):
      path = os.path.join(dir_c, name)
      if os.path.exists(path):
        out_path = path
        break
    # end if
  # end for

  return out_path
  