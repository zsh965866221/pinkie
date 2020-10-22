import os
import sys

# add lib
dir_base = os.path.split(__file__)[-2]
dir_clib = os.path.abspath(os.path.join(dir_base, 'lib', 'Release'))
if dir_clib not in sys.path:
  sys.path.insert(0, dir_clib)

