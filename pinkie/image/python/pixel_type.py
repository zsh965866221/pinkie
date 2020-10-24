# coding=utf-8

import numpy as np

dtype_list = [
  np.int8,
  np.uint8,
  np.int32,
  np.uint32,
  np.int64,
  np.uint64,
  np.float32,
  np.float64
]

dtype_dict = {}
for index, dtype in enumerate(dtype_list):
  dtype_dict[dtype] = index
