# coding=utf-8

import numpy as np
import SimpleITK as sitk

dtype_list = [
  np.int8,
  np.uint8,
  np.int16,
  np.uint16,
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

dtype_itk = {
  sitk.sitkInt8:      np.int8,
  sitk.sitkUInt8:     np.uint8,
  sitk.sitkInt16:     np.int16,
  sitk.sitkUInt16:    np.uint16,
  sitk.sitkInt32:     np.int32,
  sitk.sitkUInt32:    np.uint32,
  sitk.sitkInt64:     np.int64,
  sitk.sitkUInt64:    np.uint64,
  sitk.sitkFloat32:   np.float32,
  sitk.sitkFloat64:   np.float64
}