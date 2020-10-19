import numpy as np
import SimpleITK as sitk

path = '/data2/home/shuheng_zhang/test.png'

image = sitk.ReadImage(path, np.int8)
print(image)

