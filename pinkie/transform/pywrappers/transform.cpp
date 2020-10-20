#include "pinkie/transform/csrc/rotation.h"
#include "pinkie/transform/csrc/resample.h"

#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(pinkie_transform_python, m) {
  m.def("rotation_matrix", &pinkie::transform::rotation_matrix);
  m.def("resample_trilinear", &pinkie::transform::resample_trilinear);
}

