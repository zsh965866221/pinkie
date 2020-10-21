#include "pinkie/transform/csrc/rotation.h"
#include "pinkie/transform/csrc/resample.h"

#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(pinkie_transform_python, m) {
  m.def("rotate", &pinkie::transform::rotate);
  m.def("resample_trilinear", &pinkie::transform::resample_trilinear);
}

