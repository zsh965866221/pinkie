#include "pinkie/test/test_python_interface.h"

#include <torch/torch.h>
#include <torch/extension.h>

namespace py = pybind11;

float add(float a, float b) {
  return a + b;
}

torch::Tensor tensor_mul(
  const torch::Tensor& a, 
  const torch::Tensor& b, 
  const float c
) {
  return torch::mm(a, b) + c;
}

PYBIND11_MODULE(pinkie_python_test, module_pinkie) {
  auto module_test = module_pinkie.def_submodule("test_lib");
  module_test.def("add", &add);
  module_test.def("mul", [](float a, float b) -> float {
    return a * b;
  });
  module_test.def(
    "tensor_mul", 
    &tensor_mul,
    "tensor mul"
  );
  module_test.def(
    "mat_mul_cuda",
    &mat_mul_cuda,
    "mat_mul_cuda"
  );
}