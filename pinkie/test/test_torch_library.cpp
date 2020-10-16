#include <torch/script.h>
#include <torch/torch.h>

torch::Tensor tensor_add(const torch::Tensor& a, const torch::Tensor& b, const int64_t c) {
  return a + b + c;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("tensor_add", &tensor_add);
}