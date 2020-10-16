#ifndef PINKIE_TEST_CSRC_TEST_PYTHON_INTERFACE_H
#define PINKIE_TEST_CSRC_TEST_PYTHON_INTERFACE_H

#include <torch/torch.h>
#include <torch/extension.h>

#define BLOCK_SIZE 16

torch::Tensor mat_mul_cuda(
  torch::Tensor a,
  torch::Tensor b
);


#endif // PINKIE_TEST_CSRC_TEST_PYTHON_INTERFACE_H