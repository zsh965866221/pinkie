#pragma once

#include <torch/torch.h>
#include <torch/extension.h>

#define BLOCK_SIZE 16

torch::Tensor mat_mul_cuda(
  torch::Tensor a,
  torch::Tensor b
);
