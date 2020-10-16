#include "pinkie/test/csrc/test_python_interface.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#define BLOCK_SIZE 16

template<typename scalar_t>
__global__ 
void mat_mul_cuda_kernel(
  scalar_t* matA, 
  scalar_t* matB, 
  scalar_t* matC, 
  int64_t a_rows, 
  int64_t a_cols,
  int64_t b_rows,
  int64_t b_cols
) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  if (row >= a_rows || col >= b_cols) {
    return;
  }

  for (int i = 0; i < a_cols; i++) {
    matC[row * b_cols + col] += matA[row * a_cols + i] * matB[i * b_cols + col];
  }
}


torch::Tensor mat_mul_cuda(
  torch::Tensor a,
  torch::Tensor b
) {
  auto options =
  torch::TensorOptions()
    .dtype(a.scalar_type())
    .device(a.device().type(), a.device().index());
  auto ret = torch::zeros({a.size(0), b.size(1)}, options);
  dim3 dim_grid(a.size(0) / BLOCK_SIZE + 1, b.size(1) / BLOCK_SIZE + 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
  AT_DISPATCH_ALL_TYPES(a.type(), "mat mul cuda torch", ([&]{
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(false, a.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);
    mat_mul_cuda_kernel<scalar_t><<<dim_grid, dim_block>>>(
      a.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(),
      ret.data_ptr<scalar_t>(),
      a.size(0),
      a.size(1),
      b.size(0),
      b.size(1)
    );
  }));
  std::cout << "ret: " << ret.device() << "\n" << ret << std::endl;
  std::cout << "a: " << a.device() << "\n" << a << std::endl;
  std::cout << "b: " << b.device() << "\n" << b << std::endl;
  return ret;
}