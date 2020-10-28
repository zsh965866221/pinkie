#ifndef PINKIE_UTILS_CSRC_CUDA_HEADER_H
#define PINKIE_UTILS_CSRC_CUDA_HEADER_H

#include <assert.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 4

inline void CudaAssert(
  cudaError_t code, 
  const char* file,
  int line,
  bool abort = true
) {
  if (code != cudaSuccess) {
    std::cout 
      << file << ":" << line << "   "
      << "[cuda error]: " 
      << cudaGetErrorString(code) 
      << std::endl;
    if (abort == true) {
      assert(false);
    }
  }
}

#define CUDA_CHECK(ans) { CudaAssert((ans), __FILE__, __LINE__); }

const static dim3 DIM_BLOCK(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
dim3 generate_grid(int height, int width, int depth) {
  return dim3(
    static_cast<int>(ceil(static_cast<float>(height) / static_cast<float>(DIM_BLOCK.x))),
    static_cast<int>(ceil(static_cast<float>(width) / static_cast<float>(DIM_BLOCK.z))),
    static_cast<int>(ceil(static_cast<float>(depth) / static_cast<float>(DIM_BLOCK.z)))
  );
}

#endif // PINKIE_UTILS_CSRC_CUDA_HEADER_H