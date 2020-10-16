#ifndef PINKIE_TEST_TEST_CUDA_H
#define PINKIE_TEST_TEST_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define BLOCK_SIZE 16


inline void CudaAssert(
  cudaError_t code, 
  const char* file,
  int line,
  bool abort = true
) {
  if (code != cudaSuccess) {
    std::cout << file << ":" << line << "   "
              << "[cuda error]: " 
              << cudaGetErrorString(code) 
              << std::endl;
  }
}

#define CUDA_CHECK(ans) { CudaAssert((ans), __FILE__, __LINE__); }

void cuda_mat_mul(
  float* matA, 
  float* matB, 
  float* matC, 
  int a_rows, 
  int a_cols,
  int b_rows,
  int b_cols
);

void cpu_mat_mul(
  float* matA, 
  float* matB, 
  float* matC, 
  int a_rows, 
  int a_cols,
  int b_rows,
  int b_cols
);


#endif // PINKIE_TEST_TEST_CUDA_H