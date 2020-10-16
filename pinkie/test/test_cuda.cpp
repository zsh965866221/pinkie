
#include "pinkie/test/mat_mul.h"

#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>


int main() {
  cudaSetDevice(1);
  cudaDeviceProp cuda_prop;
  cudaGetDeviceProperties(&cuda_prop, 0);
  std::cout << cuda_prop.name 
            << "; global mem: " << cuda_prop.totalGlobalMem
            << "; threads per block: " << cuda_prop.maxThreadsPerBlock
            << "; max thread dim: " << cuda_prop.maxThreadsDim[0] << std::endl;

  const int a_rows = BLOCK_SIZE * 100;
  const int a_cols = BLOCK_SIZE * 100;
  const int b_rows = a_cols;
  const int b_cols = BLOCK_SIZE * 20;

  float* matA = new float[a_rows * a_cols];
  float* matB = new float[b_rows * b_cols];
  float* matC = new float[a_rows * b_cols];

  // set rand
  srand((unsigned)time(NULL));
  for (int i = 0; i < a_rows * a_cols; i++) {
    matA[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < b_rows * b_cols; i++) {
    matB[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  float* matA_cuda;
  float* matB_cuda;
  float* matC_cuda;

  auto time_cuda_base = std::chrono::steady_clock::now();
  
  CUDA_CHECK(cudaMallocManaged((void**)&matA_cuda, a_rows * a_cols * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged((void**)&matB_cuda, b_rows * b_cols * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged((void**)&matC_cuda, a_rows * b_cols * sizeof(float)));

  CUDA_CHECK(cudaMemcpy((void*)matA_cuda, (void*)matA, a_rows * a_cols * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((void*)matB_cuda, (void*)matB, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice));

  cuda_mat_mul(
    matA_cuda,
    matB_cuda,
    matC_cuda,
    a_rows, 
    a_cols,
    b_rows,
    b_cols
  );
  CUDA_CHECK(cudaPeekAtLastError());

  cudaDeviceSynchronize();

  auto time_cuda_end = std::chrono::steady_clock::now();
  auto time_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(time_cuda_end - time_cuda_base);
  std::cout << "cuda: " << time_cuda.count() << std::endl;

  auto time_cpu_base = std::chrono::steady_clock::now();
  cpu_mat_mul(
    matA,
    matB,
    matC,
    a_rows,
    a_cols,
    b_rows,
    b_cols
  );
  auto time_cpu_end = std::chrono::steady_clock::now();
  auto time_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(time_cpu_end - time_cpu_base);
  std::cout << "cpu: " << time_cpu.count() << std::endl;

  float difference = 0.0f;
  for (int i = 0; i < a_rows * b_cols; i++) {
    difference += fabs(matC[i] - matC_cuda[i]);
  }

  std::cout << "difference: " << difference << std::endl;

  cudaFree(matC_cuda);
  cudaFree(matA_cuda);
  cudaFree(matB_cuda);

  delete[] matA;
  delete[] matB;
  delete[] matC;
}