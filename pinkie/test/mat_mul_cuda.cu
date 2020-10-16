#include "pinkie/test/mat_mul.h"

__global__ 
void kernel_mat_mul(
    float* matA, 
    float* matB, 
    float* matC, 
    int a_rows, 
    int a_cols,
    int b_rows,
    int b_cols) {
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  if (row >= a_rows || col >= b_cols) {
    return;
  }

  float value = 0.0f;
  for (int i = 0; i < a_cols; i++) {
    value += matA[row * a_cols + i] * matB[i * b_cols + col];
  }
  matC[row * b_cols + col] = value;
}

void cuda_mat_mul(
    float* matA, 
    float* matB, 
    float* matC, 
    int a_rows, 
    int a_cols,
    int b_rows,
    int b_cols) {
  dim3 dim_grid(a_rows / BLOCK_SIZE, b_cols / BLOCK_SIZE);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
  kernel_mat_mul<<<dim_grid, dim_block>>>(
    matA, 
    matB,
    matC,
    a_rows,
    a_cols,
    b_rows,
    b_cols
  );
}