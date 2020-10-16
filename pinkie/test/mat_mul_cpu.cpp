#include "pinkie/test/mat_mul.h"

void cpu_mat_mul(
  float* matA, 
  float* matB, 
  float* matC, 
  int a_rows, 
  int a_cols,
  int b_rows,
  int b_cols
) {
  for (int row = 0; row < a_rows; row++) {
    for (int col = 0; col < b_cols; col++) {
      float value = 0.0f;
      for (int k = 0; k < a_cols; k++) {
        value += matA[row * a_cols + k] * matB[k * b_cols + col];
      }
      matC[row * b_cols + col] = value;
    }
  }
}