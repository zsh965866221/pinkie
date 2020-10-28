#include "pinkie/cuda/ops/csrc/ops.h"

#include "pinkie/utils/csrc/cuda_header.h"

template<typename T_src, typename T_dst>
__global__ void kernel_cast(
  T_src* data_src, T_dst* data_dst, 
  int height, int width, int depth
) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= height || y >= width || z >= depth) {
    return;
  }

  int index_xy = z * height * width;
  int index_x = index_xy + y * height;
  int index = index_x + x;
  data_dst[index] = static_cast<T_dst>(data_src[index]);
}

template<typename T_src, typename T_dst>
void cuda_cast(
  T_src* data_src, T_dst* data_dst, 
  int height, int width, int depth,
  cudaStream_t* stream
) {
  assert(data_src != nullptr);
  assert(data_dst != nullptr);

  dim3 dim_grid = generate_dim(height, width, depth);
  if (stream == nullptr) {
    kernel_cast<T_src, T_dst><<<dim_grid, DIM_BLOCK>>>(
      data_src, data_dst, height, width, depth
    );  
  } else {
    kernel_cast<T_src, T_dst><<<dim_grid, DIM_BLOCK, 0, *stream>>>(
      data_src, data_dst, height, width, depth
    ); 
  }
}
