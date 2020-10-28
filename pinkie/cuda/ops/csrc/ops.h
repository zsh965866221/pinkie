#ifndef PINKIE_CUDA_OPS_OPS_H
#define PINKIE_CUDA_OPS_OPS_H

#include <cuda_runtime_api.h>
#include <math.h>

namespace pinkie {

template<typename T_src, typename T_dst>
void cuda_cast(
  T_src* data_src, T_dst* data_dst, 
  int height, int width, int depth,
  cudaStream_t* stream = nullptr
);

} // namespace pinkie

#endif // PINKIE_OPS_CUDA_OPS_H
