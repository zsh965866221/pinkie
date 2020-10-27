#include <cuda_runtime_api.h>
#include <iostream>


int main() {
  cudaSetDevice(0);
  cudaDeviceProp cuda_prop;
  cudaGetDeviceProperties(&cuda_prop, 0);
  std::cout 
    << cuda_prop.name 
    << "; global mem: " << cuda_prop.totalGlobalMem
    << "; threads per block: " << cuda_prop.maxThreadsPerBlock
    << "; max thread dim: " << cuda_prop.maxThreadsDim[0] << std::endl
    << "; max thread dim: " << cuda_prop.maxThreadsDim[1] << std::endl
    << "; max thread dim: " << cuda_prop.maxThreadsDim[2] << std::endl
    << "; maxGridSize: " << cuda_prop.maxGridSize[0] << std::endl
    << "; maxGridSize: " << cuda_prop.maxGridSize[1] << std::endl
    << "; maxGridSize: " << cuda_prop.maxGridSize[2] << std::endl;
    

}