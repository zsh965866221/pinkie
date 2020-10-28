#include "pinkie/cuda/image/csrc/image.h"

#include <cuda_runtime_api.h>
#include <memory.h>

#include "pinkie/utils/csrc/cuda_header.h"

namespace pinkie {

CudaImage::CudaImage(
  const PixelType dtype,
  const bool _is_2d,
  const int device
):Image(dtype, _is_2d), device_(device) {}

CudaImage::CudaImage(const CudaImage& image, bool copy) {
  frame_ = image.frame_;
  size_ = image.size_;
  is_2d_ = image.is_2d_;
  dtype_ = image.dtype_;
  device_ = image.device_;
  if (copy == true) {
    clear_memory();
    size_t byte_size = update_buffer();
    call_device_func([&]() {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      CUDA_CHECK(cudaMemcpyPeerAsync(
        data_, device_, image.data_, 
        image.device_, byte_size, stream
      ));
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    });
  } else {
    data_ = image.data_;
    owned_ = false;
  }
}

CudaImage::CudaImage(
  const int& height,
  const int& width,
  const int& depth,
  const PixelType dtype,
  const bool _is_2d,
  const int device
) {
  size_(0) = height;
  size_(1) = width;
  size_(2) = depth;
  is_2d_ = _is_2d;
  dtype_ = dtype;
  device_ = device;

  update_buffer();
}

CudaImage::~CudaImage() {
  clear_memory();
}

void CudaImage::clear_memory() {
  if (owned_ == true && data_ != nullptr) {
    call_device_func([&]() {
      CUDA_CHECK(cudaFree(data_));
    });
    data_ = nullptr;
    owned_ = false;
  }
}

size_t CudaImage::update_buffer() {
  owned_ = true;
  size_t byte_size = pixeltype_byte_size(dtype_, size_);
  call_device_func([&]() {
    CUDA_CHECK(cudaMallocManaged((void**)&data_, byte_size));
  });
  return byte_size;
}

void CudaImage::call_device_func(std::function<void()> func) {
  CUDA_CHECK(cudaSetDevice(device_));
  func();
  CUDA_CHECK(cudaSetDevice(0));
}

} // namespace pinkie