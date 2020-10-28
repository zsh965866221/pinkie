#ifndef PINKIE_CUDA_IMAGE_IMAGE_H
#define PINKIE_CUDA_IMAGE_IMAGE_H

#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/image.h"
#include "pinkie/image/csrc/pixel_type.h"

namespace pinkie {

class CudaImage: public Image {
public:
  CudaImage(
    const PixelType dtype = PixelType_float32,
    const bool _is_2d = false,
    const int device = 0
  );
  CudaImage(const CudaImage& image, bool copy = true);
  CudaImage(
    const int& height,
    const int& width,
    const int& depth,
    const PixelType dtype = PixelType_float32,
    const bool _is_2d = false,
    const int device = 0
  );
  CudaImage(const Image& image, int device = 0);
  virtual ~CudaImage();

public:
  void set_data(
    void* data, 
    const int height, 
    const int width,
    const int depth,
    const PixelType dtype, 
    bool _is_2d = false,
    bool copy = true
  );
  void set_zero();
  void allocate(
    const int height,
    const int width,
    const int depth,
    const PixelType dtype,
    const int device = 0
  );

public:
  CudaImage* cast(const PixelType& dtype) const;
  void cast_(const PixelType& dtype);

public:
  int device() const;
  Image* cpu() const;
  CudaImage* cuda(const int device = 0) const;
  void cuda_(const int device = 0);


protected:
  void clear_memory();
  size_t update_buffer();

private:
  void call_device_func(std::function<void()> func);

private:
  int device_;

};

} // namespace pinkie

#endif // PINKIE_CUDA_IMAGE_IMAGE_H
