#ifndef PINKIE_IMAGE_CUDA_IMAGE_H
#define PINKIE_IMAGE_CUDA_IMAGE_H

#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/image.h"
#include "pinkie/image/csrc/pixel_type.h"

namespace pinkie {

class CudaImage: public Image {
public:
  CudaImage(
    const PixelType dtype = PixelType_float32,
    const bool _is_2d = false
  );
  CudaImage(const Image& image, bool copy = true);
  CudaImage(
    const int& height,
    const int& width,
    const int& depth,
    const PixelType dtype = PixelType_float32,
    const bool _is_2d = false
  );
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

public:
  Image* cast(const PixelType& dtype) const;
  void cast_(const PixelType& dtype);

protected:
  void clear_memory();
  size_t update_buffer();

private:
  int device_;

};

} // namespace pinkie

#endif // PINKIE_IMAGE_CUDA_IMAGE_H
