#ifndef PINKIE_IMAGE_CSRC_IMAGE_H
#define PINKIE_IMAGE_CSRC_IMAGE_H

#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/pixel_type.h"

namespace pinkie {

class Image {
public:
  Image(
    const PixelType dtype = PixelType_float32,
    const bool _is_2d = false
  );
  Image(const Image& image, bool copy = true);
  Image(
    const int& height,
    const int& width,
    const int& depth,
    const PixelType dtype = PixelType_float32,
    const bool _is_2d = false
  );
  virtual ~Image();

public:
  const Eigen::Vector3i& size() const;

public:
  const Frame& frame() const;
  void set_frame(const Frame& frame);

public:
  void* data() const;
  void set_data(
    void* data, 
    const Eigen::Vector3i& size, 
    const PixelType dtype,
    bool copy = true
  );
  void set_data(
    void* data, 
    const int height, 
    const int width,
    const int depth,
    const PixelType dtype, 
    bool copy = true
  );

public:
  bool is_2d() const;
  bool set_2d(bool p);

public:
  PixelType dtype() const;
  Image cast(const PixelType& dtype) const;
  void cast_(const PixelType& dtype);

private:
  void clear_memory();

private:
  void* data_;
  Eigen::Vector3i size_;
  Frame frame_;
  bool is_2d_;
  PixelType dtype_;
  bool owned_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_IMAGE_H
