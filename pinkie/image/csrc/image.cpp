#include "pinkie/image/csrc/image.h"

namespace pinkie {

Image::Image(
  const PixelType dtype,
  const bool _is_2d
):
  data_(nullptr),
  is_2d_(_is_2d), 
  dtype_(dtype),
  owned_(false)
{
  size_.setZero();
}

Image::Image(const Image& image, bool copy) {
  frame_ = image.frame_;
  size_ = image.size_;
  is_2d_ = image.is_2d_;
  dtype_ = image.dtype_;
  if (copy == true) {
    clear_memory();
    owned_ = true;
    size_t byte_size = pixeltype_byte_size(dtype_, size_);
    data_ = malloc(byte_size);
    memcpy(data_, image.data_, byte_size);
  } else {
    data_ = image.data_;
    owned_ = false;
  }
}

Image::Image(
  const int& height,
  const int& width,
  const int& depth,
  const PixelType dtype,
  const bool _is_2d
) {
  size_(0) = height;
  size_(1) = width;
  size_(2) = depth;
  is_2d_ = _is_2d;
  owned_ = true;
  dtype_ = dtype;

  size_t byte_size = pixeltype_byte_size(
    dtype_, 
    size_
  );
  data_ = malloc(byte_size);
}

Image::~Image() {
  clear_memory();
}

void Image::clear_memory() {
  if (owned_ == true && data_ != nullptr) {
    free(data_);
  }
}

const Eigen::Vector3i& Image::size() const {
  return size_;
}

const Frame& Image::frame() const {
  return frame_;
}

void Image::set_frame(const Frame& frame) {
  frame_ = frame;
}

void* Image::data() const {
  return data_;
}

void* Image::data() {
  return data_;
}

void Image::set_data(
  void* data, 
  const int height, 
  const int width,
  const int depth,
  const PixelType dtype, 
  bool _is_2d,
  bool copy
) {
  assert(data != nullptr);
  size_(0) = height;
  size_(1) = width;
  size_(2) = depth;
  dtype_ = dtype;
  is_2d_ = _is_2d;
  if (copy == true) {
    clear_memory();
    owned_ = true;
    size_t byte_size = pixeltype_byte_size(dtype_, size_);
    data_ = malloc(byte_size);
    memcpy(data_, data, byte_size);
  } else {
    owned_ = false;
  }
}

void Image::set_data(
  void* data, 
  const Eigen::Vector3i& size, 
  const PixelType dtype,
  bool _is_2d,
  bool copy
) {

  set_data(
    data,
    size(0),
    size(1),
    size(2),
    dtype,
    _is_2d,
    copy
  );
  
}

bool Image::is_2d() const {
  return is_2d_;
}

void Image::set_2d(bool p) {
  is_2d_ = p;
}

PixelType Image::dtype() const {
  return dtype_;
}

Image Image::cast(const PixelType& dtype) const {
  if (dtype == dtype_) {
    return Image(*this);
  }
  int height = size_(0);
  int width = size_(1);
  int depth = size_(2);
  Image image(
    height,
    width,
    depth,
    dtype,
    is_2d_
  );
  // for (int i = 0; i < height * width * depth; i++) {
  // }
}

} // namespace pinkie