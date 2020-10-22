#include "pinkie/image/csrc/image.h"

namespace pinkie {

Image::Image(
  const PixelType dtype,
  const bool _is_2d
): 
  dtype_(dtype),
  is_2d_(_is_2d), 
  data_(nullptr),
  owned_(false)
{
  size_.setZero();
}

Image::Image(const Image& image, bool copy) {
  frame_ = image.frame_;
  size_ = image.size_;
  is_2d_ = image.is_2d_;
  if (copy == true) {
    clear_memory();
    owned_ = false;
  } else {
    data_ = image.data_;
  }
}

Image::~Image() {
  clear_memory();
}

void Image::clear_memory() {
  if (owned_ == true && data_ != nullptr) {
    free(data_);
  }
}

} // namespace pinkie