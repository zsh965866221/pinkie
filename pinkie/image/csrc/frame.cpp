#include "pinkie/image/csrc/frame.h"

namespace pinkie {

Frame::Frame() {
  origin_ = torch::zeros({3});
  spacing_ = torch::ones({3});
  axes_ = torch::eye(3);
}

Frame::Frame(const Frame& frame) {
  origin_ = frame.origin_;
  spacing_ = frame.spacing_;
  axes_ = frame.axes_;
}

torch::Tensor Frame::origin() const {
  return origin_;
}

torch::Tensor Frame::spacing() const {
  return spacing_;
}

torch::Tensor Frame::axes() const {
  return axes_;
}

torch::Tensor Frame::axis(const int index) const {
  assert(index <= 2);
  return axes_.index({torch::indexing::Slice(), index});
}

} // namespace pinkie