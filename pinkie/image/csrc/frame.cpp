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

void Frame::set_origin(const torch::Tensor& origin) {
  assert(origin.sizes() == origin_.sizes());
  assert(origin.scalar_type() == origin_.scalar_type());
  assert(origin.device() == origin_.device());
  origin_ = origin;
}

void Frame::set_spacing(const torch::Tensor& spacing) {
  assert(spacing.sizes() == spacing_.sizes());
  assert(spacing.scalar_type() == spacing_.scalar_type());
  assert(spacing.device() == spacing_.device());
  spacing_ = spacing;
}

void Frame::set_axes(const torch::Tensor& axes) {
  assert(axes.sizes() == axes_.sizes());
  assert(axes.scalar_type() == axes_.scalar_type());
  assert(axes.device() == axes_.device());
  axes_ = axes;
}

void Frame::set_axis(const torch::Tensor& axis, const int index) {
  assert(index <= 2);
  assert(axis.dim() == 1);
  assert(axis.size(0) == axes_.size(0));
  assert(axis.scalar_type() == axes_.scalar_type());
  assert(axis.device() == axes_.device());
  axes_.index_put_({torch::indexing::Slice(), index}, axis);
}

Frame Frame::to(const c10::Device& device) const {
  auto frame = Frame(*this);
  frame.origin_ = frame.origin_.to(device);
  frame.spacing_ = frame.spacing_.to(device);
  frame.axes_ = frame.axes_.to(device);
  return frame;
}

Frame Frame::to(const std::string& device) const {
  return this->to(c10::Device(device));
}

void Frame::to_(const c10::Device& device) {
  origin_ = origin_.to(device);
  spacing_ = spacing_.to(device);
  axes_ = axes_.to(device);
}

void Frame::to_(const std::string& device) {
  this->to_(c10::Device(device));
}

c10::Device Frame::device() const {
  return origin_.device();
}

} // namespace pinkie