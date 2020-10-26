#include "pinkie/image/csrc/frame.h"

namespace pinkie {

Frame::Frame() {
  origin_.setZero();
  spacing_.setOnes();
  axes_.setIdentity();
}

Frame::Frame(const Frame& frame) {
  origin_ = frame.origin_;
  spacing_ = frame.spacing_;
  axes_ = frame.axes_;
}

void Frame::set_origin(const Eigen::Vector3f& origin) {
  origin_ = origin;
}

void Frame::set_spacing(const Eigen::Vector3f& spacing) {
  spacing_ = spacing;
}

void Frame::set_axes(const Eigen::Matrix3f& axes) {
  axes_ = axes;
}

void Frame::set_axis(const Eigen::Vector3f& axis, const size_t index) {
  assert(index < 3);
  axes_.row(index) = axis;
}

const Eigen::Vector3f& Frame::origin() const {
  return origin_;
}

const Eigen::Vector3f& Frame::spacing() const {
  return spacing_;
}

const Eigen::Matrix3f& Frame::axes() const {
  return axes_;
}

Eigen::Vector3f Frame::axis(const size_t index) const {
  assert(index < 3);
  return axes_.row(index);
}

void Frame::reset() {
  origin_.setZero();
  spacing_.setOnes();
  axes_.setIdentity();
}

Eigen::Vector3f Frame::world_to_voxel(const Eigen::Vector3f& world) const {
  return (axes_ * (world - origin_)).array() / spacing_.array();
}

Eigen::Vector3f Frame::voxel_to_world(const Eigen::Vector3f& voxel) const {
  auto ret = origin_;
  for (size_t i = 0; i < 3; i++) {
    ret += (axes_.row(i) * voxel(i) * spacing_(i));
  }
  return ret;
}

} // namespace pinkie