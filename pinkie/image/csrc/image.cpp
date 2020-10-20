#include "pinkie/image/csrc/image.h"

namespace pinkie {

Image::Image(
  const int& height,
  const int& width,
  const int& depth,
  const torch::ScalarType dtype,
  const torch::Device device,
  const bool _is_2d
) {
  assert(
    (height >= 0) &&
    (width >= 0) &&
    (depth >= 0)
  );
  auto options =
    torch::TensorOptions()
    .dtype(dtype)
    .device(device);
  data_ = torch::zeros({depth, height, width,}, options);
  size_ = torch::tensor({depth, height, width}, device);
  frame_ = Frame();
  frame_.to_(device);
  is_2d_ = _is_2d;
}

Image::Image(const bool _is_2d): Image(0, 0, 0) {
  is_2d_ = _is_2d;
}

Image::Image(const Image& image) {
  data_ = image.data_;
  frame_ = image.frame_;
  size_ = image.size_;
  is_2d_ = image.is_2d_;
}


torch::Device Image::device() const {
  return data_.device();
}

torch::ScalarType Image::dtype() const {
  return data_.scalar_type();
}

Frame Image::frame() const {
  return frame_;
}

void Image::set_frame(const Frame& frame) {
  frame_ = frame;
  if (frame_.device() != data_.device()) {
    frame_.to_(data_.device());
  }
}

void Image::to_(const torch::Device& device) {
  if (device != data_.device()) {
    data_ = data_.to(device);
    frame_.to_(device);
    size_ = size_.to(device);
  }
}

Image Image::to(const torch::Device& device) const {
  Image image;
  image.data_ = data_.to(device);
  image.frame_.to_(device);
  image.size_ = size_.to(device);
  return image;
}

Image Image::cast(const torch::ScalarType& type) const {
  Image image;
  image.data_ = data_.to(type);
  image.frame_ = frame_;
  image.size_ = size_;
  return image;
}

void Image::cast_(const torch::ScalarType& type) {
  if (data_.scalar_type() != type) {
    data_ = data_.to(type);
  }
}

const torch::Tensor& Image::data() const {
  return data_;
}

void Image::set_data(const torch::Tensor& data) {
  assert(data.dim() == 3);
  data_ = data;
  if (frame_.device() != data_.device()) {
    frame_.to_(data_.device());
    size_ = size_.to(data.device());
  }
  size_[0] = data.size(2);
  size_[1] = data.size(1);
  size_[2] = data.size(0);
}

torch::Tensor Image::size() const {
  return size_;
}

bool Image::is_2d() const {
  return is_2d_;
}

torch::Tensor Image::origin() const {
  return frame_.origin();
}

torch::Tensor Image::spacing() const {
  return frame_.spacing();
}

torch::Tensor Image::axes() const {
  return frame_.axes();
}

torch::Tensor Image::axis(int index) const {
  return frame_.axis(index);
}

torch::Tensor Image::world_to_voxel(const torch::Tensor& world) const {
  return frame_.world_to_voxel(world);
}

torch::Tensor Image::voxel_to_world(const torch::Tensor& voxel) const {
  return frame_.voxel_to_world(voxel);
}


} // namespace pinkie