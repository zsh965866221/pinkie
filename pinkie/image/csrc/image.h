#ifndef PINKIE_IMAGE_CSRC_IMAGE_H
#define PINKIE_IMAGE_CSRC_IMAGE_H

#include "pinkie/image/csrc/frame.h"

#include <string>

#include <torch/torch.h>

namespace pinkie {

class Image {
public:
  Image(
    const bool _is_2d = false
  );
  Image(const Image& image);
  Image(
    const int& height,
    const int& width,
    const int& depth,
    const torch::ScalarType dtype = torch::kFloat,
    const torch::Device device = torch::Device(torch::kCPU),
    const bool _is_2d = false
  );
  virtual ~Image() {}

public:
  torch::Tensor size() const;

public:
  Frame frame() const;
  void set_frame(const Frame& frame);

public:
  const torch::Tensor& data() const;
  void set_data(const torch::Tensor& data);

public:
  Image to(const torch::Device& device) const;
  void to_(const torch::Device& device);

public:
  torch::Device device() const;
  torch::ScalarType dtype() const;
  bool is_2d() const;

public:
  Image cast(const torch::ScalarType& type) const;
  void cast_(const torch::ScalarType& type);

public:
  torch::Tensor origin() const;
  torch::Tensor spacing() const;
  torch::Tensor axes() const;
  torch::Tensor axis(int index) const;

public:
  torch::Tensor world_to_voxel(const torch::Tensor& world) const;
  torch::Tensor voxel_to_world(const torch::Tensor& voxel) const;

private:
  torch::Tensor data_;
  torch::Tensor size_;
  Frame frame_;
  bool is_2d_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_IMAGE_H
