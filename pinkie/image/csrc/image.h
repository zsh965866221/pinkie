#ifndef PINKIE_IMAGE_CSRC_IMAGE_H
#define PINKIE_IMAGE_CSRC_IMAGE_H

#include "pinkie/image/csrc/frame.h"

#include <string>

#include <torch/torch.h>

namespace pinkie {

class Image {
public:
  Image();
  Image(const Image& image);
  Image(
    const int& height,
    const int& width,
    const int& depth,
    const torch::ScalarType& type = torch::kFloat
    const tircg::Device device = torch::Device(torch::kCPU)
  );
  virtual ~Image() {}

public:
  torch::Tensor size();

public:
  Frame frame() const;
  void set_frame(const Frame& frame);

public:
  Image to(const torch::Device& device) const;
  void to_(const torch::Device& device);

public:
  torch::Device device() const;

private:
  torch::Tensor size_;
  torch::Tensor data_;
  Frame frame_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_IMAGE_H
