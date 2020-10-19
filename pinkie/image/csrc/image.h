#ifndef PINKIE_IMAGE_CSRC_IMAGE_H
#define PINKIE_IMAGE_CSRC_IMAGE_H

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

public:
  Image to(const torch::Device& device) const;
  Image to(const std::string& device) const;
  void to_(const torch::Device& device);
  void to_(const std::string& device);

public:
  torch::Device device() const;

private:
  torch::Tensor size_;
  torch::Tensor data_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_IMAGE_H
