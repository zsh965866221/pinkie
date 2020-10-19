#ifndef PINKIE_IMAGE_CSRC_FRAME_H
#define PINKIE_IMAGE_CSRC_FRAME_H

#include <string>

#include <c10/core/Device.h>
#include <torch/torch.h>

namespace pinkie {

class Frame {
public:
  Frame();
  Frame(const Frame& frame);
  virtual ~Frame() {}

public:
  torch::Tensor origin() const;
  torch::Tensor spacing() const;
  torch::Tensor axes() const;
  torch::Tensor axis(int index) const;

public:
  void set_origin(const torch::Tensor& origin);
  void set_spacing(const torch::Tensor& spacing);
  void set_axes(const torch::Tensor& axes);
  void set_axis(const torch::Tensor& axis, const int index);

public:
  Frame to(const c10::Device& device) const;
  Frame to(const std::string& device) const;
  void to_(const c10::Device& device);
  void to_(const std::string& device);

public:
  c10::Device device() const;

private:
  torch::Tensor origin_;
  torch::Tensor spacing_;
  torch::Tensor axes_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_FRAME_H