#ifndef PINKIE_IMAGE_CSRC_FRAME_H
#define PINKIE_IMAGE_CSRC_FRAME_H

#include <string>

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
  Frame to(const torch::Device& device) const;
  void to_(const torch::Device& device);

public: 
 Frame clone() const;

public:
  torch::Device device() const;

public:
  torch::Tensor world_to_voxel(const torch::Tensor& world) const;
  torch::Tensor voxel_to_world(const torch::Tensor& voxel) const;

private:
  torch::Tensor origin_;
  torch::Tensor spacing_;
  torch::Tensor axes_;
};

} // namespace pinkie

#endif // PINKIE_IMAGE_CSRC_FRAME_H