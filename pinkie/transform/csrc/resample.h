#ifndef PINKIE_TRANSFORM_CSRC_TRANSFORM_H
#define PINKIE_TRANSFORM_CSRC_TRANSFORM_H

#include <torch/torch.h>

#include "pinkie/image/csrc/image.h"

namespace pinkie {
namespace transform {

Image resample_trilinear(
  const Image& src_image, 
  const Frame& dst_frame,
  const torch::Tensor& dst_size,
  float padding_value = 0.0f
);

} // namespace transform
} // namespace pinkie

#endif // PINKIE_TRANSFORM_CSRC_TRANSFORM_H