#ifndef PINKIE_TRANSFORM_CSRC_TRANSFORM_H
#define PINKIE_TRANSFORM_CSRC_TRANSFORM_H

#include <torch/torch.h>

#include "pinkie/image/csrc/image.h"

namespace pinkie {
namespace transform {

Image resample_trilinear(
  const Image& image, 
  const Frame& frame,
  const torch::Tensor& size
);

} // namespace transform
} // namespace pinkie

#endif // PINKIE_TRANSFORM_CSRC_TRANSFORM_H