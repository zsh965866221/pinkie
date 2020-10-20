#include "pinkie/transform/csrc/resample.h"

namespace pinkie {
namespace transform {

Image resample_trilinear(
  const Image& image, 
  const Frame& frame,
  const torch::Tensor& size
) {
  return image;
}

} // namespace transform
} // namespace pinkie
