#ifndef PINKIE_TRANSFORM_CSRC_ROTATION_H
#define PINKIE_TRANSFORM_CSRC_ROTATION_H

#include <torch/torch.h>

namespace pinkie {
namespace transform {

/** \brief comate rotation matrix with rotation axis and theta
 * \ref https://en.wikipedia.org/wiki/Rotation_matrix
*/
torch::Tensor rotate(
  const torch::Tensor& axis, 
  const float theta_radian
);

} // namespace transform
} // namespace pinkie

#endif // PINKIE_TRANSFORM_CSRC_ROTATION_H