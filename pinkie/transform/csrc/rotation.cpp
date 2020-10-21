#include "pinkie/transform/csrc/rotation.h"

namespace pinkie {
namespace transform {

torch::Tensor rotate(
  const torch::Tensor& axis, 
  const float theta_radian
) {
  assert(axis.dim() == 1);
  assert(axis.size(0) == 3);
  assert(axis.scalar_type() == torch::kFloat);

  auto accessor = axis.accessor<float, 1>();

  const float ct = std::cos(theta_radian);
  const float st = std::sin(theta_radian);
  const float one_ct = 1.0f - ct;
  const float r00 = ct + accessor[0] * accessor[0] * one_ct;
  const float r01 = accessor[0] * accessor[1] * one_ct - accessor[2] * st;
  const float r02 = accessor[0] * accessor[2] * one_ct + accessor[1] * st;
  const float r10 = accessor[1] * accessor[0] * one_ct + accessor[2] * st;
  const float r11 = ct+ accessor[1] * accessor[1] * one_ct;
  const float r12 = accessor[1] * accessor[2] * one_ct - accessor[0] * st;
  const float r20 = accessor[2] * accessor[0] * one_ct - accessor[1] * st;
  const float r21 = accessor[2] * accessor[1] * one_ct + accessor[0] * st;
  const float r22 = ct + accessor[2] * accessor[2] * one_ct;
  return torch::tensor({
    {r00, r01, r02},
    {r10, r11, r12},
    {r20, r21, r22}
  }, axis.options());

}

} // namespace transform
} // namespace pinkie
