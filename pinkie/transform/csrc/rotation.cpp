#include "pinkie/transform/csrc/rotation.h"

namespace pinkie {

Eigen::Matrix3f rotate(
  const Eigen::Vector3f& axis, 
  const float theta_radian
) {
  assert(axis.dim() == 1);
  assert(axis.size(0) == 3);
  assert(axis.scalar_type() == torch::kFloat);


  const float ct = std::cos(theta_radian);
  const float st = std::sin(theta_radian);
  const float one_ct = 1.0f - ct;
  const float r00 = ct + axis(0) * axis(0) * one_ct;
  const float r01 = axis(0) * axis(1) * one_ct - axis(2) * st;
  const float r02 = axis(0) * axis(2) * one_ct + axis(1) * st;
  const float r10 = axis(1) * axis(0) * one_ct + axis(2) * st;
  const float r11 = ct+ axis(1) * axis(1) * one_ct;
  const float r12 = axis(1) * axis(2) * one_ct - axis(0) * st;
  const float r20 = axis(2) * axis(0) * one_ct - axis(1) * st;
  const float r21 = axis(2) * axis(1) * one_ct + axis(0) * st;
  const float r22 = ct + axis(2) * axis(2) * one_ct;
  float ret[9] = {
    r00, r01, r02,
    r10, r11, r12,
    r20, r21, r22
  };
  return Eigen::Map<Eigen::Matrix3f>(ret).transpose();


}

} // namespace pinkie
