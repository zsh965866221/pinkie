#ifndef PINKIE_TRANSFORM_CSRC_ROTATION_H
#define PINKIE_TRANSFORM_CSRC_ROTATION_H

#include <Eigen/Eigen>

namespace pinkie {

/** \brief comate rotation matrix with rotation axis and theta
 * \ref https://en.wikipedia.org/wiki/Rotation_matrix
*/
Eigen::Matrix3f rotate(
  const Eigen::Vector3f& axis, 
  const float theta_radian
);

} // namespace pinkie

#endif // PINKIE_TRANSFORM_CSRC_ROTATION_H