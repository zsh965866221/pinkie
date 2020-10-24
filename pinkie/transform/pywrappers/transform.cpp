#include "pinkie/transform/pywrappers/transform.h"

#include "pinkie/transform/csrc/rotation.h"

using namespace pinkie;

void transform_rotate(
  float* axis, 
  float theta_radian, 
  float* out
) {
  assert(axis != nullptr);
  assert(out != nullptr);

  auto matrix = rotate(
    Eigen::Map<Eigen::Vector3f>(axis),
    theta_radian
  );
  memcpy(out, matrix.data(), sizeof(float) * 9);
}
