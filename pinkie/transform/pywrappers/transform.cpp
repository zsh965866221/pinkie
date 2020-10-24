#include "pinkie/transform/pywrappers/transform.h"

#include "pinkie/transform/csrc/resample.h"
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


void* transform_resample_trilinear(
  void* src_image_ptr,
  void* dst_frame_ptr,
  int* dst_size_ptr,
  float padding_value
) {
  assert(src_image_ptr != nullptr);
  assert(dst_frame_ptr != nullptr);
  assert(dst_size_ptr != nullptr);

  Image* src_image = static_cast<Image*>(src_image_ptr);
  Frame* dst_frame = static_cast<Frame*>(dst_frame_ptr);

  return resample_trilinear(
    *src_image,
    *dst_frame,
    Eigen::Map<Eigen::Vector3i>(dst_size_ptr),
    padding_value
  );
}