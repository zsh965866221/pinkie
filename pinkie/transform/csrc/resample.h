#ifndef PINKIE_TRANSFORM_CSRC_RESAMPLE_H
#define PINKIE_TRANSFORM_CSRC_RESAMPLE_H

#include "pinkie/image/csrc/image.h"

namespace pinkie {

Image* resample_trilinear(
  const Image& src_image, 
  const Frame& dst_frame,
  const Eigen::Vector3i& dst_size,
  float padding_value = 0.0f
);

} // namespace pinkie

#endif // PINKIE_TRANSFORM_CSRC_RESAMPLE_H