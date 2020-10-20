#include "pinkie/transform/csrc/resample.h"

namespace pinkie {
namespace transform {

Image resample_trilinear(
  const Image& src_image, 
  const Frame& dst_frame,
  const torch::Tensor& dst_size
) {
  auto data = src_image.data();
  auto src_frame = src_image.frame();
  auto src_axes = src_frame.axes();
  auto src_spacing = src_frame.spacing();

  auto delta_x = torch::mm(src_axes, dst_frame.axis(0));
  delta_x *= (dst_frame.spacing()[0] / src_spacing);
  auto delta_y = torch::mm(src_axes, dst_frame.axis(1));
  delta_y *= (dst_frame.spacing()[1] / src_spacing);
  auto delta_z = torch::mm(src_axes, dst_frame.axis(2));
  delta_z *= (dst_frame.spacing()[2] / src_spacing);

  auto origin = torch::mm(src_axes, dst_frame.origin()) / src_spacing

  for (int z = 0; z < dst_size[2]; z++) {
    for (int x = 0; x < dst_size[0]; x++) {
      for (int y = 0; y < dst_size[1]; y++) {
        auto curr = origin + delta_z * z + delta_x * x + delta_y * y;
        // compute trilinear resample
      }
    }
  }

  return src_image;
}

} // namespace transform
} // namespace pinkie
