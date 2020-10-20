#include "pinkie/transform/csrc/resample.h"

namespace pinkie {
namespace transform {

Image resample_trilinear(
  const Image& src_image, 
  const Frame& dst_frame,
  const torch::Tensor& dst_size,
  float padding_value
) {
  auto& src_data = src_image.data();
  auto src_frame = src_image.frame();
  auto src_axes = src_frame.axes();
  auto src_spacing = src_frame.spacing();

  torch::Tensor dst_data(src_data);

  auto delta_x = torch::mm(src_axes, dst_frame.axis(0));
  delta_x *= (dst_frame.spacing()[0] / src_spacing);
  auto delta_y = torch::mm(src_axes, dst_frame.axis(1));
  delta_y *= (dst_frame.spacing()[1] / src_spacing);
  auto delta_z = torch::mm(src_axes, dst_frame.axis(2));
  delta_z *= (dst_frame.spacing()[2] / src_spacing);

  auto origin = torch::mm(src_axes, dst_frame.origin()) / src_spacing;

  auto src_accessor = src_data.accessor<float, 3>();

  auto dst_accessor = dst_data.accessor<float, 3>();
  const int* dst_size_ptr = dst_size.data_ptr<int>();

  for (int z = 0; z < dst_size_ptr[2]; z++) {
    for (int x = 0; x < dst_size_ptr[0]; x++) {
      for (int y = 0; y < dst_size_ptr[1]; y++) {
        auto curr = origin + delta_z * z + delta_x * x + delta_y * y;
        float* curr_ptr = curr.data_ptr<float>();
        int fx = static_cast<int>(std::floor(curr_ptr[0]));
        int fy = static_cast<int>(std::floor(curr_ptr[1]));
        int fz = static_cast<int>(std::floor(curr_ptr[2]));

        float delta_x = curr_ptr[0] - static_cast<float>(fx);
        float delta_y = curr_ptr[1] - static_cast<float>(fy);
        float delta_z = curr_ptr[2] - static_cast<float>(fz);

        if (
          curr_ptr[0] < 0.0 || curr_ptr[0] > static_cast<float>(src_size.size(0) - 1) ||
          curr_ptr[1] < 0.0 || curr_ptr[1] > static_cast<float>(src_size.size(1) - 1) ||
          curr_ptr[2] < 0.0 || curr_ptr[2] > static_cast<float>(src_size.size(2) - 1)
        ) {
          dst_accessor[z, x, y] = padding_value;
          continue;
        }

        float zx[2][2];
        for (int dz = 0; dz < 2; dz++) {
          int cz = (dz == 0) ? fz : fz + 1;
          for (int dx = 0; dx < 2; dx++) {
            int cx = (dx == 0) ? fx : fx + 1;
            float value = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
              int cy = (dy == 0)? fy: fy + 1;

              float curr_delta_y = (dy == 0)? (1.0f - delta_y) : delta_y;
              float cv = 0.0f;
              if (
                curr_ptr[0] < 0.0 || curr_ptr[0] > static_cast<float>(src_size.size(0) - 1) ||
                curr_ptr[1] < 0.0 || curr_ptr[1] > static_cast<float>(src_size.size(1) - 1) ||
                curr_ptr[2] < 0.0 || curr_ptr[2] > static_cast<float>(src_size.size(2) - 1)
              ) {
                dst_accessor[z, x, y] = padding_value;
                continue;
              }

              value += ()
            } // end for dy
          } // end for dx
        } //end for dz

      } // end for y
    } // end for x
  } // end for z

  return src_image;
}

} // namespace transform
} // namespace pinkie
