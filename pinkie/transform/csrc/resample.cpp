#include "pinkie/transform/csrc/resample.h"

#include <omp.h>

namespace pinkie {
namespace transform {

long max(long a, long b) {
  return a > b ? a: b;
}

long min(long a, long b) {
  return a < b ? a : b;
}

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
  auto src_size = src_image.size();

  auto dst_spacing = dst_frame.spacing();

  auto direction_x = torch::matmul(src_axes, dst_frame.axis(0));
  direction_x *= (dst_spacing[0] / src_spacing);
  auto direction_y = torch::matmul(src_axes, dst_frame.axis(1));
  direction_y *= (dst_spacing[1] / src_spacing);
  auto direction_z = torch::matmul(src_axes, dst_frame.axis(2));
  direction_z *= (dst_spacing[2] / src_spacing);

  const long* src_size_ptr = src_size.data_ptr<long>();
  const long* dst_size_ptr = dst_size.data_ptr<long>();

  auto dst_data = torch::zeros(
    {dst_size_ptr[2], dst_size_ptr[1], dst_size_ptr[0]}, 
    src_data.options()
  );

  auto origin = torch::matmul(src_axes, dst_frame.origin() - src_frame.origin()) / src_spacing;

  if (src_image.is_2d() == true) {
    direction_z = torch::tensor({0.0, 0.0, 1.0}, direction_z.options());
    origin[2] = 0.0f;
  }

  auto src_accessor = src_data.accessor<float, 3>();
  auto dst_accessor = dst_data.accessor<float, 3>();

  #pragma omp parallel for
  for (long z = 0; z < dst_size_ptr[2]; z++) {
    for (long y = 0; y < dst_size_ptr[1]; y++) {
      for (long x = 0; x < dst_size_ptr[0]; x++) {
        float curr[3];
        for (int i = 0; i < 3; i++) {
          curr[i] = 
            origin.data_ptr<float>()[i] + 
            direction_z.data_ptr<float>()[i] * static_cast<float>(z) + 
            direction_y.data_ptr<float>()[i] * static_cast<float>(y) + 
            direction_x.data_ptr<float>()[i] * static_cast<float>(x);
        }
        long fx = static_cast<long>(std::floor(curr[0]));
        long fy = static_cast<long>(std::floor(curr[1]));
        long fz = static_cast<long>(std::floor(curr[2]));

        float delta_x = curr[0] - static_cast<float>(fx);
        float delta_y = curr[1] - static_cast<float>(fy);
        float delta_z = curr[2] - static_cast<float>(fz);

        if (
          curr[0] < 0.0 || curr[0] > src_size_ptr[0] - 1 ||
          curr[1] < 0.0 || curr[1] > src_size_ptr[1] - 1 ||
          curr[2] < 0.0 || curr[2] > src_size_ptr[2] - 1
        ) {
          dst_accessor[z][y][x] = padding_value;
          continue;
        }

        // for zy value
        float values_zy[2][2];
        for (long dz = 0; dz < 2; dz++) {
          long cz = (dz == 0) ? fz : fz + 1;
          cz = max(cz, 0);
          cz = min(cz, src_size_ptr[2] - 1);
          for (long dy = 0; dy < 2; dy++) {
            long cy = (dy == 0) ? fy : fy + 1;
            cy = max(cy, 0);
            cy = min(cy, src_size_ptr[1] - 1);

            float value = 0.0f;
            for (long dx = 0; dx < 2; dx++) {
              long cx = (dx == 0)? fx: fx + 1;
              cx = max(cx, 0);
              cx = min(cx, src_size_ptr[0] - 1);

              float curr_delta = (dx == 0)? (1.0f - delta_x) : delta_x;
              value += (src_accessor[cz][cy][cx] * curr_delta);
            } // end for dx
            values_zy[dz][dy] = value;
          } // end for dy
        } //end for dz

        // for z value
        float values_z[2];
        for (long dz = 0; dz < 2; dz++) {
          long cz = (dz == 0) ? fz : fz + 1;
          cz = max(cz, 0);
          cz = min(cz, src_size_ptr[2] - 1);

          float value = 0.0f;
          for (long dy = 0; dy < 2; dy++) {
            long cy = (dy == 0) ? fy : fy + 1;
            cy = max(cy, 0);
            cy = min(cy, src_size_ptr[1] - 1);

            float curr_delta = (dy == 0)? (1.0f - delta_y) : delta_y;
            value += (values_zy[dz][dy] * curr_delta);
          } // end for dy
          values_z[dz] = value;
        } //end for dz

        // for value
        float value = 0.0f;
        for (long dz = 0; dz < 2; dz++) {
          long cz = (dz == 0) ? fz : fz + 1;
          cz = max(cz, 0);
          cz = min(cz, src_size_ptr[2] - 1);

          float curr_delta = (dz == 0)? (1.0f - delta_z) : delta_z;
          value += (values_z[dz] * curr_delta);
        } //end for dz

        dst_accessor[z][y][x] = value;

      } // end for x
    } // end for y
  } // end for z

  Image dst_image(src_image.is_2d());
  dst_image.set_frame(dst_frame);
  dst_image.set_data(dst_data);
  return dst_image;
}

} // namespace transform
} // namespace pinkie
