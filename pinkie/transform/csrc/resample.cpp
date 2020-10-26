#include "pinkie/transform/csrc/resample.h"

#include <omp.h>

namespace pinkie {

Image* resample_trilinear(
  const Image& src_image, 
  const Frame& dst_frame,
  const Eigen::Vector3i& dst_size,
  float padding_value
) {
  assert(dst_image != nullptr);

  const auto& dst_spacing = dst_frame.spacing();
  const auto& dst_origin = dst_frame.origin();

  const auto& src_axes = src_image.frame().axes();
  const auto& src_spacing = src_image.frame().spacing();
  const auto& src_size = src_image.size();
  const auto& src_origin = src_image.frame().origin();

  Eigen::Vector3f direction_x = src_axes * dst_frame.axis(0);
  direction_x = direction_x.array() / src_spacing.array() * dst_spacing[0];
  Eigen::Vector3f direction_y = src_axes * dst_frame.axis(1);
  direction_y = direction_y.array() / src_spacing.array() * dst_spacing[1];
  Eigen::Vector3f direction_z = src_axes * dst_frame.axis(2);
  direction_z = direction_z.array() / src_spacing.array() * dst_spacing[2];

  Image* dst_image = new Image(
    dst_size(0),
    dst_size(1),
    dst_size(2),
    src_image.dtype(),
    src_image.is_2d()
  );
  dst_image->set_frame(dst_frame);

  Eigen::Vector3f origin = src_image.frame().world_to_voxel(dst_origin);

  if (src_image.is_2d() == true) {
    direction_z =  Eigen::Vector3f(0.0, 0.0, 1.0);
    origin(2) = 0.0f;
  }

  CALL_DTYPE(
    src_image.dtype(), type,
    [&]() {
      for (int z = 0; z < dst_size(2); z++) {
        int index_base_xy = z * dst_size(0) * dst_size(1);
        for (int y = 0; y < dst_size[1]; y++) {
          int index_base_x = index_base_xy + y * dst_size(0);
          for (int x = 0; x < dst_size(0); x++) {
            int index_base = index_base_x + x;
            float curr[3];
            for (int i = 0; i < 3; i++) {
              curr[i] = 
                origin(i) + 
                direction_z(i) * static_cast<float>(z) + 
                direction_y(i) * static_cast<float>(y) + 
                direction_x(i) * static_cast<float>(x);
            }
            int fx = static_cast<int>(std::floor(curr[0]));
            int fy = static_cast<int>(std::floor(curr[1]));
            int fz = static_cast<int>(std::floor(curr[2]));

            float delta_x = curr[0] - static_cast<float>(fx);
            float delta_y = curr[1] - static_cast<float>(fy);
            float delta_z = curr[2] - static_cast<float>(fz);

            if (
              curr[0] < 0.0 || curr[0] > src_size(0) - 1 ||
              curr[1] < 0.0 || curr[1] > src_size(1) - 1 ||
              curr[2] < 0.0 || curr[2] > src_size(2) - 1
            ) {
              dst_image->data<type>()[index_base] = static_cast<type>(padding_value);
              continue;
            }

            // for zy value
            float values_zy[2][2];
            for (int dz = 0; dz < 2; dz++) {
              int cz = (dz == 0) ? fz : fz + 1;
              cz = std::max(cz, 0);
              cz = std::min(cz, src_size(2) - 1);
              int index_xy = cz * dst_size(0) * dst_size(1);
              for (int dy = 0; dy < 2; dy++) {
                int cy = (dy == 0) ? fy : fy + 1;
                cy = std::max(cy, 0);
                cy = std::min(cy, src_size(1) - 1);
                int index_x = index_xy + cy * dst_size(0);

                float value = 0.0f;
                for (int dx = 0; dx < 2; dx++) {
                  int cx = (dx == 0)? fx: fx + 1;
                  cx = std::max(cx, 0);
                  cx = std::min(cx, src_size(0) - 1);
                  int index = index_x + cx;

                  float curr_delta = (dx == 0)? (1.0f - delta_x) : delta_x;
                  value += (static_cast<float>(src_image.data<type>()[index]) * curr_delta);
                } // end for dx
                values_zy[dz][dy] = value;
              } // end for dy
            } //end for dz

            // for z value
            float values_z[2];
            for (int dz = 0; dz < 2; dz++) {
              int cz = (dz == 0) ? fz : fz + 1;
              cz = std::max(cz, 0);
              cz = std::min(cz, src_size(2) - 1);

              float value = 0.0f;
              for (int dy = 0; dy < 2; dy++) {
                int cy = (dy == 0) ? fy : fy + 1;
                cy = std::max(cy, 0);
                cy = std::min(cy, src_size(1) - 1);

                float curr_delta = (dy == 0)? (1.0f - delta_y) : delta_y;
                value += (values_zy[dz][dy] * curr_delta);
              } // end for dy
              values_z[dz] = value;
            } //end for dz

            // for value
            float value = 0.0f;
            for (int dz = 0; dz < 2; dz++) {
              int cz = (dz == 0) ? fz : fz + 1;
              cz = std::max(cz, 0);
              cz = std::min(cz, src_size(2) - 1);

              float curr_delta = (dz == 0)? (1.0f - delta_z) : delta_z;
              value += (values_z[dz] * curr_delta);
            } //end for dz

            dst_image->data<type>()[index_base] = static_cast<type>(value);
          } // end for x
        } // end for y
      } // end for z
    }
  );

  return dst_image;
}

} // namespace pinkie
