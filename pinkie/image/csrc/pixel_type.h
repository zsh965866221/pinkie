#ifndef PINKIE_IMAGE_SRC_PIXEL_TYPE_H
#define PINKIE_IMAGE_SRC_PIXEL_TYPE_H

#include <string>

namespace pinkie {

enum PixelType {
  PixelType_int8 = 0,
  PixelType_uint8 = 1,
  PixelType_int32 = 2,
  PixelType_uint32 = 3,
  PixelType_int64 = 4,
  PixelType_uint64 = 5,
  PixelType_float32 = 6,
  PixelType_float64 = 7
};

size_t pixeltype_size(const PixelType type);

std::string pixeltype_string(const PixelType type);

} // namespace pinkie

#endif // PINKIE_IMAGE_SRC_PIXEL_TYPE_H