#include "pinkie/image/csrc/pixel_type.h"

#include <assert.h>

namespace pinkie {

size_t pixeltype_size(const PixelType dtype) {
  size_t size = 0;
  CALL_DTYPE(dtype, [&]() {
    size = sizeof(pixeltype_t);
  });
  return size;
}

std::string pixeltype_string(const PixelType type) {
  switch (type) {
  case PixelType_int8:          return "int8";
  case PixelType_uint8:         return "uint8";
  case PixelType_int32:         return "int32";
  case PixelType_uint32:        return "uint32";
  case PixelType_int64:         return "int64";
  case PixelType_uint64:        return "uint64";
  case PixelType_float32:       return "float32";
  case PixelType_float64:       return "float64";
  default: assert(false);       return "UnKnown";
  }
}

size_t pixeltype_byte_size(
  const PixelType type, 
  size_t h, size_t w, size_t d
) {
  return pixeltype_size(type) * h * w * d;
}

size_t pixeltype_byte_size(
  const PixelType type, 
  const Eigen::Vector3i& size
) {
  return pixeltype_byte_size(
    type,
    static_cast<size_t>(size(0)),
    static_cast<size_t>(size(1)),
    static_cast<size_t>(size(2))
  );
}

} // namespace pinkie