#include "pinkie/image/csrc/pixel_type.h"

#include <assert.h>

namespace pinkie {

size_t pixeltype_size(const PixelType type) {
  switch (type) {
  case PixelType_int8:          return sizeof(char);
  case PixelType_uint8:         return sizeof(unsigned char);
  case PixelType_int32:         return sizeof(int);
  case PixelType_uint32:        return sizeof(unsigned int);
  case PixelType_int64:         return sizeof(long);
  case PixelType_uint64:        return sizeof(unsigned long);
  case PixelType_float32:       return sizeof(float);
  case PixelType_float64:       return sizeof(double);
  default: assert(false);       return 0;
  }
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

} // namespace pinkie