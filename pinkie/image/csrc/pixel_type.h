#ifndef PINKIE_IMAGE_SRC_PIXEL_TYPE_H
#define PINKIE_IMAGE_SRC_PIXEL_TYPE_H

#include <Eigen/Eigen>
#include <functional>
#include <string>

namespace pinkie {

enum PixelType {
  PixelType_int8    = 0,
  PixelType_uint8   = 1,
  PixelType_int16   = 2,
  PixelType_uint16  = 3,
  PixelType_int32   = 4,
  PixelType_uint32  = 5,
  PixelType_int64   = 6,
  PixelType_uint64  = 7,
  PixelType_float32 = 8,
  PixelType_float64 = 9
};

size_t pixeltype_size(const PixelType type);

std::string pixeltype_string(const PixelType type);

size_t pixeltype_byte_size(
  const PixelType type, 
  const Eigen::Vector3i& size
);

size_t pixeltype_byte_size(
  const PixelType type, 
  size_t h, size_t w, size_t d
);


#define CALL_CASE_DTYPE(dtype, type, name, ...)                           \
  case dtype: {                                                           \
    typedef type name;                                                    \
    __VA_ARGS__();                                                        \
    break;                                                                \
  }


#define CALL_DTYPE(dtype, name, ...)                                      \
  switch (dtype) {                                                        \
  CALL_CASE_DTYPE(PixelType_int8, char, name, __VA_ARGS__);               \
  CALL_CASE_DTYPE(PixelType_uint8, unsigned char, name, __VA_ARGS__);     \
  CALL_CASE_DTYPE(PixelType_int16, short, name, __VA_ARGS__);             \
  CALL_CASE_DTYPE(PixelType_uint16, unsigned short, name, __VA_ARGS__);   \
  CALL_CASE_DTYPE(PixelType_int32, int, name, __VA_ARGS__);               \
  CALL_CASE_DTYPE(PixelType_uint32, unsigned int, name, __VA_ARGS__);     \
  CALL_CASE_DTYPE(PixelType_int64, long, name, __VA_ARGS__);              \
  CALL_CASE_DTYPE(PixelType_uint64, unsigned long, name, __VA_ARGS__);    \
  CALL_CASE_DTYPE(PixelType_float32, float, name, __VA_ARGS__);           \
  CALL_CASE_DTYPE(PixelType_float64, double, name, __VA_ARGS__);          \
  default:  assert(false);                                                \
  }                                                                       


} // namespace pinkie
#endif // PINKIE_IMAGE_SRC_PIXEL_TYPE_H