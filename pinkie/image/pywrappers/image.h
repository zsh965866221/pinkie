#ifndef PINKIE_IMAGE_PYWRAPPERS_IMAGE_H
#define PINKIE_IMAGE_PYWRAPPERS_IMAGE_H

#include "pinkie/utils/csrc/header.h"

#ifdef __cplusplus
extern "C" {
#endif

PINKIE_API void* image_new(int dtype, bool is_2d);
PINKIE_API void* image_clone(void* ptr, bool copy);
PINKIE_API void* image_new_owned(
  int height, int width, int depth, 
  int dtype, bool is_2d
);
PINKIE_API void image_delete(void* ptr);

PINKIE_API void image_size(void* ptr, int* out);
PINKIE_API void* image_frame(void* ptr);
PINKIE_API void image_set_frame(void* ptr, void* in);

PINKIE_API void* image_data(void* ptr);
PINKIE_API void image_set_data(
  void* ptr, void* data,
  int height, int width, int depth, 
  int dtype, bool is_2d, bool copy
);
PINKIE_API void image_set_zero(void* ptr);

PINKIE_API bool image_is_2d(void* ptr);
PINKIE_API void image_set_2d(void* ptr, bool p);

PINKIE_API int image_dtype(void* ptr);
PINKIE_API void* image_cast(void* ptr, int dtype);
PINKIE_API void image_cast_(void* ptr, int dtype);


#ifdef __cplusplus
}
#endif

#endif // PINKIE_IMAGE_PYWRAPPERS_FRAME_H