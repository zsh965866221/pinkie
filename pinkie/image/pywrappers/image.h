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

PINKIE_API void image_data(void* ptr, void* out);
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

PINKIE_API void image_origin(void* ptr, float* out);
PINKIE_API void image_spacing(void* ptr, float* out);
PINKIE_API void image_axes(void* ptr, float* out);
PINKIE_API void image_axis(void* ptr, unsigned int index, float* out);

PINKIE_API void image_set_origin(void* ptr, float* src);
PINKIE_API void image_set_spacing(void* ptr, float* src);
PINKIE_API void image_set_axes(void* ptr, float* src);
PINKIE_API void image_set_axis(void* ptr, float* src, unsigned int index);

PINKIE_API void image_world_to_voxel(void* ptr, float* src, float* out);
PINKIE_API void image_voxel_to_world(void* ptr, float* src, float* out);


#ifdef __cplusplus
}
#endif

#endif // PINKIE_IMAGE_PYWRAPPERS_FRAME_H