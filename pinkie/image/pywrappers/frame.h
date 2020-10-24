#ifndef PINKIE_IMAGE_PYWRAPPERS_FRAME_H
#define PINKIE_IMAGE_PYWRAPPERS_FRAME_H

#include "pinkie/utils/csrc/header.h"

#ifdef __cplusplus
extern "C" {
#endif

PINKIE_API void* frame_new();
PINKIE_API void* frame_clone(void* ptr);
PINKIE_API void frame_delete(void* ptr);

PINKIE_API void frame_origin(void* ptr, float* out);
PINKIE_API void frame_spacing(void* ptr, float* out);
PINKIE_API void frame_axes(void* ptr, float* out);
PINKIE_API void frame_axis(void* ptr, unsigned int index, float* out);

PINKIE_API void frame_set_origin(void* ptr, float* src);
PINKIE_API void frame_set_spacing(void* ptr, float* src);
PINKIE_API void frame_set_axes(void* ptr, float* src);
PINKIE_API void frame_set_axis(void* ptr, float* src, unsigned int index);

PINKIE_API void frame_world_to_voxel(void* ptr, float* src, float* out);
PINKIE_API void frame_voxel_to_world(void* ptr, float* src, float* out);

#ifdef __cplusplus
}
#endif

#endif // PINKIE_IMAGE_PYWRAPPERS_FRAME_H