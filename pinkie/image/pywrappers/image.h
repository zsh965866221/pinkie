#ifndef PINKIE_IMAGE_PYWRAPPERS_IMAGE_H
#define PINKIE_IMAGE_PYWRAPPERS_IMAGE_H

#include "pinkie/utils/csrc/header.h"

#ifdef __cplusplus
extern "C" {
#endif

PINKIE_API void* image_new(int dtype, bool is_2d);
PINKIE_API void* image_clone(void* ptr, bool copy);
PINKIE_API void image_delete(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // PINKIE_IMAGE_PYWRAPPERS_FRAME_H