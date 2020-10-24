#ifndef PINKIE_TRANSFORM_PYWRAPPERS_TRANSFORM_H
#define PINKIE_TRANSFORM_PYWRAPPERS_TRANSFORM_H

#include "pinkie/utils/csrc/header.h"

#ifdef __cplusplus
extern "C" {
#endif

PINKIE_API void transform_rotate(
  float* axis, 
  float theta_radian, 
  float* out
);

#ifdef __cplusplus
}
#endif

#endif // PINKIE_TRANSFORM_PYWRAPPERS_TRANSFORM_H