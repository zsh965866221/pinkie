#ifndef PINKIE_IMAGE_PYWRAPPERS_FRAME_H
#define PINKIE_IMAGE_PYWRAPPERS_FRAME_H

#ifdef __cplusplus
extern "C" {
#endif

void* frame_new();
void* frame_clone(void* ptr);
void frame_delete(void* ptr);
void frame_origin(void* ptr, float* out);
void frame_spacing(void* ptr, float* out);
void frame_axes(void* ptr, float* out);
void frame_axis(void* ptr, unsigned int index, float* out);


#ifdef __cplusplus
}
#endif

#endif // PINKIE_IMAGE_PYWRAPPERS_FRAME_H