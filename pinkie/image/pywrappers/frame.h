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

void frame_set_origin(void* ptr, float* src);
void frame_set_spacing(void* ptr, float* src);
void frame_set_axes(void* ptr, float* src);
void frame_set_axis(void* ptr, float* src, unsigned int index);

void frame_world_to_voxel(void* ptr, float* src, float* out);
void frame_voxel_to_world(void* ptr, float* src, float* out);

#ifdef __cplusplus
}
#endif

#endif // PINKIE_IMAGE_PYWRAPPERS_FRAME_H