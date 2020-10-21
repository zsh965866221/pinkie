#include "pinkie/image/pywrappers/frame.h"
#include "pinkie/image/csrc/frame.h"

using namespace pinkie;

void* frame_new() {
  return new Frame();
}

void* frame_clone(void* ptr) {
  assert(ptr != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  return new Frame(*frame);
}

void frame_delete(void* ptr) {
  assert(ptr != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  delete frame;
}

void frame_origin(void* ptr, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  
  const auto& origin = frame->origin();
  const float* src_ptr = origin.data();
  memcpy(out, src_ptr, sizeof(float) * 3);
}

void frame_spacing(void* ptr, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  
  const auto& spacing = frame->spacing();
  const float* src_ptr = spacing.data();
  memcpy(out, src_ptr, sizeof(float) * 3);
}

void frame_axes(void* ptr, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  
  const auto& axes = frame->axes();
  const float* src_ptr = axes.data();
  memcpy(out, src_ptr, sizeof(float) * 3 * 3);
}

void frame_axis(void* ptr, unsigned int index, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  
  const auto& axis = frame->axis(index);
  const float* src_ptr = axis.data();
  memcpy(out, src_ptr, sizeof(float) * 3);
}