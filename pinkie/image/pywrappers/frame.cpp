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