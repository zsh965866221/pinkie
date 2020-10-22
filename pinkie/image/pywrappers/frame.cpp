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

void frame_set_origin(void* ptr, float* src) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  frame->set_origin(Eigen::Map<Eigen::Vector3f>(src));
}

void frame_set_spacing(void* ptr, float* src) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  frame->set_spacing(Eigen::Map<Eigen::Vector3f>(src));
}

void frame_set_axes(void* ptr, float* src) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  frame->set_axes(Eigen::Map<Eigen::Matrix3f>(src));
}

void frame_set_axis(void* ptr, float* src, unsigned int index) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  frame->set_axis(Eigen::Map<Eigen::Vector3f>(src), static_cast<size_t>(index));
}

void frame_world_to_voxel(void* ptr, float* src, float* out) {
  assert(ptr != nullptr);
  assert(src != nullptr);
  assert(out != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  auto voxel = frame->world_to_voxel(
    Eigen::Map<Eigen::Vector3f>(src)
  );
  memcpy(out, voxel.data(), sizeof(float) * 3);
}

void frame_voxel_to_world(void* ptr, float* src, float* out) {
  assert(ptr != nullptr);
  assert(src != nullptr);
  assert(out != nullptr);

  Frame* frame = static_cast<Frame*>(ptr);
  auto world = frame->voxel_to_world(
    Eigen::Map<Eigen::Vector3f>(src)
  );
  memcpy(out, world.data(), sizeof(float) * 3);
}
