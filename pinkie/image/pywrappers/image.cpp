#include "pinkie/image/pywrappers/image.h"
#include "pinkie/image/csrc/image.h"


using namespace pinkie;

void* image_new(int dtype, bool is_2d) {
  return new Image(
    static_cast<PixelType>(dtype),
    is_2d
  );
}

void* image_clone(void* ptr, bool copy) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  return new Image(*image, copy);
}

void* image_new_owned(
  int height, int width, int depth, 
  int dtype, bool is_2d
) {
  return new Image(
    height, width, depth, 
    static_cast<PixelType>(dtype),
    is_2d
  );
}

void image_delete(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  delete image;
}

void image_size(void* ptr, int* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);
  Image* image = static_cast<Image*>(ptr);

  const auto& size = image->size();
  const int* src = size.data();
  memcpy(out, src, sizeof(int) * 3);
}

void* image_frame(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);

  Frame* frame = &(image->frame());
  return frame;
}

void image_set_frame(void* ptr, void* in) {
  assert(ptr != nullptr);
  assert(in != nullptr);
  Image* image = static_cast<Image*>(ptr);
  Frame* frame = static_cast<Frame*>(in);
  image->set_frame(*frame);
}

void image_data(void* ptr, void* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);
  Image* image = static_cast<Image*>(ptr);
  CALL_DTYPE(
    image->dtype(), type,
    [&]() {
      type* dst_ptr = static_cast<type*>(out);
      type* src_ptr = image->data<type>();
      size_t bytes_size = image->bytes_size();
      memcpy(dst_ptr, src_ptr, bytes_size);
    } 
  );
}

void image_set_data(
  void* ptr, void* data,
  int height, int width, int depth, 
  int dtype, bool is_2d, bool copy
) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  image->set_data(
    data,
    height,
    width,
    depth,
    static_cast<PixelType>(dtype),
    is_2d,
    copy
  );
}

void image_set_zero(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  image->set_zero();
}

bool image_is_2d(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  return image->is_2d();
}

void image_set_2d(void* ptr, bool p) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  image->set_2d(p);
}

int image_dtype(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  int dtype = static_cast<int>(image->dtype());
  return dtype;
}

void* image_cast(void* ptr, int dtype) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  Image* new_image = image->cast(static_cast<PixelType>(dtype));
  return static_cast<void*>(new_image);
}

void image_cast_(void* ptr, int dtype) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  image->cast_(static_cast<PixelType>(dtype));
}


void image_origin(void* ptr, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Image* image = static_cast<Image*>(ptr);
  
  const auto& origin = image->frame().origin();
  const float* src_ptr = origin.data();
  memcpy(out, src_ptr, sizeof(float) * 3);
}

void image_spacing(void* ptr, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Image* image = static_cast<Image*>(ptr);
  
  const auto& spacing = image->frame().spacing();
  const float* src_ptr = spacing.data();
  memcpy(out, src_ptr, sizeof(float) * 3);
}

void image_axes(void* ptr, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Image* image = static_cast<Image*>(ptr);
  
  const auto& axes = image->frame().axes();
  const float* src_ptr = axes.data();
  memcpy(out, src_ptr, sizeof(float) * 3 * 3);
}

void image_axis(void* ptr, unsigned int index, float* out) {
  assert(ptr != nullptr);
  assert(out != nullptr);

  Image* image = static_cast<Image*>(ptr);
  
  const auto& axis = image->frame().axis(index);
  const float* src_ptr = axis.data();
  memcpy(out, src_ptr, sizeof(float) * 3);
}

void image_set_origin(void* ptr, float* src) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Image* image = static_cast<Image*>(ptr);
  image->frame().set_origin(Eigen::Map<Eigen::Vector3f>(src));
}

void image_set_spacing(void* ptr, float* src) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Image* image = static_cast<Image*>(ptr);
  image->frame().set_spacing(Eigen::Map<Eigen::Vector3f>(src));
}

void image_set_axes(void* ptr, float* src) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Image* image = static_cast<Image*>(ptr);
  image->frame().set_axes(Eigen::Map<Eigen::Matrix3f>(src));
}

void image_set_axis(void* ptr, float* src, unsigned int index) {
  assert(ptr != nullptr);
  assert(src != nullptr);

  Image* image = static_cast<Image*>(ptr);
  image->frame().set_axis(Eigen::Map<Eigen::Vector3f>(src), static_cast<size_t>(index));
}

void image_world_to_voxel(void* ptr, float* src, float* out) {
  assert(ptr != nullptr);
  assert(src != nullptr);
  assert(out != nullptr);

  Image* image = static_cast<Image*>(ptr);
  auto voxel = image->frame().world_to_voxel(
    Eigen::Map<Eigen::Vector3f>(src)
  );
  memcpy(out, voxel.data(), sizeof(float) * 3);
}

void image_voxel_to_world(void* ptr, float* src, float* out) {
  assert(ptr != nullptr);
  assert(src != nullptr);
  assert(out != nullptr);

  Image* image = static_cast<Image*>(ptr);
  auto world = image->frame().voxel_to_world(
    Eigen::Map<Eigen::Vector3f>(src)
  );
  memcpy(out, world.data(), sizeof(float) * 3);
}
