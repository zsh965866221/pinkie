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

  const auto& frame = image->frame();

  Frame* out = new Frame(frame);
  return out;
}

void image_set_frame(void* ptr, void* in) {
  assert(ptr != nullptr);
  assert(in != nullptr);
  Image* image = static_cast<Image*>(ptr);
  Frame* frame = static_cast<Frame*>(in);
  image->set_frame(*frame);
}

void* image_data(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  return image->data<void>();
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
  return static_cast<PixelType>(image->dtype());
}

void* image_cast(void* ptr, int dtype) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  return image->cast(static_cast<PixelType>(dtype));
}

void image_cast_(void* ptr, int dtype) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  image->cast_(static_cast<PixelType>(dtype));
}
