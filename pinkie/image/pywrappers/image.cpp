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

void image_delete(void* ptr) {
  assert(ptr != nullptr);
  Image* image = static_cast<Image*>(ptr);
  delete image;
}
