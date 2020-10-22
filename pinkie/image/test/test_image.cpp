#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/pixel_type.h"

#include <iostream>

using namespace pinkie;

int main() {
  PixelType a = PixelType_int64;
  std::cout << pixeltype_size(a) << std::endl;
  return 0;
}
