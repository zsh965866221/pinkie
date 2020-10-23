#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/pixel_type.h"

#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace pinkie;

int main() {
  CALL_DTYPE(PixelType_float32, type, [&] () {
    std::cout << "A" << std::endl;
  });
  return 0;
}
