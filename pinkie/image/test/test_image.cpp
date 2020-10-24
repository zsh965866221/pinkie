#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/pixel_type.h"

#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace pinkie;

int main() {
  CALL_DTYPE(PixelType_float32, type, [&]() {
  CALL_DTYPE(PixelType_int32, type1, [&]() {
    type a;
    type1 b;
    std::cout << "B" << std::endl;
  });});

  auto a = Eigen::Vector3i(1,2,3);
  return 0;
}
