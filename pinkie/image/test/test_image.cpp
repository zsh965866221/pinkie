#include "pinkie/image/csrc/frame.h"
#include "pinkie/image/csrc/pixel_type.h"

#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace pinkie;

int main() {
  int b[9] = {1,2,3,4,5,6,7,8,9};
  auto s = Eigen::Map<Eigen::Matrix3i>(b);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << s(i, j) << std::endl;
    }
  }
  return 0;
}
