#include "pinkie/image/csrc/frame.h"

#include <iostream>

using namespace pinkie;

int main() {
  Frame frame;
  frame.set_axis(Eigen::Vector3f::Random(), 1);
  std::cout << frame.axes() << std::endl;

  auto a = frame.origin();
  a(0) = 10;


  auto b = a.array();
  b(1) = 10;
  std::cout << a << std::endl;
  std::cout << b << std::endl;
  return 0;
}
