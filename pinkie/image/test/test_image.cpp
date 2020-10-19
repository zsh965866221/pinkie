#include "pinkie/image/csrc/frame.h"

#include <iostream>

using namespace pinkie;

int main() {
  auto a = torch::tensor({1,2,3}, torch::kFloat);
  Frame frame;
  frame.set_axis(a, 1);
  std::cout << frame.axis(1) << std::endl;
  return 0;
}
