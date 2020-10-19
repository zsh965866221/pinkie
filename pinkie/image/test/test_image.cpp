#include "pinkie/image/csrc/frame.h"

#include <iostream>

using namespace pinkie;

int main() {
  Frame frame;
  std::cout << frame.axis(1) << std::endl;
  return 0;
}
