#include <iostream>

#include <torch/torch.h>

int main(int argc, char* argv[]) {
  torch::Tensor tensor = torch::randn({10, 10});
  auto a = tensor.to(torch::kCUDA);
  std::cout << tensor << std::endl;
  std::cout << a << std::endl;
  std::cout << a.inverse() << std::endl;

  return 0;
}