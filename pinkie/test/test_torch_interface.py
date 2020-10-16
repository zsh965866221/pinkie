import torch

torch.ops.load_library("pinkie/lib/Release/libpinkie_python_torch.so")

a = torch.randn(2, 2).to('cuda')
b = torch.randn(2, 2).to('cuda')
print(a)
print(b)
print(torch.ops.my_ops.tensor_add(a, b, 2))
