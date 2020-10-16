import pinkie
import torch
import pinkie_python_test.test_lib as lib

print(lib.add(1, 2))
print(lib.mul(2, 3))

a = torch.randn(2, 3).to('cuda:2')
b = torch.randn(3, 4).to('cuda:2')
print(a)
print(b)
print(lib.tensor_mul(a, b, 10.2))
print(torch.mm(a, b) + 10.2)
print(lib.mat_mul_cuda(a, b) + 10.2)

print('-' * 100)

a = torch.randint(0, 10, (2, 3))
b = torch.randint(0, 10, (3, 4))
print(torch.mm(a, b))
print(lib.mat_mul_cuda(a.to('cuda'), b.to('cuda')))


