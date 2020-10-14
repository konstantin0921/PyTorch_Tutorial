import torch
import numpy as np

# x = torch.empty(3)
# print(x)
# print(x.dtype)


# x = torch.ones(2, 2, dtype=torch.float16)
# print(x)
# print(x.size())


# tensor from a list
# x = torch.tensor([2.5, 1.2])
# print(x)

# basic operations
# x = torch.rand(2, 2)
# y = torch.rand(2, 2)
# print(x)
# print(y)
# z = x + y
# z = torch.add(x, y)
# print(z)

# inplace operation ends with trainling_
# y.add_(x)
# print(y)


# slicing operations
# x = torch.rand(5, 3)
# print(x)

# print(x[:, 0])  # all rows and 0th column
# print(x[1, 1])
# print(x[1, 1].size())
# # if the tensor is a scalar, one can get the actual value with .item()
# print(x[1, 1].item())


# reshape
# x = torch.rand(4, 4)
# print(x)
# y = x.view(-1, 8)
# print(y)

# convert to numpy and vice versa
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# print(type(b))

# a.add_(1)  # if data is on CPU, they share the same storage
# print(a)
# print(b)

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# a += 1
# print(b)

print(torch.cuda.is_available())  # Mac OS, False

x = torch.ones(5, requires_grad=True)
print(x)
