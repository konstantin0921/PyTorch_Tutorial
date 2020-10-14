import torch

# x = torch.randn(3, requires_grad=True)
# print(x)

# # build computiation graph
# y = x + 2
# print(y)
# z = y * y * 2
# z = z.mean()
# print(z)

# z.backward()  # dz/dx
# print(x.grad)


# 3 ways to cancel the gradient
# x.requires_grad(False)
# x.detach()
# with torch.no_grad():

# x.requires_grad_(False)
# print(x)

# y = x.detach()
# print(y)

# with torch.no_grad():
#     y = x + 2
#     print(y)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)
    # caution
    weights.grad.zero_()
