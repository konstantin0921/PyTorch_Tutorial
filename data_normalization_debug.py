import torch

# import torch.nn as nn

# import torch.optim as optim
# import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

mean = 0.28604066371917725
std = 0.3530242443084717

train_set_normal = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),  # transform the PIL image to tensor
            # normalize
            transforms.Normalize(mean, std),
        ]
    ),
)

# breakpoint here
loader = DataLoader(train_set_normal, batch_size=1)
# breakpoint here
image, label = next(iter(loader))

print(image.shape)
