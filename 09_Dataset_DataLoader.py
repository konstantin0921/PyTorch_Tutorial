import torch
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt("./data/wine.csv", delimiter=",", skiprows=1, dtype=np.float32)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]  # n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=None)

# dataset = WineDataset(transform=ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

first = dataset[0]
features, lables = first

print(type(features))
print(features)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first = dataset[0]
features, lables = first

print(type(features))
print(features)

